"""
Knowledge Graph RAG Agent (Optimized).

Optimizations over original 5-node pipeline:
① Cross-Encoder Rerank  — replaces LLM Listwise (10-40x faster, deterministic)
② Structured 2-hop Subgraph Retrieval — replaces naive regex entity expansion
③ Parallel Pipeline — asyncio.gather for independent retrieval phases
④ Graph-Aware Context Fusion — structured prompt linking paths to documents
⑤ Multi-hop Query Decomposition — explicit sub-question breakdown
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from bears.agents.base import AgentCapability, AgentResponse, BaseRAGAgent
from bears.agents.kg_agent.retrievers import GraphRetriever, VectorRetriever
from bears.core.config import get_settings
from bears.core.experiment import ExperimentConfig
from bears.database.graph.graph_store import GraphStoreManager
from bears.database.vector.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class KGAgent(BaseRAGAgent):
    """Knowledge-graph augmented RAG agent (Optimized)."""

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()

        self._vector_store = VectorStoreManager()
        self._graph_store = GraphStoreManager()
        self._vector_retriever = VectorRetriever(self._vector_store)
        self._graph_retriever = GraphRetriever(self._graph_store, llm_model=self.exp.model)
        self._llm = ChatOpenAI(
            model=self.exp.model,
            temperature=self.exp.temperature,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # ① Lazy-init cross-encoder reranker
        self._cross_encoder = None
        if self.exp.use_cross_encoder:
            try:
                from bears.agents.kg_agent.reranker import CrossEncoderReranker
                self._cross_encoder = CrossEncoderReranker(self.exp.reranker_model)
                logger.info("Cross-Encoder reranker loaded")
            except Exception as e:
                logger.warning(f"Cross-Encoder load failed, will use LLM fallback: {e}")

    @property
    def name(self) -> str:
        return "kg"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {AgentCapability.VECTOR_SEARCH, AgentCapability.GRAPH_SEARCH, AgentCapability.MULTI_HOP}

    # ================================================================
    # ⑤ Node 1: Multi-hop Query Decomposition + Expansion
    # ================================================================

    def _decompose_and_expand(self, question: str) -> Tuple[List[str], bool]:
        """Decompose multi-hop questions into sub-questions, then expand."""
        try:
            decompose_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是查詢優化專家，擅長分解多步驟問題。

任務:
1. 判斷問題是否是多跳問題（需要多步推理）
2. 如果是多跳問題，將問題拆解為 2-3 個子問題（按推理順序排列）
3. 如果是簡單問題，產生 2 個有助於檢索的查詢變體

輸出格式:
- 第一行: MULTI 或 SINGLE（標記問題類型）
- 後續每行一個查詢/子問題，不要編號

範例1 (多跳題):
問題: A的父親是在哪一年出生的？
MULTI
A的父親是誰
A的父親出生年份

範例2 (簡單題):
問題: 台灣第一座國家公園是哪座？
SINGLE
台灣國家公園歷史
台灣最早成立的國家公園名稱
"""),
                ("human", "問題：{question}"),
            ])
            chain = decompose_prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )

            lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
            is_multi_hop = False
            queries = [question]

            if lines:
                is_multi_hop = "MULTI" in lines[0].upper()
                for line in lines[1:]:
                    if line and line not in queries:
                        queries.append(line)

            return queries[:4], is_multi_hop

        except Exception as e:
            logger.error(f"Query decomposition error: {e}")
            return [question], False

    # ================================================================
    # Node 2: Vector Retrieval (clean, no naive graph expansion)
    # ================================================================

    def _retrieve_vector(self, expanded_queries: List[str], k_per_query: int = 15) -> List[Dict[str, Any]]:
        """Multi-query vector retrieval without graph-expansion noise."""
        all_candidates = []
        seen_ids: Set[str] = set()

        for query in expanded_queries:
            docs_with_meta = self._vector_retriever.retrieve_with_metadata(query, k=k_per_query)
            for doc_meta in docs_with_meta:
                doc_id = doc_meta.get("metadata", {}).get("doc_id")
                if doc_id and doc_id not in seen_ids:
                    all_candidates.append({
                        "doc_id": doc_id,
                        "content": doc_meta.get("content", ""),
                        "metadata": doc_meta.get("metadata", {}),
                    })
                    seen_ids.add(doc_id)

        return all_candidates[:30]

    # ================================================================
    # ② Node 3: Structured Subgraph Retrieval
    # ================================================================

    def _retrieve_subgraph(self, question: str) -> tuple:
        """Retrieve structured 1-hop + 2-hop subgraph paths.
        
        Returns: (paths, graph_doc_ids, graph_entities)
        """
        try:
            if self.exp.graph_expansion_hops >= 2:
                paths = self._graph_retriever.retrieve_2hop_paths(
                    question, max_entities=3, limit_per_entity=15
                )
            else:
                triplets = self._graph_retriever.retrieve(
                    question, max_entities=3, max_relations_per_entity=10
                )
                paths = [{"path_str": t} for t in triplets]

            graph_doc_ids = set()
            graph_entities: set = set()
            for p in paths:
                for key in ("mid_doc_id", "end_doc_id", "rel_doc_id", "neighbor_doc_id"):
                    doc_id = p.get(key)
                    if doc_id:
                        graph_doc_ids.add(doc_id)
                # Collect intermediate & end entities for secondary search
                for ent_key in ("mid", "end", "start"):
                    val = p.get(ent_key)
                    if val and len(str(val)) > 2:  # skip noise like 'a', 'of'
                        graph_entities.add(str(val))

            return paths, list(graph_doc_ids), list(graph_entities)

        except Exception as e:
            logger.warning(f"Subgraph retrieval error: {e}")
            return [], [], []

    def _retrieve_by_entities(
        self, entities: List[str], k_per_entity: int = 5
    ) -> List[Dict[str, Any]]:
        """Secondary vector search using graph-extracted entity names.
        
        This fills retrieval gaps in 2-hop queries: when the original question
        retrieves docs about Entity A but misses docs about Entity B (the next hop),
        searching directly for Entity B's name finds those missing docs.
        """
        all_docs: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for entity in entities[:6]:  # cap at 6 entities
            try:
                docs = self._vector_retriever.retrieve_with_metadata(entity, k=k_per_entity)
                for doc in docs:
                    doc_id = doc.get("metadata", {}).get("doc_id")
                    if doc_id and doc_id not in seen_ids:
                        all_docs.append({
                            "doc_id": doc_id,
                            "content": doc.get("content", ""),
                            "metadata": doc.get("metadata", {}),
                        })
                        seen_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"Entity-based retrieval failed for '{entity}': {e}")

        logger.debug(f"Entity-based retrieval added {len(all_docs)} extra candidates")
        return all_docs

    # ================================================================
    # ① Node 4: Cross-Encoder Rerank
    # ================================================================

    def _rerank(self, question: str, candidates: List[Dict]) -> Tuple[List[str], List[str]]:
        """Rerank with Cross-Encoder; fall back to LLM Listwise if unavailable."""
        if not candidates:
            return [], []

        top_k = self.exp.top_k

        if self._cross_encoder is not None:
            try:
                contents, ids = self._cross_encoder.rerank(question, candidates, top_k=top_k)
                if contents:
                    logger.debug(f"Cross-Encoder reranked {len(candidates)} -> {len(contents)}")
                    return contents, ids
            except Exception as e:
                logger.warning(f"Cross-Encoder failed, falling back to LLM: {e}")

        return self._rerank_llm_fallback(question, candidates, top_k)

    def _rerank_llm_fallback(self, question: str, candidates: List[Dict], top_k: int) -> Tuple[List[str], List[str]]:
        """LLM listwise reranking fallback (improved: 600 chars vs original 300)."""
        try:
            candidate_texts = [
                f"[{i}] {c['content'][:600]}"
                for i, c in enumerate(candidates[:20], 1)
            ]
            candidates_str = "\n\n".join(candidate_texts)

            rerank_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一個文檔排序專家。根據問題選出最相關的文檔，只返回文檔編號用逗號分隔，不要其他說明。"),
                ("human", "問題：{question}\n\n候選文檔：\n{candidates}\n\n請選出最相關的 {top_k} 篇："),
            ])
            chain = rerank_prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question, "candidates": candidates_str, "top_k": top_k},
                config={"callbacks": callbacks} if callbacks else {},
            )

            found_numbers = re.findall(r'\d+', response.content)
            selected_indices = [int(n) - 1 for n in found_numbers if 1 <= int(n) <= len(candidates)]

            reranked_contents, reranked_ids = [], []
            for idx in selected_indices[:top_k]:
                if 0 <= idx < len(candidates):
                    reranked_contents.append(candidates[idx]["content"])
                    doc_id = candidates[idx].get("doc_id")
                    if doc_id:
                        reranked_ids.append(doc_id)

            if not reranked_contents and candidates:
                reranked_contents = [c["content"] for c in candidates[:top_k]]
                reranked_ids = [c.get("doc_id") for c in candidates[:top_k] if c.get("doc_id")]

            return reranked_contents, reranked_ids

        except Exception as e:
            logger.error(f"LLM Rerank fallback error: {e}")
            return [c["content"] for c in candidates[:top_k]], [
                c.get("doc_id") for c in candidates[:top_k] if c.get("doc_id")
            ]

    # ================================================================
    # ④ Node 5: Graph-Aware Context Fusion
    # ================================================================

    def _fuse_context(
        self,
        reranked_contents: List[str],
        graph_paths: List[Dict],
        is_multi_hop: bool,
    ) -> str:
        """Fuse vector docs and graph paths into a structured prompt.

        Key: keep paths to max 5 to avoid overwhelming the LLM.
        """
        parts = []

        # Only include top 5 most relevant graph paths (reduced from 15)
        if graph_paths:
            path_lines = [f"  • {p['path_str']}" for p in graph_paths[:5] if p.get("path_str")]
            if path_lines:
                parts.append("【知識圖譜推理路徑】\n" + "\n".join(path_lines))

        if reranked_contents:
            doc_lines = [f"[文檔 {i}]\n{c}" for i, c in enumerate(reranked_contents, 1)]
            parts.append("【檢索文檔】\n" + "\n\n".join(doc_lines))

        return "\n\n".join(parts) if parts else "無相關上下文"

    # ================================================================
    # Node 6: Answer Generation
    # ================================================================

    def _generate_answer(
        self, question: str, fused_context: str, sub_questions: Optional[List[str]] = None
    ) -> str:
        """Generate final answer from fused context, with optional CoT guidance."""

        # Build CoT sub-question hint for multi-hop questions
        cot_hint = ""
        if sub_questions and len(sub_questions) > 1:
            sub_q_lines = "\n".join(f"  {i}. {sq}" for i, sq in enumerate(sub_questions, 1))
            cot_hint = f"\n\n推理步驟提示：\n{sub_q_lines}\n請依序從上下文找到每個子問題的答案，最後給出最終答案。"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一個專業的繁體中文問答助手,擅長從知識圖譜與文件中進行多步驟推理。

回答原則:
1. 上下文包含兩部分:【知識圖譜推理路徑】顯示實體之間的關聯三元組;【檢索文檔】提供原文事實。請結合兩者推理。
2. 多跳問題(問題涉及「A 的 B 的 C」一類關聯)請依推理步驟提示逐步比對實體,確認每一跳都有依據再給出最終答案。
3. 回答時請以 [文檔 i] 標註所引用的文檔編號;若答案來自圖譜路徑,請以 [路徑] 標註。
4. 對於事實性問題請精簡作答(常為一兩個詞或一個數字),不需要冗長解釋。
5. 僅當上下文中「完全找不到任何推理線索」時,才輸出:「根據提供的資訊無法回答此問題」,並簡述缺少哪類資訊。
6. 不得使用先驗知識,僅以提供的上下文為依據。
7. 全程使用繁體中文回答。"""),
            ("human", """【上下文】
{context}

【問題】
{question}{cot_hint}

請依據上述原則,直接給出答案:"""),
        ])

        try:
            chain = prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"context": fused_context, "question": question, "cot_hint": cot_hint},
                config={"callbacks": callbacks} if callbacks else {},
            )

            return response.content.strip()
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"Error generating answer: {e}"

    # ================================================================
    # ③ Main entry point — Parallel Pipeline
    # ================================================================

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        exp = experiment or self.exp

        try:
            total_start = time.time()

            # Phase 1: Query Decomposition (single LLM call)
            expanded_queries, is_multi_hop = self._decompose_and_expand(question)

            # Phase 2: Parallel retrieval (vector + subgraph are independent)
            retrieval_start = time.time()
            loop = asyncio.get_event_loop()

            candidates, subgraph_result = await asyncio.gather(
                loop.run_in_executor(None, self._retrieve_vector, expanded_queries),
                loop.run_in_executor(None, self._retrieve_subgraph, question),
            )
            graph_paths, graph_doc_ids, graph_entities = subgraph_result

            # Secondary entity-based retrieval for 2-hop gap filling
            if is_multi_hop and graph_entities:
                entity_docs = await loop.run_in_executor(
                    None, self._retrieve_by_entities, graph_entities
                )
                # Merge into candidates (dedup by doc_id)
                seen_ids = {c["doc_id"] for c in candidates if c.get("doc_id")}
                for ed in entity_docs:
                    if ed.get("doc_id") not in seen_ids:
                        candidates.append(ed)
                        seen_ids.add(ed["doc_id"])
                logger.debug(f"After entity merge: {len(candidates)} total candidates")

            retrieval_time = time.time() - retrieval_start

            # Phase 3: Cross-Encoder Rerank
            rerank_start = time.time()
            reranked_contents, reranked_ids = self._rerank(question, candidates)
            rerank_time = time.time() - rerank_start

            # Only keep reranked top_k IDs for evaluation ranking.
            # Graph doc_ids are used for context enrichment but NOT added
            # to retrieved_doc_ids to avoid inflating the list and hurting MAP.
            all_retrieved_ids = reranked_ids[:exp.top_k]

            # Phase 4: Context Fusion
            fused_context = self._fuse_context(reranked_contents, graph_paths, is_multi_hop)

            # Phase 5: Generation (with sub-question CoT for multi-hop)
            generation_start = time.time()
            sub_questions = expanded_queries[1:] if is_multi_hop else None
            answer = self._generate_answer(question, fused_context, sub_questions=sub_questions)
            generation_time = time.time() - generation_start

            confidence = min(1.0, len(all_retrieved_ids) / max(exp.top_k, 1))

            return AgentResponse(
                answer=answer,
                retrieved_doc_ids=all_retrieved_ids,
                context=reranked_contents,
                confidence=confidence,
                retrieval_time=retrieval_time + rerank_time,
                generation_time=generation_time,
                metadata={
                    "agent": "kg",
                    "num_candidates": len(candidates),
                    "num_graph_paths": len(graph_paths),
                    "is_multi_hop": is_multi_hop,
                    "rerank_method": "cross_encoder" if self._cross_encoder else "llm_listwise",
                    "rerank_time": round(rerank_time, 3),
                    "graph_expansion_hops": exp.graph_expansion_hops,
                },
            )
        except Exception as e:
            logger.error(f"KGAgent failed: {e}")
            return AgentResponse(
                answer=f"KG agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
