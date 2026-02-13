"""
Knowledge Graph RAG Agent.

Ported from archive/GraphRag_hybrid_1/app/services/rag/graph_rag.py.
Wraps the 5-node pipeline logic into a BaseRAGAgent:
  1. query_expansion -> 2. vector retrieval + graph expansion ->
  3. LLM rerank -> 4. graph retrieval -> 5. answer generation
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set

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
    """Knowledge-graph augmented RAG agent.

    Implements the full 5-node pipeline from the archive graph_rag.py.
    """

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

    @property
    def name(self) -> str:
        return "kg"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {AgentCapability.VECTOR_SEARCH, AgentCapability.GRAPH_SEARCH, AgentCapability.MULTI_HOP}

    # ---- Node 1: Query Expansion ----

    def _query_expansion(self, question: str) -> List[str]:
        """Generate 3 query variants for multi-query retrieval."""
        try:
            expansion_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是查詢優化專家,擅長分解多步驟問題。

任務:
1. 識別問題中的核心實體和關係
2. 產生 3 個有助於找到相關文檔的查詢變體
3. 如果是多步驟問題,考慮拆解中間步驟

要求:
- 變體應涵蓋不同角度或中間步驟
- 使用同義詞、實體別名
- 每個問題用換行分隔，不要編號

範例 (多跳題):
原始問題: A的父親是在哪一年出生的？
變體問題:
A的父親是誰
A的父親出生日期
A的家族背景

原始問題: B國的首都在哪？
變體問題:
B國的行政中心位於何處
B國首都名稱
B國政府所在地
"""),
                ("human", "原始問題：{question}"),
            ])
            chain = expansion_prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )

            expanded = [question]
            for line in response.content.strip().split("\n"):
                line = line.strip()
                if line and line not in expanded:
                    expanded.append(line)
            return expanded[:3]
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [question]

    # ---- Node 2: Multi-query vector retrieval + graph expansion ----

    def _retrieve_vector_with_graph_expansion(self, expanded_queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve candidates using multi-query vector search + graph entity expansion."""
        all_candidates = []
        seen_ids: Set[str] = set()
        initial_entities: Set[str] = set()

        for query in expanded_queries:
            docs_with_meta = self._vector_retriever.retrieve_with_metadata(query, k=30)
            for doc_meta in docs_with_meta:
                doc_id = doc_meta.get("metadata", {}).get("doc_id")
                if doc_id and doc_id not in seen_ids:
                    all_candidates.append({
                        "doc_id": doc_id,
                        "content": doc_meta.get("content", ""),
                        "metadata": doc_meta.get("metadata", {}),
                    })
                    seen_ids.add(doc_id)

                    # 從文檔的 entities metadata 中收集實體
                    entities = doc_meta.get("metadata", {}).get("entities", [])
                    if isinstance(entities, list) and entities:
                        initial_entities.update(entities[:3])  # 每篇文檔取前3個實體
                    else:
                        # Fallback: metadata 沒有 entities 時，從內容粗略提取
                        content = doc_meta.get("content", "")
                        if content:
                            chinese_names = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|教授|博士|總統|主席)?', content[:800])
                            english_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', content[:800])
                            potential = list(set(chinese_names[:5] + english_names[:5]))
                            initial_entities.update(potential[:3])

        # Graph expansion
        if initial_entities:
            try:
                related = asyncio.run(
                    self._graph_retriever.get_related_entities(list(initial_entities)[:5], max_neighbors=3)
                )
                for entity in related[:5]:
                    entity_docs = self._vector_retriever.retrieve_with_metadata(entity, k=2)
                    for doc_meta in entity_docs:
                        doc_id = doc_meta.get("metadata", {}).get("doc_id")
                        if doc_id and doc_id not in seen_ids:
                            all_candidates.append({
                                "doc_id": doc_id,
                                "content": doc_meta.get("content", ""),
                                "metadata": doc_meta.get("metadata", {}),
                            })
                            seen_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"Graph expansion failed: {e}")

        return all_candidates[:30]

    # ---- Node 3: LLM Rerank ----

    def _rerank(self, question: str, candidates: List[Dict]) -> tuple[List[str], List[str]]:
        """LLM listwise reranking. Returns (reranked_contents, reranked_ids)."""
        if not candidates:
            return [], []

        try:
            candidate_texts = []
            for idx, cand in enumerate(candidates[:30], 1):
                preview = cand["content"][:300]
                candidate_texts.append(f"[{idx}] {preview}...")

            candidates_str = "\n\n".join(candidate_texts)

            rerank_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一個文檔排序專家。請根據問題，從候選文檔中選出最相關的 5 篇文檔。只需返回文檔編號（例如：1,3,5,8,12），用逗號分隔，不要其他說明。"),
                ("human", "問題：{question}\n\n候選文檔：\n{candidates}"),
            ])
            chain = rerank_prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question, "candidates": candidates_str},
                config={"callbacks": callbacks} if callbacks else {},
            )

            found_numbers = re.findall(r'\d+', response.content)
            selected_indices = []
            for num_str in found_numbers:
                idx = int(num_str)
                if 1 <= idx <= len(candidates):
                    selected_indices.append(idx - 1)

            reranked_contents = []
            reranked_ids = []
            for idx in selected_indices[:5]:
                if 0 <= idx < len(candidates):
                    reranked_contents.append(candidates[idx]["content"])
                    doc_id = candidates[idx].get("doc_id")
                    if doc_id:
                        reranked_ids.append(doc_id)

            if not reranked_contents and candidates:
                reranked_contents = [c["content"] for c in candidates[:5]]
                reranked_ids = [c.get("doc_id") for c in candidates[:5] if c.get("doc_id")]

            return reranked_contents, reranked_ids

        except Exception as e:
            logger.error(f"Rerank error: {e}")
            contents = [c["content"] for c in candidates[:5]]
            ids = [c.get("doc_id") for c in candidates[:5] if c.get("doc_id")]
            return contents, ids

    # ---- Node 4: Graph Retrieval ----

    async def _retrieve_graph(self, question: str) -> List[str]:
        """Retrieve graph context (entity relationships)."""
        try:
            return await self._graph_retriever.retrieve(question, max_entities=3, max_relations_per_entity=10)
        except Exception as e:
            logger.warning(f"Graph retrieval error: {e}")
            return []

    # ---- Node 5: Answer Generation ----

    def _generate_answer(self, question: str, vector_context: List[str], graph_context: List[str]) -> str:
        """Generate final answer from combined vector and graph contexts."""
        context_parts = []
        if vector_context:
            context_parts.append("【向量檢索上下文】\n" + "\n\n".join(vector_context))
        if graph_context:
            context_parts.append("【圖譜檢索上下文】\n" + "\n".join(graph_context))
        context_str = "\n\n".join(context_parts) if context_parts else "無相關上下文"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一個專業的問答助手,擅長整合多篇文檔資訊並進行多步驟邏輯推理。

                回答策略:
                1. 先分析問題結構,識別需要幾個步驟
                2. 對於多步驟問題,依序完成每個步驟:
                   - 第 1 步: 從文檔中找到起點資訊 (如人名、實體)。**注意：必須精確匹配實體名稱，避免混淆同名或相似實體。**
                   - 第 2 步: 用第 1 步的結果在文檔中找中間資訊 (如關係、屬性)
                   - 第 3 步: 用第 2 步的結果找最終答案
                3. 對於涉及數值或比例的問題:
                   - **優先尋找直接答案**：如果文檔中直接提供了數值或時間長度（如「歷時83年」、「佔比20%」），**必須直接引用**，禁止自行計算。
                   - 只有在文檔未直接提供答案時，才根據文檔中的數據進行計算。
                   - 若文檔提供的數據與問題詢問的角度相反，請進行簡單的數學轉換 (如 100% - X%) 以驗證答案。
                4. **邏輯一致性檢查**：
                   - 對於是非題（Yes/No），確保你的結論（是/否）與你的解釋完全一致。
                   - 例如：如果解釋是「A來自德國，B來自美國」，結論必須是「不是」（來自不同國家）。
                5. 整合所有資訊得出結論

                重要原則:
                - **嚴格以文檔為準**：如果文檔中有明確資訊，必須優先使用文檔內容，而非預訓練知識（例如不要使用外部知識補充年份）
                - 即使資訊分散在 2-4 篇不同文檔,也要努力整合
                - 對於關係類問題 (如"繼父"),明確推理關係鏈
                - 遇到數字問題,請精確核對文檔中的數據,不要憑印象回答
                - 只有在上下文完全沒有相關資訊時,才回答「根據提供的資料無法回答此問題」

                **輸出格式要求**：
                請務必按照以下 XML 格式輸出你的思考過程和最終答案：
                <reasoning>
                這裡寫下你的逐步推理過程...
                1. 根據文檔X...
                2. 發現...
                3. 因此...
                </reasoning>
                <answer>
                這裡寫下最終答案（精簡、直接）
                </answer>

                範例參考:

                範例1 - 優先使用文檔數值:
                Q: 某朝代持續了多久？
                Doc: "某朝代建立於100年，歷時300年，於400年滅亡。"
                <reasoning>
                1. 文檔明確提到「歷時300年」。
                2. 雖然400-100=300，但文檔已有直接答案。
                </reasoning>
                <answer>
                300年
                </answer>

                範例2 - 數值比較 (通用邏輯):
                Q: 蘋果和橘子哪一個比較重？
                Doc: "蘋果重200克，橘子重150克。"
                <reasoning>
                1. 文檔指出蘋果重200克。
                2. 文檔指出橘子重150克。
                3. 200克 > 150克，所以蘋果比較重。
                </reasoning>
                <answer>
                蘋果
                </answer>

                範例3 - 跨文檔推理 (地理/機構):
                Q: Alpha公司的總部所在的城市，其市長是誰？
                Doc1: "Alpha公司的總部位於貝克市。"
                Doc2: "貝克市的市長是詹姆斯·史密斯。"
                <reasoning>
                1. 從Doc1得知Alpha公司總部在貝克市。
                2. 從Doc2得知貝克市市長是詹姆斯·史密斯。
                3. 因此答案是詹姆斯·史密斯。
                </reasoning>
                <answer>
                詹姆斯·史密斯
                </answer>

                範例4 - 實體區分 (科學/定義):
                Q: 什麼是「光合作用」的主要產物？
                Doc1: "呼吸作用產生二氧化碳和水。"
                Doc2: "光合作用將光能轉化為化學能，產生葡萄糖和氧氣。"
                <reasoning>
                1. 問題詢問光合作用的產物。
                2. Doc1描述呼吸作用，不相關。
                3. Doc2明確指出光合作用產生葡萄糖和氧氣。
                </reasoning>
                <answer>
                葡萄糖和氧氣
                </answer>"""),
            ("human", "上下文:\n{context}\n\n問題:{question}\n\n請依照指定格式輸出:"),
        ])

        try:
            chain = prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"context": context_str, "question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )

            content = response.content
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if match:
                return match.group(1).strip()

            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
            if reasoning_match:
                return content.replace(reasoning_match.group(0), "").strip()

            return content
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"Error generating answer: {e}"

    # ---- Main entry point ----

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        exp = experiment or self.exp

        try:
            # Node 1: Query expansion
            expanded = self._query_expansion(question)

            # Node 2: Vector retrieval + graph expansion
            candidates = self._retrieve_vector_with_graph_expansion(expanded)

            # Node 3: Rerank
            vector_context, retrieved_ids = self._rerank(question, candidates)

            # Node 4: Graph retrieval
            graph_context = await self._retrieve_graph(question)

            # Node 5: Generate answer
            answer = self._generate_answer(question, vector_context, graph_context)

            confidence = min(1.0, len(retrieved_ids) / max(exp.top_k, 1))

            return AgentResponse(
                answer=answer,
                retrieved_doc_ids=retrieved_ids,
                context=vector_context + graph_context,
                confidence=confidence,
                metadata={
                    "agent": "kg",
                    "num_candidates": len(candidates),
                    "graph_relations": len(graph_context),
                },
            )
        except Exception as e:
            logger.error(f"KGAgent failed: {e}")
            return AgentResponse(
                answer=f"KG agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
