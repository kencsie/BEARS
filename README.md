# BEARS — Router-Guided Agentic RAG Orchestrator

BEARS 是一個多 Agent RAG（Retrieval-Augmented Generation）系統。透過 LLM Router 分析使用者問題，動態調度不同 RAG Agent 協作，最後由 Orchestrator 合併最佳答案。

## 架構總覽

```
User Question
      │
      ▼
┌──────────┐
│  Router   │  ← LLM 分類意圖，決定派發哪些 Agent
└────┬─────┘
     │  fan-out
     ▼
┌─────────┐  ┌──────────┐  ┌───────────┐
│ Hybrid  │  │    KG    │  │  Agentic  │   ← 各 Agent 獨立 retrieve + generate
└────┬────┘  └────┬─────┘  └─────┬─────┘
     │            │              │
     └────────────┼──────────────┘
                  ▼
           ┌───────────┐
           │   Merge   │  ← 選擇 confidence 最高的答案
           └───────────┘
                  │
                  ▼
            Final Answer
```

## 目錄結構

```
BEARS/
├── src/bears/                  # 主要套件
│   ├── core/                   # 系統設定 + 實驗參數
│   │   ├── config.py           #   Pydantic Settings + .env（secrets）
│   │   ├── experiment.py       #   Pydantic BaseModel + YAML（實驗參數）
│   │   └── langfuse_helper.py  #   Langfuse observability
│   ├── database/               # 資料層
│   │   ├── vector/             #   ChromaDB 向量儲存 + 資料載入
│   │   └── graph/              #   Neo4j 圖譜儲存 + 建構
│   ├── router/                 # 路由層
│   │   ├── base.py             #   BaseRouter ABC + RouterOutput
│   │   └── llm_router.py      #   GPT-4o-mini 分類路由
│   ├── agents/                 # Agent 層
│   │   ├── base.py             #   BaseRAGAgent ABC + AgentResponse
│   │   ├── registry.py         #   Agent 註冊表（動態 import）
│   │   ├── hybrid_agent/       #   向量搜尋 + RRF 融合
│   │   ├── kg_agent/           #   知識圖譜 5 節點 pipeline
│   │   ├── agentic_agent/      #   多步驟迭代檢索
│   │   └── multimodal_agent/   #   Stub（未來擴充）
│   ├── orchestrator/           # 編排層（LangGraph StateGraph）
│   │   ├── state.py            #   OrchestratorState
│   │   ├── nodes.py            #   router / agent wrapper / merge 節點
│   │   └── graph.py            #   StateGraph 組裝 + 入口函式
│   └── evaluation/             # 評估系統
│       ├── schemas.py          #   SourceMetrics, QuestionDetail
│       ├── metrics.py          #   Hit Rate, MRR, MAP
│       ├── evaluator.py        #   AgentEvaluator + OrchestratorEvaluator
│       └── cli.py              #   CLI 入口（bears-eval）
├── experiments/                # 實驗參數 YAML（進 git）
│   └── default.yaml
├── data/                       # 評估資料
│   └── queries.json
├── archive/                    # 原始實作（僅供參考，不修改）
├── pyproject.toml
└── .env.example
```

## 快速開始

### 前置需求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)（套件管理器）
- OpenAI API Key
- Neo4j 資料庫（KG Agent 需要）

### 1. 安裝

```bash
git clone <repo-url> && cd BEARS
git checkout dev

# 安裝所有依賴
uv sync
```

### 2. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`，填入實際的 API Key 和資料庫連線資訊：

```env
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# 選填（不設定則不啟用追蹤）
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

### 3. 驗證安裝

```bash
# 確認設定可載入
uv run python -c "from bears.core.config import Settings; print('OK')"

# 確認實驗參數可載入
uv run python -c "from bears.core.experiment import ExperimentConfig; print(ExperimentConfig.from_yaml('experiments/default.yaml'))"

# 確認所有 Agent 可載入
uv run python -c "
from bears.agents.registry import AGENT_REGISTRY
print('Registered agents:', list(AGENT_REGISTRY.keys()))
"
```

## 使用方式

### 以 Python 呼叫單一 Agent

```python
import asyncio
from bears.agents.registry import get_agent
from bears.core.experiment import ExperimentConfig

# 載入實驗參數（或使用預設值）
exp = ExperimentConfig.from_yaml("experiments/default.yaml")

# 選擇 Agent：hybrid / kg / agentic
agent = get_agent("hybrid", experiment=exp)

# 執行
result = asyncio.run(agent.run("台灣第一座國家公園是哪座？"))

print(result.answer)
print(result.retrieved_doc_ids)
print(result.confidence)
```

### 以 Python 呼叫 Orchestrator（Router → Agent → Merge）

```python
import asyncio
from bears.orchestrator.graph import run_orchestrated_rag

result = asyncio.run(run_orchestrated_rag("美國銷售額第二大汽車租賃公司的執行長是誰？"))

print(result["answer"])
print(result["retrieved_doc_ids"])
```

### 以 CLI 評估 Agent

```bash
# 評估 hybrid agent（使用預設參數）
uv run bears-eval --agent hybrid

# 評估 kg agent，指定實驗參數檔
uv run bears-eval --agent kg --config experiments/default.yaml

# 評估 agentic agent，限制 10 題（快速測試）
uv run bears-eval --agent agentic --limit 10

# 評估 orchestrator 端到端
uv run bears-eval --orchestrator

# 指定自訂 queries 檔案
uv run bears-eval --agent hybrid --queries data/queries.json

# 輸出結果到檔案
uv run bears-eval --agent hybrid > results/hybrid_eval.json
```

CLI 也可用 `python -m` 方式呼叫：

```bash
uv run python -m bears.evaluation.cli --agent hybrid --config experiments/default.yaml
```

#### 評估輸出格式

```json
{
  "overall": {
    "total_questions": 100,
    "hit_rate": 0.85,
    "partial_hit_rate": 0.72,
    "mrr": 0.68,
    "map": 0.55,
    "generation_pass_rate": 0.70,
    "avg_total_time": 3.2
  },
  "by_source": {
    "drcd": { "..." },
    "hotpotqa": { "..." },
    "2wiki": { "..." }
  },
  "by_question_type": {
    "single-hop": { "..." },
    "multi-hop": { "..." }
  }
}
```

## 設定系統

BEARS 採用**雙層設定分離**設計：

| 層級 | 內容 | 存放方式 | 進 Git？ |
|------|------|----------|----------|
| **系統設定** | API Keys、DB 連線、Langfuse | `.env` (Pydantic Settings) | 否 |
| **實驗參數** | model、top_k、rerank alpha/beta | `experiments/*.yaml` (Pydantic BaseModel) | 是 |

### 建立新實驗

```yaml
# experiments/exp_topk10.yaml
model: "gpt-4o-mini"
temperature: 0.0
top_k: 10
rerank_alpha: 0.6
rerank_beta: 0.4
agent: "hybrid"
```

```bash
uv run bears-eval --agent hybrid --config experiments/exp_topk10.yaml
```

## 可用 Agent

| 名稱 | 說明 | 來源 |
|------|------|------|
| `hybrid` | 向量搜尋 + 多查詢擴展 + RRF 融合 | `archive/hybrid_rag/` |
| `kg` | 知識圖譜 5 節點 pipeline（查詢擴展→向量+圖譜擴展→LLM 重排序→圖譜檢索→生成） | `archive/GraphRag_hybrid_1/` |
| `agentic` | 多步驟迭代：逐步檢索→LLM 重排序→推理下一步→距離+LLM 評分融合→生成 | `archive/AgenticFlow/` |
| `multimodal` | Stub（未實作，confidence 固定 0.0） | — |

## 開發指南

### 新增一個 Agent

1. 建立目錄 `src/bears/agents/your_agent/`
2. 建立 `__init__.py` 和 `agent.py`
3. 繼承 `BaseRAGAgent`，實作 `name`、`capabilities`、`run()` 三個介面：

```python
from bears.agents.base import BaseRAGAgent, AgentCapability, AgentResponse
from bears.core.experiment import ExperimentConfig

class YourAgent(BaseRAGAgent):
    def __init__(self, experiment=None):
        self.exp = experiment or ExperimentConfig()

    @property
    def name(self) -> str:
        return "your_agent"

    @property
    def capabilities(self):
        return {AgentCapability.VECTOR_SEARCH}

    async def run(self, question, experiment=None) -> AgentResponse:
        # 你的 retrieval + generation 邏輯
        return AgentResponse(
            answer="...",
            retrieved_doc_ids=["doc_1", "doc_2"],
            context=["..."],
            confidence=0.8,
        )
```

4. 在 `src/bears/agents/registry.py` 註冊：

```python
AGENT_REGISTRY["your_agent"] = {
    "module": "bears.agents.your_agent.agent",
    "class_name": "YourAgent",
    "enabled": True,
}
```

5. 測試：

```bash
uv run bears-eval --agent your_agent --limit 5
```

### 使用共用資料庫模組

Agent 內部可直接 import 共用的資料庫模組：

```python
from bears.database.vector.vector_store import VectorStoreManager
from bears.database.graph.graph_store import GraphStoreManager

vector_store = VectorStoreManager()
docs = vector_store.search("查詢文字", k=5)

graph_store = GraphStoreManager()
relations = graph_store.query_entity("實體名稱", limit=10)
```

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini（可在 YAML 設定切換）
- **向量資料庫**: ChromaDB
- **圖譜資料庫**: Neo4j
- **編排框架**: LangGraph (StateGraph)
- **LLM 框架**: LangChain
- **Observability**: Langfuse（選填）
- **套件管理**: uv + pyproject.toml (hatchling)
