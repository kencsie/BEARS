# Router-Guided Agentic RAG Orchestrator - Implementation Plan

## Context

將現有 `GraphRag_hybrid_1/` 的 5 節點線性 pipeline 重構為 Router-Guided Orchestrator 架構。Router 分類意圖後，動態調度不同 RAG Agent 協作。

**設計決策：**
1. 使用 `src/bears/` (src layout) 取代 `app/`
2. 使用 `uv` + `pyproject.toml` 管理專案，不用 `requirements.txt`
3. 暫不需要 FastAPI，library-first 設計
4. Agent 註冊用 **Python dict**（不建 `configs/agents.yaml`），跟 LangGraph 生態一致
5. **不搬 `static/`**，留在 archive；未來 Streamlit 前端放 `app/`，等團隊討論再建
6. **刪除 `services/`** — `database/` 按 DB 類型分資料夾（含 store + ingestion），`evaluation/` 獨立為頂層模組，retrieval 邏輯由各 agent 自己實作
7. **設定雙層分離** — 系統設定（secrets、DB 連線）用 Pydantic Settings + `.env`；實驗參數（top_k、α/β、model）用 Pydantic BaseModel + YAML，方便版本控制與重現

## Target Structure

```
BEARS/
├── archive/
│   └── GraphRag_hybrid_1/          # 原有系統（僅供參考）
├── src/
│   └── bears/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # 系統設定：Pydantic Settings + .env（secrets、DB 連線）
│       │   ├── experiment.py       # 實驗參數：Pydantic BaseModel + YAML（top_k、α/β、model）
│       │   └── langfuse_helper.py  # 從 archive 複製（Langfuse callback）
│       ├── database/
│       │   ├── __init__.py
│       │   ├── vector/
│       │   │   ├── __init__.py
│       │   │   ├── vector_store.py     # ChromaDB 連線 + CRUD
│       │   │   └── vector_builder.py   # 讀 corpus → 建立向量（原 data_loader.py）
│       │   └── graph/
│       │       ├── __init__.py
│       │       ├── graph_store.py      # Neo4j 連線 + CRUD
│       │       └── graph_builder.py    # LLM 抽 entity → 建立知識圖譜
│       ├── router/
│       │   ├── __init__.py
│       │   ├── base.py             # BaseRouter ABC + RouterOutput
│       │   └── llm_router.py       # LLM-based 初始 router
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py             # BaseRAGAgent ABC（核心合約）
│       │   ├── registry.py         # Python dict 驅動 agent 註冊
│       │   ├── hybrid_agent/
│       │   │   ├── __init__.py
│       │   │   └── agent.py        # stub（組員自行實作）
│       │   ├── kg_agent/
│       │   │   ├── __init__.py
│       │   │   └── agent.py        # stub
│       │   ├── agentic_agent/
│       │   │   ├── __init__.py
│       │   │   └── agent.py        # stub
│       │   └── multimodal_agent/
│       │       ├── __init__.py
│       │       └── agent.py        # stub
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── state.py            # OrchestratorState TypedDict
│       │   ├── graph.py            # StateGraph 組裝 + run_orchestrated_rag() 入口
│       │   └── nodes.py            # 所有 node 函數 + agent wrapper
│       └── evaluation/
│           ├── __init__.py
│           ├── schemas.py           # 評估相關 schemas（從 archive 搬入）
│           ├── evaluator.py         # AgentEvaluator + OrchestratorEvaluator
│           ├── metrics.py           # 從 archive 複製
│           └── cli.py               # CLI 入口（python -m bears.evaluation.cli）
├── experiments/                    # 實驗參數 YAML（進 git）
│   └── default.yaml
├── data/                           # 從 archive 複製
├── pyproject.toml                  # uv 管理
└── .env.example
```

**與原方案的差異：**
- 刪除 `services/` — ingestion 併入 `database/`（按 DB 類型分資料夾），evaluation 升為頂層模組
- `database/` 按 DB 類型分：`vector/`（store + vector_builder）、`graph/`（store + builder）
- 刪除 `configs/` — agent 註冊改用 Python dict
- 不搬 `static/` — 留在 archive，未來 Streamlit 前端放 `app/`，等團隊討論
- 保留 `core/` — `config.py` 和 `langfuse_helper.py` 放 `core/`，與 archive 原始設計一致
- 刪除 `models/` — 評估 schemas 搬到 `evaluation/schemas.py`，agent/router schemas 各自定義在自己的 `base.py`
- Retrieval 邏輯由各 agent 自行實作，不設共用 retriever 層
- 新增 `core/experiment.py` + `experiments/` — 設定雙層分離（系統設定 vs 實驗參數）
- 新增 `evaluation/cli.py` — 組員可用 CLI 獨立評測自己的 agent

## Implementation Steps

### Step 1: 搬移 + 建立專案骨架

1. `mv GraphRag_hybrid_1 archive/`（已有的 `archive/` 目錄）
2. `uv init` 建立 `pyproject.toml`，設定 `src/bears/` 為 package
3. 建立所有目錄和 `__init__.py`
4. 從 archive 複製共用模組，**將所有 `from app.` import 改為 `from bears.`**
   - `database/vector/`：vector_store.py + vector_builder.py（原 data_loader.py）
   - `database/graph/`：graph_store.py + graph_builder.py
   - `evaluation/`：schemas.py + evaluator.py + metrics.py
5. `langfuse_helper.py` 放 `src/bears/core/`（`config.py` 和 `experiment.py` 在 Step 2 新建）
6. 複製 `data/` 和 `.env.example` 到專案根目錄
7. 建立 `experiments/default.yaml`

**pyproject.toml 關鍵設定：**
```toml
[project]
name = "bears"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3",
    "langchain-openai>=0.3",
    "langgraph>=0.6",
    "langfuse>=3.7",
    "chromadb>=1.4",
    "neo4j>=5.28",
    "openai>=2.17",
    "pydantic>=2.12",
    "pydantic-settings>=2.11",
    "python-dotenv>=1.2",
    "pyyaml>=6.0",
    "instructor>=1.14",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bears"]

[project.scripts]
bears-eval = "bears.evaluation.cli:main"
```

> `bears-eval` 讓安裝後可直接用 `bears-eval --agent hybrid ...` 執行評測。
> 開發階段也可用 `uv run python -m bears.evaluation.cli --agent hybrid ...`。

### Step 2: 設定雙層分離 — `config.py` + `experiment.py`

設定分成兩層，各司其職：

| 層 | 內容 | 存放方式 | 進 git？ |
|----|------|---------|---------|
| **系統設定** | API keys、DB 連線、Langfuse | Pydantic Settings + `.env` | 否（`.env` 在 `.gitignore`） |
| **實驗參數** | top_k、rerank α/β、model、temperature | Pydantic BaseModel + YAML | 是（`experiments/*.yaml`） |

#### `core/config.py` — 系統設定

```python
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets — 必填，缺了啟動就失敗
    OPENAI_API_KEY: str
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    # Observability — 選填（沒設就不啟用 Langfuse）
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # 路徑
    DATA_DIR: Path = Path("data")

    @property
    def corpus_path(self) -> Path:
        return self.DATA_DIR / "corpus.json"

    @property
    def queries_path(self) -> Path:
        return self.DATA_DIR / "queries.json"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

#### `core/experiment.py` — 實驗參數

```python
import yaml
from pydantic import BaseModel

class ExperimentConfig(BaseModel):
    """一次實驗的參數組合。可從 YAML 載入，也可直接建構。"""

    # LLM
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Retrieval
    top_k: int = 5

    # Reranking（雙分數加權）
    rerank_alpha: float = 0.7   # vector distance 權重
    rerank_beta: float = 0.3    # LLM grade 權重

    # Agent 選擇（CLI 單一 agent 評測時使用）
    agent: str = "hybrid"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
```

#### `experiments/default.yaml` — 預設實驗參數

```yaml
model: "gpt-4o-mini"
temperature: 0.0
top_k: 5
rerank_alpha: 0.7
rerank_beta: 0.3
agent: "hybrid"
```

#### 使用方式

```python
# Agent 內部：系統設定取 secrets，實驗參數取可調值
from bears.core.config import settings
from bears.core.experiment import ExperimentConfig

class HybridRAGAgent(BaseRAGAgent):
    def __init__(self, experiment: ExperimentConfig | None = None):
        self.exp = experiment or ExperimentConfig()
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,     # 系統設定
            model=self.exp.model,                 # 實驗參數
            temperature=self.exp.temperature,     # 實驗參數
        )

    def retrieve(self, question, top_k=None):
        k = top_k or self.exp.top_k              # 實驗參數
        ...
```

### Step 3: 核心合約 — `src/bears/agents/base.py`

新建。定義 `AgentCapability`, `AgentResponse`, `BaseRAGAgent` ABC。
- `AgentResponse` 包含完整答案（`answer`）、context、confidence、metadata
- 每個 agent 端到端自行處理 retrieval + generation，直接回傳答案

### Step 4: Router 介面 — `src/bears/router/base.py` + `llm_router.py`

新建。定義 `RouterOutput`, `BaseRouter` ABC。
`llm_router.py`：用 GPT-4o-mini + structured output 做初始分類。

### Step 5: Orchestrator — `state.py` + `graph.py` + `nodes.py`

- `state.py`：`OrchestratorState(TypedDict)` 擴展原有 `GraphState`，加入 `route_decision`, `agent_results` (Annotated[..., operator.add]), `retry_count`, `trace_id` 等
- `nodes.py`：所有 node 函數 + `wrap_agent_as_node()`
  - `router_node` (Command 動態路由), `fan_out_dispatch` (Send API), `merge_node`（選擇最佳答案）, `quality_gate`
  - 不需要 `generate_node` — 每個 agent 自行生成答案，merge 只負責選擇
- `graph.py`：`create_orchestrator_workflow(registry)` 組裝 StateGraph + `run_orchestrated_rag(question)` 入口

### Step 6: Agent Registry — `registry.py`（Python dict）

`registry.py`：Python dict 定義 agent 清單，動態 import + 實例化。

```python
AGENT_REGISTRY = {
    "hybrid": {
        "module": "bears.agents.hybrid_agent.agent",
        "class_name": "HybridRAGAgent",
        "enabled": True,
    },
    "kg": {
        "module": "bears.agents.kg_agent.agent",
        "class_name": "KGAgent",
        "enabled": False,
    },
    # ...
}
```

### Step 7: 各 Agent Stub

為每個 agent 目錄建立 stub 實作（回傳空 AgentResponse + 固定 confidence），讓組員可以獨立填入邏輯：
- `hybrid_agent/agent.py` — stub
- `kg_agent/agent.py` — stub
- `agentic_agent/agent.py` — stub
- `multimodal_agent/agent.py` — stub

原有 `graph_rag.py` 的邏輯留在 archive 供參考，組員自行決定如何包裝進各自的 agent。

各 agent 自行 import `bears.database.vector.vector_store` / `bears.database.graph.graph_store` 來做 retrieval，不經過共用 retriever 層。

### Step 8: Evaluation — Agent Adapter + CLI

#### 8a: `evaluator.py` — Agent Adapter 模式

修改 `src/bears/evaluation/evaluator.py`，設計兩層評估器：

```python
# src/bears/evaluation/evaluator.py

class AgentEvaluator:
    """通用 agent 評估器，支援單一 agent 獨立評測。
    讓每位組員可以獨立評測自己的 agent，不需要經過 orchestrator。
    """

    def __init__(self, agent: BaseRAGAgent, experiment: ExperimentConfig | None = None):
        self.agent = agent
        self.exp = experiment or ExperimentConfig()

    def evaluate(self, queries_path: str) -> EvalResult:
        """對單一 agent 跑全部 queries，計算 metrics。
        top_k、rerank 參數等從 self.exp 取得。
        - Hit Rate, Partial Hit Rate, MRR, MAP
        - LLM-as-judge (generation pass rate)
        - 按 source_dataset / question_type 分群統計
        """
        ...

    def evaluate_single(self, question: str, gold_doc_ids: list, gold_answer: str) -> QuestionDetail:
        """評測單一問題"""
        ...


class OrchestratorEvaluator:
    """評估整個 orchestrator pipeline（Router → Agent → Merge 的端到端表現）"""

    def evaluate(self, queries_path: str) -> EvalResult:
        result = run_orchestrated_rag(question)
        ...
```

**為什麼用 Adapter 模式**：IMPLEMENTATION_PLAN 有多個 agent，evaluator 需要能分別評估每個 agent 的表現（不只是 orchestrator 的最終輸出）。Adapter 模式讓評估邏輯與 agent 實作解耦。

**參考實作**：
- `AgenticFlow/eval_runner.py` 的 `YourRAGSystem` adapter（lines 139-243）
- `archive/GraphRag_hybrid_1/app/services/evaluation/evaluator.py` 的指標計算和分群統計
- `hybrid_rag/scripts/calculate_metrics.py` 的 Hit Rate / MRR / MAP 計算

#### 8b: `cli.py` — CLI 入口

新建 `src/bears/evaluation/cli.py`，作為 `python -m bears.evaluation.cli` 的入口：

```bash
# 用預設參數評測單一 agent
uv run python -m bears.evaluation.cli --agent hybrid

# 用 YAML 實驗參數檔（不同 top_k、α/β 組合）
uv run python -m bears.evaluation.cli --agent hybrid --config experiments/exp_1.yaml

# 評測 orchestrator 整體
uv run python -m bears.evaluation.cli --orchestrator --config experiments/exp_1.yaml

# 輸出格式：JSON 報告（by_source, by_question_type, per_question details）
```

CLI 內部邏輯：
1. `--agent <name>`：從 `AGENT_REGISTRY` 動態載入指定 agent → 包裝為 `AgentEvaluator` → 跑評測
2. `--orchestrator`：呼叫 `OrchestratorEvaluator` → 跑評測
3. `--config <path>`：載入 YAML 實驗參數檔（`ExperimentConfig.from_yaml()`），未指定則用 `ExperimentConfig` 預設值
4. `--queries <path>`：指定 queries JSON 檔路徑（預設 `data/queries.json`）
5. 不提供個別參數的 CLI 覆寫（如 `--top-k`）——所有實驗參數必須透過 YAML 檔管理，確保可追蹤、可重現
6. 結果輸出到 stdout（JSON），可 redirect 到檔案

#### 8c: 評估 schemas

評估 schemas（`SourceMetrics`, `QuestionDetail`, `EvalResult` 等）放在 `evaluation/schemas.py`，從 archive 搬入並調整 import。

## Key Files to Copy from Archive (with `app.` → `bears.` import fix)

| Source (archive/) | Destination (src/bears/) | Changes |
|---|---|---|
| `app/core/config.py` | `core/config.py` | 依 Step 2 重寫（雙層分離，只保留系統設定） |
| — (新建) | `core/experiment.py` | 實驗參數 `ExperimentConfig`（見 Step 2） |
| — (新建) | `experiments/default.yaml` | 預設實驗參數（見 Step 2） |
| `app/core/langfuse_helper.py` | `core/langfuse_helper.py` | `from app.` → `from bears.` |
| `app/models/schemas.py` | `evaluation/schemas.py` | 只保留評估相關 schemas |
| `app/database/vector_store.py` | `database/vector/vector_store.py` | `from app.` → `from bears.` |
| `app/database/graph_store.py` | `database/graph/graph_store.py` | `from app.` → `from bears.` |
| `app/services/ingestion/data_loader.py` | `database/vector/vector_builder.py` | `from app.` → `from bears.`，重新命名 |
| `app/services/ingestion/graph_builder.py` | `database/graph/graph_builder.py` | `from app.` → `from bears.` |
| `app/services/evaluation/metrics.py` | `evaluation/metrics.py` | 直接複製 |
| `app/services/evaluation/evaluator.py` | `evaluation/evaluator.py` | AgentEvaluator + OrchestratorEvaluator（見 Step 8） |
| — (新建) | `evaluation/cli.py` | CLI 入口（`python -m bears.evaluation.cli`） |
| `data/` | `data/` (project root) | 直接複製 |
| `.env.example` | `.env.example` (project root) | 直接複製 |

**不搬的檔案：**
- `static/` — 留在 archive
- `app/services/retrieval/` — 各 agent 自行實作 retrieval 邏輯
- `app/services/rag/graph_rag.py` — 留在 archive 供參考
- `app/services/service_layer.py` — FastAPI 用的統一入口，不需要
- `app/main.py` — FastAPI app，不需要

## Verification

1. `uv sync` 確認依賴安裝正常
2. `uv run python -c "from bears.orchestrator.graph import run_orchestrated_rag"` 確認 import chain 正確
3. `uv run python -c "from bears.core.config import settings"` 確認系統設定載入正常（需有 `.env`）
4. `uv run python -c "from bears.core.experiment import ExperimentConfig; ExperimentConfig.from_yaml('experiments/default.yaml')"` 確認實驗參數載入正常
5. `uv run python -m bears.evaluation.cli --agent hybrid --config experiments/default.yaml` 確認 CLI 可執行
6. `run_orchestrated_rag()` 回傳結構與原 `run_graph_rag()` 相容
