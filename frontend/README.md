# 1. 設定開發環境

## 1.1 安裝 Node.js 和 npm

從 [Node.js 官網](https://nodejs.org/) 下載並安裝最新版本的 Node.js，它會自動安裝 npm(建議安裝.msi到全域中而非docker)。

## 1.2 安裝 Vite

開啟終端機並執行以下指令來全域安裝 Vite：

```bash

npminstall-gcreate-vite

```

## 1.3 建立新專案

在 `BEARS/`下使用 Vite 建立新專案：(已建立，不須執行)

```bash
npm create vite@latest frontend -- --template react
```

## 1.4 安裝依賴

進入專案目錄並安裝依賴：

```bash
cd frontend
npm install
```

## 1.5 啟動開發伺服器

啟動開發伺服器來預覽應用程式：

```bash
npm run dev
```


# 2. Frontend 介紹

BEARS RAG 評估系統的前端介面，使用 React + Vite 建構。

## 2.1 技術棧

| 類別     | 技術             |
| -------- | ---------------- |
| 框架     | React 19 + Vite  |
| 路由     | react-router-dom |
| API 呼叫 | axios            |
| 圖表     | recharts         |

## 2.2 安裝

```bash
npm install react-router-dom axios recharts
```

Vite dev server 預設在 `http://localhost:5173`，已設定 proxy 將 `/api` 請求轉發到後端 `http://localhost:8000`。

ps fastapi開發模式啟動指令

```bash
uv run uvicorn bears.api.api:app --reload --port 8000
```

## 2.3 專案結構

```
frontend/src/
├── components/          # 共用元件
│   ├── AgentCard.jsx    # Agent 資訊卡片
│   ├── MetricsTable.jsx # 指標表格
│   └── ProgressBar.jsx  # 評估進度條
├── pages/
│   ├── Dashboard.jsx    # 主頁：agents + 啟動評估
│   ├── EvalResult.jsx   # 評估結果詳情
│   └── History.jsx      # 歷史結果列表
├── services/
│   └── api.js           # Axios 封裝，呼叫後端 API
├── App.jsx              # 路由設定
├── App.css              # 全域樣式
├── index.css            # Design tokens
└── main.jsx             # React entry point
```

# 3. 後端 API

前端對接的 API endpoints（後端運行在 `http://localhost:8000`）：

## 3.1 評估 (`/api/eval/`)

| Endpoint                         | 方法 | 說明             |
| -------------------------------- | ---- | ---------------- |
| `/api/eval/start`              | POST | 啟動評估任務     |
| `/api/eval/status/{task_id}`   | GET  | 查詢評估進度     |
| `/api/eval/results/{task_id}`  | GET  | 取得評估結果     |
| `/api/eval/history`            | GET  | 列出歷史評估檔案 |
| `/api/eval/history/{filename}` | GET  | 讀取特定歷史結果 |
| `/api/eval/agents`             | GET  | 列出可用 agents  |
| `/api/eval/queries/stats`      | GET  | 題目統計         |

## 3.2 問答 (`/api/`)

| Endpoint        | 方法 | 說明     |
| --------------- | ---- | -------- |
| `/api/query`  | POST | 單題問答 |
| `/api/health` | GET  | 健康檢查 |

## 3.3 實驗參數 (`/api/experiments/`)

| Endpoint                    | 方法   | 說明             |
| --------------------------- | ------ | ---------------- |
| `/api/experiments`        | GET    | 列出所有實驗參數 |
| `/api/experiments`        | POST   | 建立新實驗參數   |
| `/api/experiments/{name}` | GET    | 讀取特定實驗參數 |
| `/api/experiments/{name}` | PUT    | 更新實驗參數     |
| `/api/experiments/{name}` | DELETE | 刪除實驗參數     |
