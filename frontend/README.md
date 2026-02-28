
## 設定開發環境

### 1 安裝 Node.js 和 npm

從 [Node.js 官網](https://nodejs.org/) 下載並安裝最新版本的 Node.js，它會自動安裝 npm(建議安裝.msi到全域中而非docker)。


### 2 安裝 Vite

開啟終端機並執行以下指令來全域安裝 Vite：

```bash
npm install -g create-vite
```

### 3 建立新專案

在 `BEARS/`下使用 Vite 建立新專案：(已建立，不須執行)

```bash
npm create vite@latest frontend -- --template react
```

### 4 安裝依賴

進入專案目錄並安裝依賴：

```bash
cd frontend
npm install
```

### 5 啟動開發伺服器

啟動開發伺服器來預覽應用程式：

```bash
npm run dev
```
