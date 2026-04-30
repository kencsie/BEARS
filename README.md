# 多模態檢索與問答專案 (Multimodal RAG)

這個專案整理了三種不同類型的多模態檢索與問答系統：圖片 (Image)、影片 (Video) 以及音訊 (Audio)。

## 專案目錄結構

```text
/multimodal
  /image_qa          # 圖片檢索與問答 (InfoQA)
    /data            # 存放圖片資料與 Ground Truth
    run_4_strategies.py  # 評測四種圖片檢索策略的腳本
  /video_qa          # 影片檢索與問答 (How2QA)
    /data            # 存放影片檔案 (.mp4) 與 JSON 標籤
    build_how2qa_db.py       # 建立影片片段向量資料庫
    build_enriched_how2qa.py # 產生增強後的問答資料
    how2qa_eval.py           # 影片 RAG 效能評測
    download_how2qa_videos.py # 下載範例影片
  /audio_qa          # 音訊檢索與問答 (Spoken SQuAD)
    /data            # 存放音訊檔案 (.mp3) 與 JSON 資料集
    build_audio_rag_db.py    # 建立音訊向量資料庫
    audio_rag_search.py      # 自動化評測音訊檢索
    audio_search_tool.py     # 搜尋工具介面
    generate_audio_squad.py  # 使用 TTS 產生音訊檔案
  /utils             # 共用工具與模組
    DB.py            # 資料庫操作
    RAG.py           # RAG 核心邏輯
  .env               # 環境變數設定 (API Keys, Mongo URI)
```

## 各子專案說明

### 1. 圖片檢索 (Image QA)
*   **資料集**: Infographic RAG Data.
*   **技術**: 使用 OpenAI CLIP 進行文搜圖，並結合 GPT-4o-mini 進行 OCR 與影像描述。
*   **主要腳本**: `image_qa/run_4_strategies.py` - 比較 Hybrid Search、Visual Search 等不同策略。

### 2. 影片檢索 (Video QA)
*   **資料集**: How2QA (100 支影片).
*   **技術**: 將影片切分為片段，使用 Whisper 轉錄語音，並用 GPT-4o-mini Vision 產生畫面描述。
*   **主要腳本**: `video_qa/how2qa_eval.py` - 評測影片片段的檢索精準度。

### 3. 音訊檢索 (Audio QA)
*   **資料集**: Spoken SQuAD.
*   **技術**: 將文本轉換為音訊 (TTS)，使用 OpenAI Embedding 進行語意搜尋。
*   **主要腳本**: `audio_qa/audio_rag_search.py` - 測試音訊背景下的問答命中率。

## 如何開始

1.  請確保根目錄下的 `.env` 檔案已正確設定 `OPENAI_API_KEY` 與 `MONGO_URI`。
2.  進入各個子目錄後，即可執行對應的腳本。
3.  範例：`cd image_qa`, `python run_4_strategies.py`
