# Multimodal RAG 專案

三種模態的檢索增強問答系統：**圖片 (Image)**、**影片 (Video)**、**音訊 (Audio)**。  
向量資料庫使用 **MongoDB Atlas Vector Search**，LLM 追蹤使用 **Langfuse**。

---

## 專案結構

```
multimodal/
├── image_qa/
│   ├── build_image_db.py           # 建立圖片知識庫（OCR + VLM + CLIP → MongoDB）
│   ├── run_hybrid_rerank.py        # 評測腳本（Hybrid + LLM Rerank，命中率 84%）
│   ├── rag_eval_hybrid_rerank.csv  # 評測結果
│   └── data/
│       └── infographic_rag_data_strict/  # 100 張圖片 + ground_truth_en.json
│
├── video_qa/
│   ├── download_how2qa_videos.py   # 下載 How2QA 影片片段
│   ├── build_how2qa_db.py          # 建立影片片段向量資料庫（Whisper + VLM + CLIP）
│   ├── build_enriched_how2qa.py    # 產生增強問答資料並存入 DB
│   ├── how2qa_eval.py              # 評測腳本（Hybrid + Query Expansion + Rerank）
│   ├── how2qa_rewritten_results.csv  # 評測結果
│   └── data/
│       ├── how2qa_100.json             # 原始 100 筆問答
│       ├── how2qa_100_enriched.json    # 增強後問答（含 enriched_question）
│       ├── how2qa_100_rewritten.json   # 改寫後問答（含 rewritten_question）
│       └── How2QA_100_Videos/          # 100 支 .mp4 影片片段
│
├── audio_qa/
│   ├── generate_audio_squad.py        # TTS 產生 .mp3 音訊檔
│   ├── build_audio_rag_db.py          # 建立音訊向量資料庫（Embedding → MongoDB）
│   ├── build_audio_knowledge_base.py  # 建立音訊知識庫（含 base64 音訊）
│   ├── audio_rag_search.py            # 評測腳本（Semantic Search）
│   ├── audio_search_tool.py           # 互動式搜尋介面
│   ├── audio_rag_full_eval.csv        # 評測結果
│   └── data/
│       ├── spoken_squad_100.json      # 100 筆英文問答資料
│       └── Spoken_SQuAD_Audio/        # 100 個 .mp3 音訊檔
│
└── .env                               # API Keys 設定
```

---

## 資料庫狀態

| 模態 | MongoDB DB | Collection | 狀態 |
|------|-----------|------------|------|
| Image QA | `infographic_rag_strict_db` | `strict_data` | ✅ 已建置完成 |
| Video QA | `how2qa_enriched_db` | `video_segments` | ✅ 已建置完成 |
| Audio QA | `audio_rag_db` | `audio_knowledge_base` | ✅ 已建置完成 |

> 若需重建，執行各模組的 `build_*.py` 腳本。重建前建議先清空對應的 MongoDB Collection。

---

## 評測結果

| 模態 | 策略 | 命中率 |
|------|------|--------|
| Image QA | Hybrid（文字 + CLIP）+ LLM Rerank | **84%** |
| Video QA | Hybrid（OpenAI + CLIP）+ Query Expansion + LLM Rerank | **Accuracy 72% / Top-10 Recall 90%** |
| Audio QA | Semantic Search（text-embedding-3-small） | **75%** |

---

## 環境設定

**Python 環境**：`C:\Miniconda3\envs\code\python.exe`（Python 3.11）

**`.env` 必要欄位**：
```env
OPENAI_API_KEY=sk-...
MONGO_URI=mongodb+srv://...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 執行方式

所有腳本使用 `__file__` 相對路徑，**可在任意目錄下執行**，不需要 `cd` 進子資料夾。

```bash
# 評測
python image_qa/run_hybrid_rerank.py
python video_qa/how2qa_eval.py
python audio_qa/audio_rag_search.py

# 互動式音訊搜尋
python audio_qa/audio_search_tool.py

# 重建資料庫（通常不需要，DB 已建置完成）
python image_qa/build_image_db.py
python video_qa/build_enriched_how2qa.py
python audio_qa/build_audio_knowledge_base.py
```
