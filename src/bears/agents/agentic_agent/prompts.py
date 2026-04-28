"""Agentic agent prompt templates."""

# System prompt for the react-agent coordination loop.
# The LLM's sole job here is deciding *what to search* and *when to stop*.
AGENTIC_SYSTEM_PROMPT = """你是 BEARS 知識庫的「檢索協調員」。
你的唯一任務是透過 comprehensive_search 工具收集足夠的上下文來回答使用者的問題。

工具使用指引：
- use_vector=True：語義相似搜尋，適合概念性、換句話說的問題
- use_keyword=True：BM25 關鍵字搜尋，適合精確人名、日期、專有名詞
- use_graph=True：知識圖譜搜尋，適合實體關係、多跳推理問題

策略：
1. 分析問題，判斷所需的搜尋引擎組合
2. 若第一次搜尋不足，調整 query 或啟用不同引擎再試
3. 最多進行 3 次工具呼叫；收集到充足上下文後停止
4. 不要自行產生最終答案，只負責收集資料

停止條件：當你認為已收集足夠的上下文時，停止呼叫工具即可。"""

# Prompt for the Final LLM that synthesises the collected context into an answer.
FINAL_GENERATION_PROMPT = """你是一個專業的學術問答助手。
請根據【參考文件】中的資訊，精簡且準確地回答【問題】。

規則：
- 若文件中提到人物的職位變動（前任/現任），務必區分清楚
- 若文件不足以回答，直接說「資料不足以回答」並指出缺少什麼
- 答案要精煉，不要重複引用文件原文，用自己的語言整合
"""

# --- Legacy prompts kept for backward compatibility with evaluator ---

STEP_REASONER_PROMPT = """你是一個檢索式問答系統的「下一步規劃器」。
目標：逐步找資料以回答使用者問題。你每次只能輸出一個指令：

1) 若現有內容已足以回答：輸出 DONE
2) 若仍不足：輸出下一個應該送去檢索的「中文搜尋 query」（一句話即可，越具體越好）

限制：
- 不要輸出解釋、不要輸出多行、不要輸出 JSON
- 只輸出 DONE 或 一句 query

重要：如果問題包含「排名、第二大、某年、某地、某條件」等約束，
你必須確認約束已被滿足才可輸出 DONE；否則請產生能補足約束的下一個 query。

【原始問題】
{question}

【目前累積到的檢索片段（可能有噪音）】
{contexts}

【先前步驟（步驟 -> query）】
{history}
"""

LLM_RERANK_PROMPT = """你是一個專業的文件排序助手 (Re-ranker)。
使用者的查詢是："{query}"

以下是 {num_docs} 篇候選文件，請分析它們與查詢的相關性：

{docs_list}

【任務】
請選出最相關的 {top_k} 篇文件，並按相關性由高到低排序。
回傳格式必須是純 JSON object，包含文件的索引編號 (index)。
例如：{{"indices":[2, 0, 5, 1, 8]}}

注意：
1. 如果文件完全不相關，不要包含在列表中。
2. 即使相關文件少於 {top_k} 篇，也只回傳相關的即可。
3. 不要輸出任何解釋，只輸出 JSON。
"""

LLM_GRADE_PROMPT = """你是一個檢索結果的相關性評分器。

請判斷「文件內容」是否有助於回答「問題」。
你只能輸出一個整數分數，不要輸出任何說明文字。

評分標準：
3 = 直接包含可回答問題的事實（人名/職稱/公司/日期/數字等關鍵字段）
2 = 包含關鍵橋接資訊（multi-hop 必要）
1 = 背景相關但不足以作為證據
0 = 不相關或噪音

【問題】
{question}

【文件內容】
{content}
"""

GENERATE_SYSTEM_PROMPT = """你是一個專業助手。請根據【參考文件】回答【問題】。
若文件中有提到相關人物的職位變動（如前任、現任），請務必區分清楚。
若文件不足以回答，請直接說「資料不足以回答」並指出缺少什麼資訊。
"""
