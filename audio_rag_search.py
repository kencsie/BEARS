import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from langfuse.openai import OpenAI as LangfuseOpenAI

load_dotenv()

# ================= 設定區 =================
DB_NAME = "audio_rag_db"
COLLECTION_NAME = "squad_segments"
JSON_FILE = "spoken_squad_100.json"

# 初始化
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# 使用 Langfuse 封裝的 OpenAI
client = LangfuseOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LANGFUSE_BASE_URL") # 確保 Langfuse 有正確掛載
)

def get_query_vector(text):
    """計算問題的向量"""
    res = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def search_audio_context(query_vector):
    """在 MongoDB 執行向量搜尋"""
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 10,
                "limit": 1
            }
        },
        {
            "$project": {
                "_id": 0,
                "context": 1,
                "audio_filename": 1,
                "question": 1,
                "answer": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None

def generate_answer(question, context):
    """呼叫 GPT-4o 根據檢索內容回答"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the provided context (which is transcribed from an audio clip). If the answer is not in the context, say you don't know."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        name="audio-rag-qa" # Langfuse 中的 Trace 名稱
    )
    return response.choices[0].message.content

def main():
    if not os.path.exists(JSON_FILE):
        print("❌ 找不到評測檔案")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    results = []
    print(f"🚀 開始聲音 RAG 評測 (總計 {len(qa_data)} 題)...")

    for item in tqdm(qa_data):
        q_text = item["question"]
        gt_ans = item["answer"]
        
        # 1. 搜尋
        q_vector = get_query_vector(q_text)
        retrieved = search_audio_context(q_vector)
        
        if retrieved:
            r_context = retrieved["context"]
            r_audio = retrieved["audio_filename"]
            r_score = retrieved["score"]
            # 2. 回答
            ai_ans = generate_answer(q_text, r_context)
            # 3. 判斷命中 (這裡以是否找回正確的 context ID 或音檔名為準)
            # 在 SQuAD 中，我們通常判斷檢索到的 context 是否包含正確答案
            is_hit = (r_audio == f"audio_{qa_data.index(item)}.mp3") 
        else:
            r_context = "Not Found"
            r_audio = "None"
            r_score = 0
            ai_ans = "Search Failed"
            is_hit = False

        results.append({
            "Question": q_text,
            "GT Answer": gt_ans,
            "Retrieved Audio": r_audio,
            "Hit": is_hit,
            "Score": r_score,
            "AI Answer": ai_ans
        })

    # 存檔
    df = pd.DataFrame(results)
    df.to_csv("audio_rag_eval_results.csv", index=False, encoding="utf-8-sig")
    
    print(f"\n✅ 評測完成！結果已存入 audio_rag_eval_results.csv")
    print(f"📈 整體命中率: {df['Hit'].mean():.1%}")

if __name__ == "__main__":
    main()
