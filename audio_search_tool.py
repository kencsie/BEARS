import os
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from langfuse.openai import OpenAI as LangfuseOpenAI

load_dotenv()

# ================= 設定區 =================
DB_NAME = "audio_rag_db"
COLLECTION_NAME = "audio_knowledge_base"

# 初始化
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# 使用 Langfuse 追蹤
client = LangfuseOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def search_and_answer(question):
    print(f"\n🔍 正在搜尋問題: {question}")
    
    # 1. 轉向量
    res = client.embeddings.create(input=question, model="text-embedding-3-small")
    query_vector = res.data[0].embedding
    
    # 2. 向量檢索
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
        {"$project": {"_id": 0, "context": 1, "audio_id": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    
    results = list(collection.aggregate(pipeline))
    if not results:
        return "找不到相關音檔內容。"
    
    top_result = results[0]
    context = top_result["context"]
    audio_id = top_result["audio_id"]
    
    print(f"✅ 找到最匹配音檔: audio_{audio_id}.mp3 (分數: {top_result['score']:.4f})")
    
    # 3. 生成回答
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided audio transcript context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        name="audio-rag-interactive"
    )
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_q = input("\n請輸入你的問題 (輸入 'q' 退出): ")
        if user_q.lower() == 'q':
            break
        answer = search_and_answer(user_q)
        print(f"\n💡 AI 回答:\n{answer}")
