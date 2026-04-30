import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ================= 設定區 =================
DB_NAME = "audio_rag_db"
COLLECTION_NAME = "squad_segments"
JSON_FILE = "data/spoken_squad_100.json"

# 初始化
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """計算 OpenAI 向量"""
    try:
        res = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return res.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def ingest_data():
    if not os.path.exists(JSON_FILE):
        print(f"❌ 找不到 {JSON_FILE}")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🧹 清空舊資料庫: {DB_NAME}.{COLLECTION_NAME}")
    collection.delete_many({})

    print(f"🚀 開始入庫 {len(data)} 筆資料...")

    docs = []
    for i, item in enumerate(tqdm(data)):
        # 計算 Context 的向量 (這是檢索的關鍵)
        vector = get_embedding(item["context"])
        
        doc = {
            "squad_id": item["id"],
            "context": item["context"],
            "question": item["question"],
            "answer": item["answer"],
            "audio_filename": f"audio_{i}.mp3",
            "embedding": vector
        }
        docs.append(doc)

    if docs:
        collection.insert_many(docs)
        print(f"✅ 成功存入 {len(docs)} 筆資料！")
    
    print("\n💡 記得在 MongoDB Atlas 上建立 Vector Search Index：")
    print("   Index Name: vector_index")
    print("   Path: embedding")
    print("   Dimensions: 1536")

if __name__ == "__main__":
    ingest_data()
