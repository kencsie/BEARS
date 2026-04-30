import os
import json
import base64
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ================= 設定區 =================
DB_NAME = "audio_rag_db"
COLLECTION_NAME = "audio_knowledge_base" # 重新命名為知識庫
JSON_FILE = "data/spoken_squad_100.json"
AUDIO_DIR = "data/Spoken_SQuAD_Audio"

# 初始化
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """純粹對 Context 進行語義向量計算"""
    try:
        res = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return res.data[0].embedding
    except:
        return []

def get_audio_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def ingest_pure_knowledge():
    if not os.path.exists(JSON_FILE):
        print("Error: JSON file not found.")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Cleaning collection: {COLLECTION_NAME}...")
    collection.delete_many({})

    print(f"Start ingesting {len(data)} pure audio contexts...")

    docs = []
    for i, item in enumerate(tqdm(data)):
        audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.mp3")
        
        if not os.path.exists(audio_path):
            continue

        # 只存核心資料
        doc = {
            "context": item["context"],
            "audio_base64": get_audio_base64(audio_path),
            "embedding": get_embedding(item["context"]),
            "audio_id": i
        }
        docs.append(doc)

    if docs:
        collection.insert_many(docs)
        print(f"\nSuccessfully created Audio Knowledge Base: {COLLECTION_NAME}")

if __name__ == "__main__":
    ingest_pure_knowledge()
