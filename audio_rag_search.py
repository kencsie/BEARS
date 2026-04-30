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
COLLECTION_NAME = "audio_knowledge_base"
JSON_FILE = "spoken_squad_100.json"

# 初始化
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# 使用 Langfuse 封裝的 OpenAI (修正過的初始化)
client = LangfuseOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_query_vector(text):
    res = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return res.data[0].embedding

def search_audio_context(query_vector):
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
                "audio_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None

def generate_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on the context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        name="audio-rag-batch-eval"
    )
    return response.choices[0].message.content

def main():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    results = []
    print(f"Starting automated evaluation for 100 questions...")

    for i, item in enumerate(tqdm(qa_data)):
        q_text = item["question"]
        gt_ans = item["answer"]
        
        # 1. Search
        q_vector = get_query_vector(q_text)
        retrieved = search_audio_context(q_vector)
        
        if retrieved:
            r_context = retrieved["context"]
            r_audio_id = retrieved["audio_id"]
            r_score = retrieved["score"]
            # Check hit
            is_hit = (r_audio_id == i)
            # 2. Answer
            ai_ans = generate_answer(q_text, r_context)
        else:
            is_hit = False
            ai_ans = "Search Failed"
            r_audio_id = -1
            r_score = 0

        results.append({
            "ID": i,
            "Question": q_text,
            "GT": gt_ans,
            "Hit": is_hit,
            "Retrieved_ID": r_audio_id,
            "Score": r_score,
            "AI_Answer": ai_ans
        })

    df = pd.DataFrame(results)
    df.to_csv("audio_rag_full_eval.csv", index=False, encoding="utf-8-sig")
    
    print(f"\nEvaluation Complete!")
    print(f"Top-1 Retrieval Hit Rate: {df['Hit'].mean():.1%}")
    print(f"Detailed results saved to audio_rag_full_eval.csv")

if __name__ == "__main__":
    main()
