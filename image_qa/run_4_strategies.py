import os
import json
import base64
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pymongo import MongoClient
from langfuse.openai import OpenAI
from langfuse import observe, get_client
from transformers import CLIPProcessor, CLIPModel

# ================= 設定區 =================
DB_NAME = "infographic_rag_strict_db"
COLLECTION_NAME = "strict_data" 

JSON_FILE = "data/infographic_rag_data_strict/ground_truth_en.json"
IMAGE_FOLDER = "data/infographic_rag_data_strict"

# ================= 環境初始化 =================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_HOST"]       = LANGFUSE_HOST
else:
    print("⚠️ 警告：未設定 LANGFUSE keys，追蹤功能將不啟用。")

print("⏳ 系統初始化中...")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("   ↳ 載入 CLIP 模型...")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

print("✅ 初始化完成！準備開始評測...\n")

# ================= 核心函式 =================

@observe(as_type="generation", name="text-embedding")
def get_text_embedding(text):
    if not text: return []
    try:
        text = text.replace("\n", " ")
        res = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        return res.data[0].embedding
    except: return []

@observe(as_type="generation", name="clip-text-embedding")
def get_clip_text_embedding(text):
    """計算 CLIP 文字向量，加入英文翻譯與防崩潰機制"""
    if not text: return []
    try:
        # 1. 中翻英
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translation engine. Translate the user's Chinese question into simple English keywords for image retrieval. Do not answer it."},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.1
        )
        en_text = res.choices[0].message.content.strip()
        
        # 2. 截斷保護
        inputs = clip_processor(text=[en_text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
            
        # 3. 相容性處理
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs.pooler_output
            
        return (outputs / outputs.norm(p=2, dim=-1, keepdim=True))[0].tolist()
    except Exception as e:
        print(f"\n⚠️ CLIP 錯誤: {e}")
        return []

@observe(name="vector-search")
def perform_single_vector_search(vector, path, limit=10):
    """執行單一向量搜尋"""
    if not vector: return []
    pipeline = [
        {"$vectorSearch": {"index": "vector_index", "path": path, "queryVector": vector, "numCandidates": 100, "limit": limit}},
        {"$project": {"_id": 0, "filename": 1, "ocr_text": 1, "vlm_description": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    return list(collection.aggregate(pipeline))

@observe(name="hybrid-vector-search")
def perform_hybrid_vector_search(vec1, path1, vec2, path2, limit=10):
    """🔥 執行雙向量混合搜尋 (使用 RRF 倒數排名融合)"""
    if not vec1 or not vec2: return []
    
    # 搜尋策略 A (拿前 20 名)
    res1 = list(collection.aggregate([
        {"$vectorSearch": {"index": "vector_index", "path": path1, "queryVector": vec1, "numCandidates": 100, "limit": 20}},
        {"$project": {"filename": 1, "ocr_text": 1, "vlm_description": 1}}
    ]))
    
    # 搜尋策略 B (拿前 20 名)
    res2 = list(collection.aggregate([
        {"$vectorSearch": {"index": "vector_index", "path": path2, "queryVector": vec2, "numCandidates": 100, "limit": 20}},
        {"$project": {"filename": 1, "ocr_text": 1, "vlm_description": 1}}
    ]))
    
    # RRF 計分 
    rrf_scores = {}
    docs = {}
    k = 60
    
    for rank, doc in enumerate(res1):
        fname = doc['filename']
        rrf_scores[fname] = rrf_scores.get(fname, 0) + 1.0 / (k + rank + 1)
        docs[fname] = doc
        
    for rank, doc in enumerate(res2):
        fname = doc['filename']
        rrf_scores[fname] = rrf_scores.get(fname, 0) + 1.0 / (k + rank + 1)
        docs[fname] = doc
        
    if not rrf_scores: return []
    
    sorted_fnames = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [docs[f] for f in sorted_fnames[:limit]]

@observe(as_type="generation", name="llm-rerank")
def llm_rerank(question, candidates):
    """使用 LLM 針對 top N 候選名單進行重新排序 (Rerank)"""
    if not candidates: 
        return "Not Found"
    if len(candidates) == 1: 
        return candidates[0]['filename']

    docs_str = ""
    for i, doc in enumerate(candidates):
        docs_str += f"--- Document {i+1} ---\nFilename: {doc['filename']}\nOCR: {doc.get('ocr_text', '')}\nDescription: {doc.get('vlm_description', '')}\n\n"

    prompt = f"""
    You are an expert at selecting the most relevant document to answer a user's question.
    Please review the following candidate documents and choose the most relevant one for the question.
    
    Question: {question}

    Candidates:
    {docs_str}

    Output ONLY the exact Filename of the best document. No other text.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Rerank error: {e}")
        return candidates[0]['filename']

@observe(as_type="generation", name="generate-rag-answer")
def generate_rag_answer(question, image_filename):
    if image_filename == "Not Found" or not image_filename:
        return "無法檢索到相關圖片"
        
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    if not os.path.exists(image_path): return "錯誤：找不到圖片"
        
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一個 RAG 助手。請根據圖片回答使用者的問題。"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"問題：{question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]}
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e: return f"生成錯誤: {e}"


# ================= 主程式 =================

@observe(name="process-question")
def process_question(item, results):
    q_text = item['question']
    gt_ans = item['answer']
    gt_img = item['source_filename']
    qid = item['id']
    
    # 1. 計算兩種基礎向量
    vec_semantic = get_text_embedding(q_text)
    vec_visual   = get_clip_text_embedding(q_text)
    
    # 2. 進行 RAG 的兩大核心搜尋：取得前 10 名
    # 第一種: 用文搜圖加上 OCR (及 VLM) 文字 hybrid
    # 注意：新 DB 中，整合的文本向量欄位叫做 'embedding'
    res_hybrid = perform_hybrid_vector_search(vec_semantic, "embedding", vec_visual, "image_vector", limit=10)
    
    # 第二種: 純用文搜圖 (Visual)
    res_visual = perform_single_vector_search(vec_visual, "image_vector", limit=10)
    
    # 3. 處理四種策略結果
    ans_hybrid = res_hybrid[0]['filename'] if res_hybrid else "Not Found"
    ans_visual = res_visual[0]['filename'] if res_visual else "Not Found"
    
    ans_hybrid_rerank = llm_rerank(q_text, res_hybrid[:5]) if res_hybrid else "Not Found"
    ans_visual_rerank = llm_rerank(q_text, res_visual[:5]) if res_visual else "Not Found"
    
    # 統一產生 RAG 回答並填入紀錄表
    def evaluate(best_filename, strat_name):
        ai_ans = generate_rag_answer(q_text, best_filename)
        is_hit = (best_filename == gt_img)
        
        results[strat_name].append({
            "ID": qid,
            "Question": q_text,
            "GT Answer": gt_ans,
            "GT Image": gt_img,
            "Retrieved Image": best_filename,
            "Hit": is_hit,
            "AI Answer": ai_ans
        })

    evaluate(ans_hybrid, "Hybrid")
    evaluate(ans_visual, "Visual")
    evaluate(ans_visual_rerank, "Visual_Rerank")
    evaluate(ans_hybrid_rerank, "Hybrid_Rerank")


def main():
    if not os.path.exists(JSON_FILE): return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    # 定義 4 種策略
    results = {
        "Hybrid": [],
        "Visual": [],
        "Visual_Rerank": [],
        "Hybrid_Rerank": []
    }
    
    print(f"🚀 開始評測 {len(qa_data)} 題 x 4 種策略...")
    
    for item in tqdm(qa_data, desc="評測進度"):
        process_question(item, results)
        
    get_client().flush() # 強制送出 Langfuse 事件

    # 4. 輸出 CSV 檔案
    print("\n💾 正在儲存 CSV 檔案...")
    for name, data in results.items():
        if not data: continue 
            
        df = pd.DataFrame(data)
        filename = f"rag_eval_4strats_{name.lower()}.csv"
        
        cols = ["ID", "Question", "GT Answer", "GT Image", "Retrieved Image", "Hit", "AI Answer"]
        df = df[cols]
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        
        hit_rate = df["Hit"].mean()
        print(f"   📄 {filename} (命中率: {hit_rate:.1%})")

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()