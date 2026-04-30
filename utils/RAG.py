import os
import json
import base64
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ================= 設定區 =================
# 1. 資料庫設定
DB_NAME = "infographic_rag_strict_db"
COLLECTION_NAME = "strict_data" 

# 2. 檔案路徑
JSON_FILE = "infographic_rag_data_strict/ground_truth.json"
IMAGE_FOLDER = "infographic_rag_data_strict" # 圖片資料夾

# ================= 環境初始化 =================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not MONGO_URI or not OPENAI_API_KEY:
    print("❌ 錯誤：請檢查 .env 檔案")
    exit()

print("⏳ 系統初始化中...")

# 連線
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 載入 CLIP (用於 Visual Search)
print("   ↳ 載入 CLIP 模型 (用於以文搜圖)...")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

print("✅ 初始化完成！準備開始評測...\n")

# ================= 核心函式 =================

def get_text_embedding(text):
    """計算 text-embedding-3-small 向量"""
    if not text: return []
    try:
        text = text.replace("\n", " ")
        res = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        return res.data[0].embedding
    except: return []

def get_clip_text_embedding(text):
    """計算 CLIP 文字向量 (終極修復版：處理 Tensor 格式與英文翻譯)"""
    if not text: return []
    
    try:
        # 1. 將中文翻譯成英文 (CLIP 只懂英文)
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translation engine. Translate the user's Chinese question into simple, highly descriptive English keywords for image retrieval. Do not answer the question."},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0.1
        )
        en_text = res.choices[0].message.content.strip()
        
        # 2. 文字轉 Tensor (加入 truncation 防止超長崩潰)
        inputs = clip_processor(
            text=[en_text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        )
        
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
            
        # 3. 🔥 關鍵修復：處理不同版本的 Transformers 回傳格式
        if not isinstance(outputs, torch.Tensor):
            if hasattr(outputs, 'text_embeds'):
                text_features = outputs.text_embeds
            elif hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            else:
                #  fallback: 如果還是找不到，取第一個元素
                text_features = outputs[0]
        else:
            text_features = outputs
            
        # 4. 正規化並轉成 list
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].tolist()
        
    except Exception as e:
        print(f"\n⚠️ CLIP 向量計算失敗: {e}")
        return []

def perform_vector_search(vector, path, limit=1):
    """執行 MongoDB Vector Search"""
    if not vector: return None
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index", 
                "path": path, 
                "queryVector": vector,
                "numCandidates": 100,
                "limit": limit
            }
        },
        {"$project": {"_id": 0, "filename": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None

def generate_rag_answer(question, image_filename):
    """RAG 生成：根據檢索到的圖片回答問題"""
    image_path = os.path.join(IMAGE_FOLDER, image_filename)
    if not os.path.exists(image_path): return "錯誤：找不到圖片"
        
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一個 RAG 助手。請根據圖片回答使用者的問題。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"問題：{question}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成錯誤: {e}"

# ================= 主程式 =================

def main():
    if not os.path.exists(JSON_FILE):
        print("❌ 找不到測試檔")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    # 準備 4 個容器來裝結果
    results = {
        "Mixed": [],
        "OCR": [],
        "VLM": [],
        "Visual": []
    }
    
    print(f"🚀 開始評測 {len(qa_data)} 題 x 4 策略...")

    for item in tqdm(qa_data, desc="評測進度"):
        q_text = item['question']
        gt_ans = item['answer']
        gt_img = item['source_filename']
        qid = item['id']
        
        # 1. 準備向量
        vec_semantic = get_text_embedding(q_text)    # 語意向量
        vec_visual = get_clip_text_embedding(q_text) # 視覺向量 (以文搜圖)
        
        # 2. 定義 4 種策略的參數
        strategies = {
            # "Mixed":  {"path": "embedding",      "vec": vec_semantic},
            # "OCR":    {"path": "embedding_ocr",  "vec": vec_semantic},
            # "VLM":    {"path": "embedding_vlm",  "vec": vec_semantic},
            "Visual": {"path": "image_vector",   "vec": vec_visual}
        }
        
        # 3. 執行 4 種策略
        for name, strategy in strategies.items():
            # A. 檢索
            retrieved = perform_vector_search(strategy["vec"], strategy["path"])
            
            if retrieved:
                r_img = retrieved['filename']
                score = retrieved['score']
                # B. 生成 (呼叫 GPT-4o)
                ai_ans = generate_rag_answer(q_text, r_img)
            else:
                r_img = "Not Found"
                score = 0
                ai_ans = "無法檢索到相關圖片"
            
            # C. 判斷命中 (Hit)
            is_hit = (r_img == gt_img)
            
            # D. 記錄結果
            results[name].append({
                "ID": qid,
                "Question": q_text,
                "GT Answer": gt_ans,
                "GT Image": gt_img,         # 這裡放標準答案圖片，方便比對
                "Retrieved Image": r_img,   # 這裡放檢索到的圖片
                "Hit": is_hit,              # 是否命中
                "Score": score,
                "AI Answer": ai_ans
            })

    # 4. 輸出 4 個 CSV 檔案
    print("\n💾 正在儲存 CSV 檔案...")
    
    for name, data in results.items():
        if not data:
            continue
        df = pd.DataFrame(data)
        filename = f"rag_eval_{name.lower()}.csv"
        
        # 調整欄位順序
        cols = ["ID", "Question", "GT Answer", "GT Image", "Retrieved Image", "Hit", "Score", "AI Answer"]
        df = df[cols]
        
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        
        # 計算命中率
        hit_rate = df["Hit"].mean()
        print(f"   📄 {filename} (命中率: {hit_rate:.1%})")

    print("\n🎉 全部完成！請查看生成的 4 個 CSV 檔。")

if __name__ == "__main__":
    main()