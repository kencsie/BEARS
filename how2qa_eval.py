import os
import json
import csv
import time
import torch
import re
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from langfuse.openai import OpenAI as LangfuseOpenAI

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    def observe(*args, **kwargs): return lambda f: f
    langfuse_context = None

load_dotenv()

# Config
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "how2qa_enriched_db"
COLLECTION_NAME = "video_segments"
INPUT_JSON = "how2qa_100_enriched.json"
OUTPUT_CSV = "how2qa_qa_comparison_results.csv"

# Clients
lf_client = LangfuseOpenAI(api_key=OPENAI_API_KEY)
db_client = MongoClient(MONGO_URI)
db = db_client[DB_NAME]
collection = db[COLLECTION_NAME]

# CLIP Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

@observe(name="get_search_embedding")
def get_query_embedding(text):
    try:
        response = lf_client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except: return None

@observe(name="get_clip_text_embedding")
def get_clip_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
        if hasattr(outputs, "text_embeds"):
            text_features = outputs.text_embeds
        elif isinstance(outputs, torch.Tensor):
            text_features = outputs
        else:
            text_features = outputs[0]
            
    if text_features.ndim == 3:
        text_features = text_features.mean(dim=1)
    elif text_features.ndim == 1:
        text_features = text_features.unsqueeze(0)
    
    text_features = text_features.view(-1, 512)[0:1]
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    vec = text_features.flatten().cpu().detach().numpy().tolist()
    return [float(x) for x in vec]

@observe(name="how2qa_rerank_and_qa")
def rerank_and_qa(query_text, candidates):
    """
    Rerank candidates AND generate a short answer.
    Returns: (best_vid, short_answer, reasoning)
    """
    if not candidates: return None, "No answer", "No candidates"
    
    context_list = []
    for i, cand in enumerate(candidates):
        context_list.append(f"--- Candidate [{i}] (ID: {cand['video_id']}) ---\n"
                           f"Visual: {cand.get('vlm_description', 'N/A')}\n"
                           f"Audio: {cand.get('transcript', 'N/A')}")
    
    context_str = "\n".join(context_list)
    
    prompt = (
        f"User Question: '{query_text}'\n\n"
        f"Available Video Contexts:\n{context_str}\n\n"
        "TASK:\n"
        "1. Select the BEST candidate index (0-9) that answers the question.\n"
        "2. Provide a very SHORT answer (max 15 words) to the question based on that video.\n"
        "3. Provide a brief 1-sentence reason for your choice.\n\n"
        "FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n"
        "Index: [Number]\n"
        "Answer: [Short Answer]\n"
        "Reason: [Reasoning]"
    )
    
    try:
        response = lf_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        content = response.choices[0].message.content
        
        # Parse results
        idx = 0
        short_ans = "N/A"
        reason = "N/A"
        
        idx_match = re.search(r"Index:\s*(\d+)", content)
        if idx_match: idx = int(idx_match.group(1))
        
        ans_match = re.search(r"Answer:\s*(.*)", content)
        if ans_match: short_ans = ans_match.group(1).strip()
        
        reason_match = re.search(r"Reason:\s*(.*)", content)
        if reason_match: reason = reason_match.group(1).strip()
        
        best_vid = candidates[idx]['video_id'] if 0 <= idx < len(candidates) else candidates[0]['video_id']
        return best_vid, short_ans, reason
        
    except Exception as e:
        print(f"Rerank/QA Error: {e}")
        return candidates[0]['video_id'], "Error", str(e)

@observe(name="how2qa_hybrid_search")
def search_video_hybrid(query_text):
    openai_vec = get_query_embedding(query_text)
    clip_vec = get_clip_text_embedding(query_text)
    if openai_vec is None: return []

    # Path A: OpenAI
    pipeline_a = [
        {"$vectorSearch": {"index": "hybrid_index", "path": "embedding", "queryVector": openai_vec, "numCandidates": 200, "limit": 10}},
        {"$project": {"video_id": 1, "vlm_description": 1, "transcript": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    results_a = list(collection.aggregate(pipeline_a))
    
    # Path B: CLIP
    results_b = []
    if clip_vec:
        pipeline_b = [
            {"$vectorSearch": {"index": "hybrid_index", "path": "clip_embedding", "queryVector": clip_vec, "numCandidates": 200, "limit": 10}},
            {"$project": {"video_id": 1, "vlm_description": 1, "transcript": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]
        try:
            results_b = list(collection.aggregate(pipeline_b))
        except: pass
    
    combined_results = {}
    for r in results_a:
        combined_results[r['video_id']] = {**r, "score": r['score'] * 0.7}
    for r in results_b:
        if r['video_id'] in combined_results:
            combined_results[r['video_id']]['score'] += r['score'] * 0.3
        else:
            combined_results[r['video_id']] = {**r, "score": r['score'] * 0.3}
            
    final_list = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
    return final_list[:10]

@observe(name="how2qa_evaluation_v2")
def run_eval():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    top1_correct = 0
    top10_correct = 0
    
    print(f"Starting Video-QA RAG Evaluation (Hybrid + Rerank + QA)...")
    for item in tqdm(data):
        target_vid = item['video_id']
        question = item['enriched_question']
        gt_answer = item.get('answer', 'N/A')
        
        # 1. Hybrid Search
        candidates = search_video_hybrid(question)
        top10_ids = [c['video_id'] for c in candidates]
        
        # 2. Rerank & Answer
        retrieved_vid, llm_answer, reason = rerank_and_qa(question, candidates)
        
        is_correct_id = (retrieved_vid == target_vid)
        if is_correct_id: top1_correct += 1
        if target_vid in top10_ids: top10_correct += 1
        
        results.append({
            "Question": question,
            "GT_Answer": gt_answer,
            "LLM_Short_Answer": llm_answer,
            "ID_Correct": "✅" if is_correct_id else "❌",
            "GT_Video_ID": target_vid,
            "LLM_Picked_ID": retrieved_vid,
            "Reasoning": reason,
            "Top10_Recall": "YES" if target_vid in top10_ids else "NO"
        })

    # Stats
    acc = (top1_correct / len(data)) * 100
    recall = (top10_correct / len(data)) * 100
    print(f"\nFinal Stats: Accuracy {acc:.2f}% | Recall {recall:.2f}%")

    # CSV Output
    if results:
        keys = results[0].keys()
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Detailed QA results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_eval()
