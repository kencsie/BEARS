import os
import json
import csv
import torch
import re
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from transformers import CLIPProcessor, CLIPModel
from langfuse.openai import OpenAI as LangfuseOpenAI

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    def observe(*args, **kwargs): return lambda f: f
    langfuse_context = None

load_dotenv(find_dotenv())

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "how2qa_enriched_db"
COLLECTION_NAME = "video_segments"
INPUT_JSON = os.path.join(BASE_DIR, "data", "how2qa_100_rewritten.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "how2qa_rewritten_results.csv")

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

@observe(name="how2qa_expand_query")
def expand_query(question):
    """Generate 2 alternative phrasings to improve search recall."""
    try:
        response = lf_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"Given this video search question, generate 2 alternative search queries "
                f"that emphasize different keywords or angles. Be concise.\n"
                f"Question: {question}\n"
                f"Return only the 2 queries, one per line, no numbering."
            )}],
            max_tokens=100,
            temperature=0.3
        )
        lines = response.choices[0].message.content.strip().split('\n')
        extras = [l.strip() for l in lines if l.strip()][:2]
        return [question] + extras
    except:
        return [question]

@observe(name="how2qa_rerank_and_qa")
def rerank_and_qa(query_text, candidates):
    """
    Rerank candidates via per-candidate scoring, then generate a short answer.
    Returns: (best_vid, short_answer, reasoning)
    """
    if not candidates: return None, "No answer", "No candidates"

    context_list = []
    for i, cand in enumerate(candidates):
        context_list.append(f"--- Candidate [{i}] ---\n"
                           f"Visual: {cand.get('vlm_description', 'N/A')}\n"
                           f"Audio: {cand.get('transcript', 'N/A')}")

    context_str = "\n".join(context_list)

    prompt = (
        f"User Question: '{query_text}'\n\n"
        f"Available Video Contexts:\n{context_str}\n\n"
        "TASK:\n"
        "Score each candidate from 0-10 on how well it answers the question.\n"
        "Key rules:\n"
        "- Give 10 only if the candidate SPECIFICALLY answers the exact question.\n"
        "- Candidates with similar topics but missing the specific detail asked should score ≤5.\n"
        "- If multiple candidates cover the same topic, differentiate by which one has the EXACT answer.\n\n"
        "Then select the highest-scoring candidate and provide a short answer.\n\n"
        "FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n"
        "Scores: [score0, score1, score2, ...] (one number per candidate)\n"
        "Index: [Best candidate number]\n"
        "Answer: [Short Answer, max 15 words]\n"
        "Reason: [One sentence]"
    )

    try:
        response = lf_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0
        )
        content = response.choices[0].message.content

        idx = 0
        short_ans = "N/A"
        reason = "N/A"

        # Try to pick best by scores first
        scores_match = re.search(r"Scores:\s*\[([^\]]+)\]", content)
        if scores_match:
            try:
                scores = [float(s.strip()) for s in scores_match.group(1).split(',')]
                idx = int(scores.index(max(scores)))
            except:
                pass

        # Fallback to explicit Index field
        idx_match = re.search(r"Index:\s*(\d+)", content)
        if idx_match:
            idx = int(idx_match.group(1))

        ans_match = re.search(r"Answer:\s*(.*)", content)
        if ans_match: short_ans = ans_match.group(1).strip()

        reason_match = re.search(r"Reason:\s*(.*)", content)
        if reason_match: reason = reason_match.group(1).strip()

        best_vid = candidates[idx]['video_id'] if 0 <= idx < len(candidates) else candidates[0]['video_id']
        return best_vid, short_ans, reason

    except Exception as e:
        print(f"Rerank/QA Error: {e}")
        return candidates[0]['video_id'], "Error", str(e)

RRF_K = 60  # standard RRF constant

def _rrf_merge(ranked_lists):
    """Reciprocal Rank Fusion: combine multiple ranked lists into one score dict."""
    scores = {}
    meta = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            vid = doc['video_id']
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if vid not in meta:
                meta[vid] = doc
    return scores, meta

@observe(name="how2qa_hybrid_search")
def search_video_hybrid(query_text):
    queries = expand_query(query_text)
    all_lists = []

    for q in queries:
        openai_vec = get_query_embedding(q)
        clip_vec = get_clip_text_embedding(q)
        if openai_vec is None:
            continue

        # Path A: OpenAI embedding
        pipeline_a = [
            {"$vectorSearch": {"index": "hybrid_index", "path": "embedding", "queryVector": openai_vec, "numCandidates": 500, "limit": 15}},
            {"$project": {"video_id": 1, "vlm_description": 1, "transcript": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]
        results_a = list(collection.aggregate(pipeline_a))
        if results_a:
            all_lists.append(results_a)

        # Path B: CLIP embedding
        if clip_vec:
            pipeline_b = [
                {"$vectorSearch": {"index": "hybrid_index", "path": "clip_embedding", "queryVector": clip_vec, "numCandidates": 500, "limit": 15}},
                {"$project": {"video_id": 1, "vlm_description": 1, "transcript": 1, "score": {"$meta": "vectorSearchScore"}}}
            ]
            try:
                results_b = list(collection.aggregate(pipeline_b))
                if results_b:
                    all_lists.append(results_b)
            except:
                pass

    if not all_lists:
        return []

    rrf_scores, meta = _rrf_merge(all_lists)
    final_list = sorted(
        [{"video_id": vid, **meta[vid], "score": score} for vid, score in rrf_scores.items()],
        key=lambda x: x['score'], reverse=True
    )
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
        question = item['rewritten_question']
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
