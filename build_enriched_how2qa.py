import os
import json
import base64
import time
import subprocess
from pymongo import MongoClient
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from langfuse.openai import OpenAI as LangfuseOpenAI

load_dotenv()

# Langfuse Compatibility
try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    try:
        from langfuse import observe, langfuse_context
    except ImportError:
        def observe(*args, **kwargs): return lambda f: f
        langfuse_context = None

load_dotenv()

# Config
import torch
from transformers import CLIPProcessor, CLIPModel

# CLIP Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Config
FFMPEG_PATH = r"C:\Miniconda3\envs\code\Lib\site-packages\static_ffmpeg\bin\win32\ffmpeg.exe"
INPUT_DIR = "C:\\m_rag_webqa\\How2QA_100_Videos"
INPUT_JSON = "how2qa_100.json"
OUTPUT_JSON = "how2qa_100_enriched.json"
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
lf_client = LangfuseOpenAI(api_key=OPENAI_API_KEY) # For tracked LLM calls

db_client = MongoClient(MONGO_URI)
db = db_client["how2qa_enriched_db"]
collection = db["video_segments"]

def get_clip_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features[0].cpu().detach().numpy().tolist()

def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_whisper_transcript(video_path, vid):
    audio_path = f"tmp_{vid}.mp3"
    try:
        subprocess.run([FFMPEG_PATH, "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", audio_path], 
                       stderr=subprocess.DEVNULL, check=True)
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        
        # Give Windows a tiny bit of time to release the file handle
        time.sleep(0.5)
        try: os.remove(audio_path)
        except: pass
        
        return transcript.text
    except Exception as e:
        try:
            if os.path.exists(audio_path): os.remove(audio_path)
        except: pass
        return ""

@observe(name="vlm_analysis")
def get_vlm_description(base64_img, original_q):
    prompt = f"Describe this video frame in detail. Focus on background, colors, tools used, and any specific text or actions. The original question was: '{original_q}'."
    try:
        response = lf_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except: return ""

@observe(name="question_enrichment")
def enrich_question(original_q, vlm_desc):
    prompt = f"Given this visual description: '{vlm_desc}', rewrite the original question '{original_q}' to include specific visual details that make it unique and easier to search for. Keep it concise but discriminative."
    try:
        response = lf_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content
    except: return original_q

def get_openai_text_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

@observe(name="full_ingestion")
def process_how2qa():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched_data = []
    print(f"Dense Ingestion (5 Segments/Video) & Enriched Question Generation...")

    for item in tqdm(data):
        vid = item['video_id']
        start = item['start_time']
        end = item['end_time']
        video_path = os.path.join(INPUT_DIR, f"{vid}_{start}_{end}.mp4")
        if not os.path.exists(video_path): continue

        # 1. Get Transcript (once per video)
        transcript = get_whisper_transcript(video_path, vid)
        
        # 2. Dense Sampling: 5 Segments per video
        segments_vlm = []
        for i in range(5):
            # Calculate timestamp for 5 segments across the duration
            timestamp = i * 2 # Every 2 seconds (assuming 10s avg video)
            frame_path = f"tmp_{vid}_{i}.jpg"
            subprocess.run([FFMPEG_PATH, "-y", "-i", video_path, "-ss", str(timestamp), "-vframes", "1", frame_path], stderr=subprocess.DEVNULL)
            
            if os.path.exists(frame_path):
                b64_img = encode_image_base64(frame_path)
                vlm_desc = get_vlm_description(b64_img, item['question'])
                # Get CLIP Image Vector
                clip_vec = get_clip_image_embedding(frame_path)
                
                segments_vlm.append(vlm_desc)
                
                # Embedding for THIS segment
                combined_info = f"Transcript: {transcript}\nVisual Segment: {vlm_desc}"
                vec = get_openai_text_embedding(combined_info)
                
                # 3. Save Segment to DB (Adding clip_embedding)
                doc = {
                    "video_id": vid,
                    "segment_idx": i,
                    "timestamp": timestamp,
                    "vlm_description": vlm_desc,
                    "transcript": transcript,
                    "embedding": vec,
                    "clip_embedding": clip_vec,
                    "image_base64": b64_img,
                    "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                collection.insert_one(doc)
                os.remove(frame_path)
        
        # 4. Use the first segment to enrich the question for JSON only
        if segments_vlm:
            enriched_q = enrich_question(item['question'], segments_vlm[0])
            item['enriched_question'] = enriched_q
        
        enriched_data.append(item)
        
        # Save JSON incrementally
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_how2qa()
