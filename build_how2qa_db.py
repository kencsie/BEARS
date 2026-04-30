import os
import time
import base64
import subprocess
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from langfuse.openai import OpenAI
from langfuse import observe
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ================= 設定區 =================
VIDEO_DIR = "How2QA_100_Videos"
DB_NAME = "how2qa_rag_db"
COLLECTION_NAME = "video_segments"
FFMPEG_PATH = r"C:\Miniconda3\envs\code\Lib\site-packages\static_ffmpeg\bin\win32\ffmpeg.exe"

# 確保環境變數關閉 symlinks 警告並離線使用 transformers（如果之前已下載）
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST

# === 初始化 ===
print("Initializing system...")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("   Loading CLIP model...")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

print(f"Ready! Knowledge base: [{DB_NAME}] -> [{COLLECTION_NAME}]\n")


# === Helper functions ===

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_audio(video_path: str, audio_path: str) -> bool:
    """提取影片聲音為 MP3，用於 Whisper 語音辨識"""
    cmd = [
        FFMPEG_PATH if os.path.exists(FFMPEG_PATH) else "ffmpeg",
        "-y", "-i", video_path,
        "-q:a", "0", "-map", "a",
        audio_path,
        "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True)
        return os.path.exists(audio_path)
    except subprocess.CalledProcessError:
        return False

def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """透過指定的時間戳 (秒) 截取一張畫面"""
    cmd = [
        FFMPEG_PATH if os.path.exists(FFMPEG_PATH) else "ffmpeg",
        "-y", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
        "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError:
        return False

@observe(as_type="generation", name="whisper_audio_transcription")
def get_whisper_segments(audio_path: str):
    """呼叫 OpenAI Whisper API 進行語音轉文字並取得時間戳區段"""
    try:
        with open(audio_path, "rb") as f:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=f, 
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        # API 回傳的 segments 會包含 start, end, text
        return transcription.segments if hasattr(transcription, 'segments') else transcription.get('segments', [])
    except Exception as e:
        print(f"   Warning Whisper Error: {e}")
        return []

@observe(as_type="generation", name="vlm_frame_description")
def get_vlm_description(base64_img: str, audio_text: str) -> str:
    """GPT-4o-mini Vision: 產生影像描述，並將語音內容作為 Context"""
    prompt = (
        "You are a multimodal AI assistant analyzing a video frame. "
        "Describe the visual content of this frame in detail. "
        "Focus on the people, objects, actions, and the environment. "
    )
    if audio_text:
        prompt += f"\nFor context, the audio transcript at this exact moment is: '{audio_text}'. Do NOT just repeat the transcript; explain what is visibly happening in relation to the audio."
        
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    ],
                }
            ],
            max_tokens=300,
            name="vlm_frame_description"
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   Warning VLM Error: {e}")
        return ""

@observe(as_type="generation", name="openai_text_embedding")
def get_openai_text_embedding(text: str):
    """文字 Embedding (給 OpenAI)"""
    if not text:
        return None
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
            name="text_embedding"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"   Warning Embedding Error: {e}")
        return None

def get_clip_image_embedding(image_path: str):
    """CLIP 圖像 Embedding"""
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs.pooler_output
        image_features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return image_features[0].tolist()
    except Exception as e:
        print(f"   Warning CLIP Error: {e}")
        return None


# === Main pipeline ===

@observe(name="process_how2qa_video")
def process_video(filename: str):
    video_path = os.path.join(VIDEO_DIR, filename)
    audio_path = f"temp_audio_{filename}.mp3"
    
    # 1. 抽取聲音
    has_audio = extract_audio(video_path, audio_path)
    segments = []
    if has_audio:
        segments = get_whisper_segments(audio_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    # 如果影片沒聲音，或是 whisper 沒抓到文字，我們手動切一個片段來代表整支短片
    if not segments:
        segments = [{"start": 0.0, "end": 5.0, "text": ""}]
        
    processed_count = 0
    # 2. 針對每個語音片段 (Segment) 進行處理
    for idx, seg in enumerate(segments):
        # dict / object handle
        start = seg['start'] if isinstance(seg, dict) else seg.start
        end = seg['end'] if isinstance(seg, dict) else seg.end
        text = seg['text'] if isinstance(seg, dict) else seg.text
        text = text.strip()
        
        # 抓取該片段的「中間時間點」當作代表畫面
        mid_time = (start + end) / 2.0
        frame_path = f"temp_frame_{filename}_{idx}.jpg"
        
        # 截取畫面
        if not extract_frame(video_path, mid_time, frame_path):
            continue
            
        # A. Image → Base64
        base64_str = encode_image_base64(frame_path)
        
        # B. VLM Description (GPT-4o-mini Vision)
        vlm_desc = get_vlm_description(base64_str, text)
        
        # C. Vectors
        combined_text = f"Audio Transcript: {text}\nVisual Description: {vlm_desc}"
        text_vec = get_openai_text_embedding(combined_text)
        img_vec = get_clip_image_embedding(frame_path)
        
        # D. Upsert to MongoDB
        doc_id = f"{filename}_seg_{idx}"
        doc_data = {
            "segment_id": doc_id,
            "filename": filename,
            "start_time": start,
            "end_time": end,
            "mid_time": mid_time,
            "transcript": text,
            "vlm_description": vlm_desc,
            "image_base64": base64_str,
            "embedding": text_vec,      # 描述向量 (包含語音+影像描述)
            "image_vector": img_vec,    # 圖像向量 (CLIP)
            "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        collection.replace_one({"segment_id": doc_id}, doc_data, upsert=True)
        processed_count += 1
        
        # 清理暫存圖片
        if os.path.exists(frame_path):
            os.remove(frame_path)
            
    return processed_count

def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Folder not found: {VIDEO_DIR}")
        return

    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")]
    print(f"Found {len(videos)} videos. Starting video ingestion...")

    total_segments = 0
    for filename in tqdm(videos, desc="Processing Videos"):
        segs = process_video(filename)
        total_segments += segs

    # 確保 Langfuse logs 全部送出
    try:
        from langfuse import get_client
        lf = get_client()
        lf.flush()
    except:
        pass

    print(f"\nIngestion complete! Processed and uploaded {total_segments} video segments.")
    print(f"Remember to create Vector Indexes in MongoDB Atlas for [{DB_NAME}.{COLLECTION_NAME}] (text and image vectors)!")

if __name__ == "__main__":
    main()
