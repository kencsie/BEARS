import os
import time
import base64
import torch
from dotenv import load_dotenv
from pymongo import MongoClient
from langfuse.openai import OpenAI   # drop-in wrapper: auto-captures model, tokens & cost in Langfuse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm
from langfuse import observe, get_client   # Langfuse v4 API

# ================= 設定區 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "infographic_rag_data_strict")

# 2. 資料庫設定
DB_NAME = "infographic_rag_strict_db"
COLLECTION_NAME = "strict_data"

# ================= 環境設定 =================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

load_dotenv()
MONGO_URI           = os.getenv("MONGO_URI")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not MONGO_URI or not OPENAI_API_KEY:
    print("❌ Error: Please check MONGO_URI and OPENAI_API_KEY in .env")
    exit()

if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
    print("⚠️  Warning: LANGFUSE keys not set — tracing disabled.")
else:
    # v4: set env vars so get_client() picks them up automatically
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_HOST"]       = LANGFUSE_HOST

# === 初始化 ===
print("⏳ Initializing system...")

mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[DB_NAME]
collection   = db[COLLECTION_NAME]

openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("   ↳ Loading CLIP model...")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model      = CLIPModel.from_pretrained(clip_model_name)
clip_processor  = CLIPProcessor.from_pretrained(clip_model_name)

print("   ↳ Loading RapidOCR model...")
ocr_engine = RapidOCR()

print(f"✅ Ready! Knowledge base: [{DB_NAME}] -> [{COLLECTION_NAME}]\n")


# === Helper functions ===

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_ocr_text_rapid(image_path: str) -> str:
    """RapidOCR: extract raw text from image."""
    try:
        result, _ = ocr_engine(image_path)
        if not result:
            return ""
        return " ".join(line[1] for line in result)
    except Exception as e:
        print(f"   ⚠️ OCR Error: {e}")
        return ""


@observe(as_type="generation", name="vlm-description")
def get_vlm_description(base64_img: str, filename: str = "unknown") -> str:
    """
    GPT-4o Vision: Generate a detailed English description of the infographic.
    🔥 Prompt strategy: RAG-optimized — emphasize data, entities, and chart structure.
    """
    prompt = (
        "You are a professional infographic analyst. "
        "To enable accurate retrieval, describe the image in detail:\n"
        "1. [Title & Topic]: What is this infographic about?\n"
        "2. [Key Data]: Extract all specific values — years, percentages, amounts, rankings, counts.\n"
        "3. [Entity Recognition]: List all countries, companies, products, or proper nouns mentioned.\n"
        "4. [Chart Structure]: Is this a bar chart, map, flowchart, pie chart, or other type?\n\n"
        "Respond in English only. Do NOT use Markdown. Output plain text paragraphs."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    ],
                }
            ],
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"   ⚠️ VLM Error: {e}")
        return ""


@observe(as_type="generation", name="text-embedding")
def get_openai_text_embedding(text: str, filename: str = "unknown"):
    """Compute text embedding vector (for semantic search)."""
    if not text:
        return None
    try:
        text = text.replace("\n", " ")
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"   ⚠️ Embedding Error: {e}")
        return None


def get_clip_image_embedding(image_path: str):
    """Compute image visual vector (for visual similarity search)."""
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
        print(f"   ⚠️ CLIP Error: {e}")
        return None


# === Main pipeline ===

@observe(name="ingest-image")
def process_image(filename: str) -> bool:
    """Process a single image: OCR → VLM → Embedding → MongoDB upsert."""
    file_path = os.path.join(IMAGE_FOLDER, filename)

    # A. Image → Base64
    base64_str = encode_image_base64(file_path)

    # B. OCR
    ocr_txt = get_ocr_text_rapid(file_path)

    # C. VLM description (English)
    vlm_desc = get_vlm_description(base64_str, filename=filename)

    # D. Vectors
    combined_text = f"OCR text: {ocr_txt}\nImage description: {vlm_desc}"
    combined_vec  = get_openai_text_embedding(combined_text, filename=filename)
    img_vec       = get_clip_image_embedding(file_path)

    # E. Upsert to MongoDB
    if ocr_txt or vlm_desc:
        doc_data = {
            "filename":        filename,
            "file_path":       file_path,
            "upload_date":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_base64":    base64_str,
            "ocr_text":        ocr_txt,
            "vlm_description": vlm_desc,
            "embedding":       combined_vec,
            "image_vector":    img_vec,
            "metadata": {
                "source": "InfographicVQA_Strict",
                "type":   "infographic",
            },
        }
        collection.replace_one({"filename": filename}, doc_data, upsert=True)
        return True
    else:
        print(f"⚠️  Content extraction failed (OCR & VLM both empty): {filename}")
        return False


def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Folder not found: {IMAGE_FOLDER}")
        return

    files = sorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"📂 Found {len(files)} images. Starting knowledge base ingestion...\n")

    success_count = 0
    for filename in tqdm(files, desc="Ingesting"):
        if collection.find_one({"filename": filename}):
            continue
        if process_image(filename):
            success_count += 1

    # Flush all pending Langfuse events before exit
    lf = get_client()
    lf.flush()

    print(f"\n🎉 Ingestion complete! Uploaded {success_count} documents.")
    print(f"👉 Remember to create a Vector Index in MongoDB Atlas for [{DB_NAME}.{COLLECTION_NAME}]!")


if __name__ == "__main__":
    main()