import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 設定
JSON_FILE = "spoken_squad_100.json"
AUDIO_DIR = "Spoken_SQuAD_Audio"

if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

def generate_audio():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Start generating {len(data)} English audio files...")

    for i, item in enumerate(tqdm(data)):
        audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.mp3")
        
        # skip if exists
        if os.path.exists(audio_path):
            continue

        context_text = item["context"]
        
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy", 
                input=context_text[:4096] 
            )
            response.stream_to_file(audio_path)
        except Exception as e:
            print(f"\nError generating audio_{i}: {e}")

    print(f"\nAll done! Audio files are in {AUDIO_DIR}/")

if __name__ == "__main__":
    generate_audio()
