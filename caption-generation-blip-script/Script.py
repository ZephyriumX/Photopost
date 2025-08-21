import os
import argparse
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai  # OpenRouter API works with OpenAI SDK
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from dotenv import load_dotenv

#  CONFIG 

# Set OpenRouter API Key and Base URL
load_dotenv() 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please set it in your .env file.")

openai.api_key = OPENROUTER_API_KEY
openai.api_base = "https://openrouter.ai/api/v1"

# Use device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=False)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Initialize CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

#  FUNCTIONS 

def load_prompt(prompt_path="prompt.txt"):
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "bossy mafia in black tuxedo"

def call_llm(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="mistralai/mistral-7b-instruct",  # or any other available model
        messages=[
            {"role": "system", "content": "You are an Instagram influencer who writes creative, stylish captions."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def describe_image_blip(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(raw_image ,return_tensors="pt").to(device)
    out= blip_model.generate(
    **inputs,
    num_beams=5,
    max_length=50,
    early_stopping=True)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def generate_caption(user_prompt, image_description, tone=None):
    full_prompt = f"""
    Image Description: {image_description}
    Vibe Prompt: {user_prompt}
    """
    if tone:
        full_prompt += f"Tone: {tone}\n"
    full_prompt += "Create 1 creative, stylish Instagram caption using emojis and up to 2 hashtags."

    return call_llm(full_prompt)

def score_with_clip(image_path, captions):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    MAX_CLIP_LEN = 75
    safe_captions = [cap if len(cap) <= MAX_CLIP_LEN else cap[:MAX_CLIP_LEN] for cap in captions]
    text_tokens = clip.tokenize(safe_captions).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)
    scores = similarity.tolist()
    return list(zip(captions, scores))

# MAIN FLOW 

def main(user_prompt=None, tone=None, top_k=3):
    if not user_prompt:
        user_prompt = load_prompt()

    image_dir = "./session_1"  # Folder should exist in same directory
    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png"))
    ])

    for img_path in image_paths:
        print(f"\n🖼️ Processing: {img_path}")

        # 1. Describe the image
        description = describe_image_blip(img_path)
        print(f"📝 Description: {description}")

        # 2. Generate 5 captions
        candidates = [generate_caption(user_prompt, description, tone) for _ in range(5)]

        # 3. Score with CLIP
        ranked = score_with_clip(img_path, candidates)
        ranked.sort(key=lambda x: x[1], reverse=True)

        # 4. Show top results
        print("\n🏆 Top Captions:")
        for i, (cap, score) in enumerate(ranked[:top_k], 1):
            print(f"{i}. ({score:.3f}) {cap}")

# ------------------------- CLI ENTRY -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=False, help="Vibe prompt like 'bossy mafia in black tux'")
    parser.add_argument('--tone', type=str, help="Optional tone: emotional, edgy, funny, etc.")
    parser.add_argument('--top_k', type=int, default=3, help="Top N captions to return per image")

    args = parser.parse_args()
    main(args.prompt, args.tone, args.top_k)






#3 or 5 image or 5 in total for all images - future improvement
#better blip description or its okay? - future improvement

#v2
#🔁 “Another 5” → regenerate + re-rank captions
#💾 Save top captio n + image path + prompt + tone to local DB (SQLite or JSON)
#📂 Batch mode for top-N best images from gallery
