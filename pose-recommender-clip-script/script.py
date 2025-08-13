import os
import re
import requests
import uuid
import shutil
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def score_image_quality(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = np.mean(gray)
    contrast = np.std(gray)
    return sharp, bright, contrast

def fetch_images(prompt, num_images=10, session_index=0):
    folder = f"temp_images_{uuid.uuid4().hex[:6]}"
    os.makedirs(folder, exist_ok=True)

    params = {
        "q": prompt,
        "tbm": "isch",
        "ijn": "0",  # batch index
        "api_key": SERPAPI_KEY,
    }
    res = requests.get("https://serpapi.com/search", params=params).json()

    all_images = res.get("images_results", [])
    start = session_index * num_images
    end = start + num_images
    image_links = [img["original"] for img in all_images[start:end]]

    saved_files = []
    for i, link in enumerate(image_links):
        try:
            data = requests.get(link).content
            img = Image.open(BytesIO(data)).convert("RGB")
            path = os.path.join(folder, f"img_{i+1}.jpg")
            img.save(path)
            saved_files.append(path)
            print(f"✅ Saved {path}")
        except Exception as e:
            print(f"❌ Failed to save image {i+1}: {e}")
    return saved_files, folder

def get_clip_score(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    return float(probs[0].item())

# 🔹 New: Auto critical keyword extractor
def extract_critical_keywords(prompt):
    """
    Extracts key 'must-have' description for filtering (e.g. clothing).
    Looks for 'wearing ...', 'in a ...', 'with a ...' patterns.
    """
    keywords = []
    match = re.search(r'wearing\s+([^,]+)', prompt, re.IGNORECASE)
    if match:
        keywords.append(match.group(1).strip())

    match2 = re.search(r'in a\s+([^,]+)', prompt, re.IGNORECASE)
    if match2:
        keywords.append(match2.group(1).strip())

    match3 = re.search(r'with a\s+([^,]+)', prompt, re.IGNORECASE)
    if match3:
        keywords.append(match3.group(1).strip())

    return ", ".join(keywords) if keywords else prompt

def run_session(prompt, session_count):
    critical_prompt = extract_critical_keywords(prompt)
    print(f"\n🔄 Starting session {session_count} for prompt: \"{prompt}\"\n")
    print(f"🛡 Critical keyword filter: \"{critical_prompt}\" (CLIP-based)\n")

    image_paths, temp_folder = fetch_images(prompt, num_images=10, session_index=session_count - 1)
    results = []

    for path in image_paths:
        # Step 1: Critical keyword filter
        crit_score = get_clip_score(path, critical_prompt)
        if crit_score < 0.3:
            print(f"🚫 Rejected {path} — Critical score too low ({crit_score:.2f})")
            continue

        # Step 2: Main scoring
        clip_score = get_clip_score(path, prompt)
        sharp, bright, contrast = score_image_quality(path)
        final_score = (
            (clip_score * 3.0) +
            (sharp / 100.0) +
            (bright / 255.0) +
            (contrast / 100.0)
        )
        results.append({"filename": path, "score": final_score})

    # Step 3: Save top 5
    top5 = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    out_folder = os.path.join("fetched", f"session_{session_count}")
    os.makedirs(out_folder, exist_ok=True)

    for idx, r in enumerate(top5):
        dest = os.path.join(out_folder, f"top_{idx+1}.jpg")
        Image.open(r["filename"]).save(dest)
        print(f"📥 Saved top image to: {dest}")

    shutil.rmtree(temp_folder)
    print(f"\n📁 Review your top 5 photos in: {out_folder}\n")

if __name__ == "__main__":
    print("📸 Photo Pose Recommendation App")

    outfit = input("👕 What are you wearing today? ")
    location = input("📍 Where are you going? ")
    vibe = input("✨ What vibe do you want in the photo? ")
    gender = input("👤 Your gender? ")

    search_prompt = f"{gender} wearing {outfit} posing at {location} with {vibe} vibe photoshoot with high resolution"

    print(f"\n🧠 Final Search Prompt: {search_prompt}\n")

    os.makedirs("fetched", exist_ok=True)

    #  AUTO-DETECT session number
    existing_sessions = [d for d in os.listdir("fetched") if d.startswith("session_")]
    session_nums = [int(d.split("_")[1]) for d in existing_sessions if d.split("_")[1].isdigit()]
    session = max(session_nums, default=0) + 1

    while True:
        run_session(search_prompt, session)
        more = input("🔁 Want more pose ideas? (yes/no): ").strip().lower()
        if more != "yes":
            print("👋 All done! Stay slayin’ 💅")
            break
        session += 1
