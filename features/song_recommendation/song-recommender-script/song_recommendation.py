
"""
vibe2songs.py
--------------
Extract vibe keywords from a prompt and ask an LLM (OpenRouter)
to recommend Bollywood + Hollywood songs, returning structured JSON.

Usage:
    pip install -r requirements.txt
    python vibe2songs.py --prompt "man posing in black tuxedo bossy mafia vibe photoshoot"

Environment:
    Create a `.env` file with:
        OPENROUTER_API_KEY=sk-...
"""

import os
import re
import json
import argparse
import textwrap
import requests
from dotenv import load_dotenv

# -------------------------
# Load environment
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# External libs (optional)
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    HAS_KEYBERT = True
except Exception:
    HAS_KEYBERT = False

# -------------------------
# Provider + Models
# -------------------------
DEFAULT_PROVIDER = "openrouter"
DEFAULT_EXTRACT_MODEL = "gpt-4o-mini"
DEFAULT_RECOMMEND_MODEL = "gpt-4o-mini"

# Keep track of duplicates across runs
seen_songs = set()


def call_llm(provider, model, messages, temperature=0.0, max_tokens=500):
    """Call OpenRouter LLM only."""
    if provider != "openrouter":
        raise ValueError("Only OpenRouter provider is supported now.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# -------------------------
# JSON helper
# -------------------------
def safe_json_load(s: str):
    """Try to extract JSON object/array from text and parse it."""
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for pat in [r"(\{.*\})", r"(\[.*\])"]:
        m = re.search(pat, s, flags=re.S)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                candidate2 = re.sub(r",\s*}", "}", candidate)
                candidate2 = re.sub(r",\s*]", "]", candidate2)
                try:
                    return json.loads(candidate2)
                except Exception:
                    continue
    raise ValueError("Could not parse JSON from response.")


# -------------------------
# Keyword extraction
# -------------------------
def extract_keywords_keybert(prompt: str, top_n=6):
    if not HAS_KEYBERT:
        return []
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw = KeyBERT(model=emb_model)
    results = kw.extract_keywords(
        prompt,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=True,
        diversity=0.6,
        top_n=top_n,
    )
    return [r[0].lower() for r in results]


def extract_keywords_llm(prompt: str, provider, model=DEFAULT_EXTRACT_MODEL, max_keywords=8):
    system = (
        "You are a concise extractor. Read a creative prompt and return ONLY a JSON array of "
        "3–8 lowercase vibe/keyword strings (short words or phrases). "
        "Examples: [\"bossy\", \"mafia\", \"cinematic\", \"romantic\"]."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Prompt: {prompt}"},
    ]
    text = call_llm(provider, model, messages, temperature=0.0, max_tokens=200)
    try:
        arr = safe_json_load(text)
        return [str(x).lower().strip() for x in arr if x]
    except Exception:
        tokens = re.sub(r"[^\w\s-]", " ", prompt).lower().split()
        return list(dict.fromkeys([t for t in tokens if len(t) > 2]))[:max_keywords]


# -------------------------
# Merge & refine
# -------------------------
def merge_keywords(*lists):
    merged = []
    for lst in lists:
        for k in lst or []:
            k = re.sub(r"\s+", " ", k.strip().lower())
            if k and k not in merged:
                merged.append(k)
    return merged


def refine_keywords(keywords, keep_bigrams=True):
    STOP = {
        "photo",
        "photoshoot",
        "pose",
        "man",
        "woman",
        "person",
        "image",
        "picture",
        "shot",
        "photography",
    }
    cleaned = [k for k in keywords if k not in STOP and len(k) > 1 and len(k.split()) <= 3]
    if not keep_bigrams:
        cleaned = [k.split()[0] for k in cleaned]
    return cleaned[:8]


# -------------------------
# Recommend songs
# -------------------------
def recommend_songs(keywords, provider, num_each=5, model=DEFAULT_RECOMMEND_MODEL, tone=None):
    system = textwrap.dedent(
        """
You are a music recommendation assistant.
Your job is to recommend Bollywood and Hollywood songs based ONLY on the provided vibe keywords.

⚠️ STRICT RULES:
- Bollywood picks must also strictly match the given vibe, not generic romantic fillers.
- Hollywood and Bollywood both must align with the vibe (energy, mood, theme).
- DO NOT add romantic/love/heartbreak songs unless the vibe explicitly mentions those.
- Always mix BOTH Bollywood and Hollywood songs unless user specifies otherwise.
- Avoid duplicates within a single batch AND across runs. (Treat title+artist as unique key.)
- If unsure, choose neutral/atmospheric tracks that still fit the vibe rather than random hits.
- Return 8–10 songs total, balanced between Bollywood and Hollywood.

OUTPUT FORMAT (strict JSON only, no extra text):
{
  "songs": [
    {"title": "Song Name", "artist": "Artist", "origin": "Bollywood/Hollywood"},
    ...
  ]
}
"""
    ).strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"keywords": keywords, "num_each": num_each, "tone": tone or ""})},
    ]
    text = call_llm(provider, model, messages, temperature=0.7, max_tokens=800)
    raw = safe_json_load(text)

    # Deduplicate against seen_songs
    global seen_songs
    filtered = {"songs": []}
    for s in raw.get("songs", []):
        key = f"{s.get('title','').lower()}::{s.get('artist','').lower()}"
        if key not in seen_songs:
            seen_songs.add(key)
            filtered["songs"].append(s)

    return filtered


# -------------------------
# Pipeline
# -------------------------
def build_pipeline(prompt, provider, use_keybert=True, num_each=5, model=DEFAULT_RECOMMEND_MODEL, tone=None):
    local_kw = extract_keywords_keybert(prompt) if use_keybert and HAS_KEYBERT else []
    llm_kw = extract_keywords_llm(prompt, provider=provider)
    merged = merge_keywords(local_kw, llm_kw)
    refined = refine_keywords(merged)
    if not refined:
        refined = refine_keywords(prompt.split())
    return recommend_songs(refined, provider, num_each=num_each, model=model, tone=tone)


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Vibe -> Songs recommender (OpenRouter only)")
    parser.add_argument("--prompt", required=True, help="Prompt describing photo/vibe")
    parser.add_argument("--no-keybert", action="store_true", help="Disable KeyBERT extraction")
    parser.add_argument("--num-each", type=int, default=5, help="Songs per category")
    parser.add_argument("--model", type=str, default=DEFAULT_RECOMMEND_MODEL, help="Model to use")
    parser.add_argument("--tone", type=str, help="Optional tone hint")
    args = parser.parse_args()

    # interactive loop
    while True:
        result = build_pipeline(
            args.prompt,
            provider="openrouter",
            use_keybert=not args.no_keybert,
            num_each=args.num_each,
            model=args.model,
            tone=args.tone,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

        again = input("\nDo you want more recommendations? (yes/no): ").strip().lower()
        if again != "yes":
            print("\n👉 Thankss! Come back againn! 👋")
            break


if __name__ == "__main__":
    main()
