# Photopost 📸✨

An AI-powered tool to simplify creating social media posts.  
Users can click → pick → post with automated assistance:

- **CLIP** → Suggests poses & backgrounds based on prompt.
- **NIMA** → Ranks multiple clicked photos and selects the most aesthetic.
- **BLIP** → Generates captions for selected photos.
- **Spotify API** → Suggests background music.

## 🔹 NIMA Module (your part)

- **Train**: Train NIMA on AVA dataset (`photopost/nima/train.py`).
- **Inference**: Rank uploaded images (`photopost/nima/infer.py` or `tflite_infer.py`).
- **FastAPI Router**: Exposes `/nima/rate` endpoint (`photopost/nima/router.py`).

### Setup

```bash
git clone https://github.com/ZephyriumX/Photopost.git
cd Photopost
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
