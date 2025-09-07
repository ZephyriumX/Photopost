# photopost/nima/router.py
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# import tflite wrapper (from the scaffold)
from .tflite_infer import NIMATFLite

router = APIRouter(prefix="/nima", tags=["nima"])

# Default TFLite model path (relative to this file).
TFLITE_PATH = Path(__file__).resolve().parent / "models" / "nima.tflite"
_interpreter = None

def _get_interpreter():
    """Singleton: create and cache TFLite interpreter."""
    global _interpreter
    if _interpreter is None:
        if not TFLITE_PATH.exists():
            # Helpful error: tells you to place the tflite model
            raise RuntimeError(f"TFLite model not found at: {TFLITE_PATH}\n"
                               "Place your photopost/nima/models/nima.tflite file there.")
        _interpreter = NIMATFLite(str(TFLITE_PATH))
    return _interpreter

def _load_image_bytes(b: bytes):
    """Load image bytes to a tf tensor (H,W,3) uint8/float; no normalization here."""
    img = Image.open(io.BytesIO(b)).convert("RGB")
    arr = np.array(img)
    return tf.convert_to_tensor(arr)

@router.post("/rate")
async def rate_images(files: List[UploadFile] = File(...)):
    """
    Accepts multiple files and returns a ranked list by MOS (descending).
    Response: {"ranked": [{"filename":..., "mean":..., "std":...}, ...]}
    """
    try:
        interp = _get_interpreter()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for f in files:
        content = await f.read()
        try:
            img = _load_image_bytes(content)
        except Exception as e:
            # invalid image file
            continue

        # The TFLite wrapper in the scaffold may expose `score(img)` or `predict_mos(img)`.
        # Try common names for compatibility.
        try:
            mean, std, dist = interp.score(img)           # preferred
        except AttributeError:
            # fallback to predict_mos (maybe returns (mean, dist) or just dist)
            out = interp.predict_mos(img)
            if isinstance(out, tuple) and len(out) == 2:
                mean, dist = out
                dist = np.array(dist)
                idx = np.arange(1, 11, dtype=np.float32)
                std = float(np.sqrt(np.sum(dist * (idx - mean) ** 2)))
            else:
                # if it returned distribution only
                dist = np.array(out)
                idx = np.arange(1, 11, dtype=np.float32)
                mean = float(np.sum(dist * idx))
                std = float(np.sqrt(np.sum(dist * (idx - mean) ** 2)))

        results.append({"filename": f.filename, "mean": float(mean), "std": float(std)})

    # sort descending by mean
    results.sort(key=lambda x: x["mean"], reverse=True)
    return {"ranked": results}


# make it runnable by uvicorn directly: uvicorn photopost.nima.router:app --reload --port 8001
app = FastAPI(title="Photopost NIMA")
app.include_router(router)

