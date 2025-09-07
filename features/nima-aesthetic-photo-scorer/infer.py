# infer.py
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .model import build_nima
from .loss import emd_loss

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    """Load and preprocess image for NIMA."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return tf.expand_dims(img, axis=0)  # shape (1,224,224,3)

def predict_score(model, img_path):
    """Predict mean aesthetic score + distribution."""
    x = preprocess_image(img_path)
    preds = model.predict(x)  # (1,10)
    preds = preds[0]
    mean_score = np.sum(np.arange(1, 11) * preds)
    return mean_score, preds

def main(args):
    # Load model (supports .h5 or checkpoint dir)
    if args.model.endswith(".h5"):
        model = load_model(args.model, custom_objects={"emd_loss": emd_loss})
    else:
        # rebuild same model arch and load checkpoint
        model = build_nima()
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(tf.train.latest_checkpoint(args.model)).expect_partial()
    
    print(f"[INFO] Model loaded from {args.model}")
    
    # Loop over images
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"[WARN] Skipping {img_path}, not found.")
            continue
        score, dist = predict_score(model, img_path)
        print(f"\nImage: {img_path}")
        print(f"  Mean aesthetic score: {score:.3f}")
        print(f"  Distribution (1–10): {np.round(dist, 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NIMA aesthetic inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .h5 model OR checkpoint dir")
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="List of image paths to score")
    args = parser.parse_args()
    main(args)
