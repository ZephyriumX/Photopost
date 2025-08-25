# photopost/nima/infer.py
import numpy as np
import tensorflow as tf
from .utils import dist_to_mos

class NIMAKeras:
    """Simple wrapper to run Keras NIMA models (.h5) and return (mean, std, dist)."""
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def score_image(self, img_np):
        """
        img_np: Tensor or numpy array with dtype uint8/float. Shape: [H,W,3]
        Returns: mean (float), std (float), dist (list of 10 floats)
        """
        img = tf.image.resize(img_np, (224, 224))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        x = tf.expand_dims(img, 0)  # batch dim
        p = self.model(x).numpy()[0]  # (10,)
        mean, std = dist_to_mos(p)
        return float(mean), float(std), p.tolist()
