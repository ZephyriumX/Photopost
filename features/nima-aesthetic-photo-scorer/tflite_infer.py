# photopost/nima/tflite_infer.py
import numpy as np
import tensorflow as tf

class NIMATFLite:
    def __init__(self, tflite_path):
        # Load TFLite model and allocate tensors
        self.interp = tf.lite.Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()

    def _preprocess(self, img_np):
        # Resize to MobileNetV2 input size
        img = tf.image.resize(img_np, (224, 224))
        img = tf.cast(img, tf.float32)
        # Apply same preprocessing as used during training
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        arr = np.expand_dims(img.numpy(), 0).astype(np.float32)  # add batch dim
        return arr

    def predict_dist(self, img_np):
        # Get input tensor
        x = self._preprocess(img_np)
        # Feed to interpreter
        self.interp.set_tensor(self.input_details[0]["index"], x)
        self.interp.invoke()
        # Get output tensor (distribution over 1–10 aesthetic scores)
        dist = self.interp.get_tensor(self.output_details[0]["index"])[0]
        return dist

    def predict_mos(self, img_np):
        # Predict distribution
        dist = self.predict_dist(img_np)
        # Compute Mean Opinion Score (MOS)
        scores = np.arange(1, 11)  # [1,2,...,10]
        mos = float(np.sum(dist * scores))
        return mos, dist
