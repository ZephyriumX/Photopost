# photopost/nima/tflite_convert.py
import tensorflow as tf
import argparse
import numpy as np

def representative_data_gen(images_list, img_size=(224,224)):
    for img_path in images_list:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = tf.expand_dims(img, 0)
        yield [img]




def convert(keras_path, tflite_out, quantize=False, rep_images=None):
    model = tf.keras.models.load_model(keras_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if rep_images is not None:
        converter.representative_dataset = lambda: representative_data_gen(rep_images)
    tflite_model = converter.convert()
    with open(tflite_out, "wb") as f:
        f.write(tflite_model)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--keras", required=True)
    p.add_argument("--out", default="nima.tflite")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--rep_images", nargs="*", default=None)
    args = p.parse_args()
    convert(args.keras, args.out, quantize=args.quantize, rep_images=args.rep_images)