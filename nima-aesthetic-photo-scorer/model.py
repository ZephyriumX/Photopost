# photopost/nima/model.py
import tensorflow as tf
from tensorflow import keras
from .loss import emd_loss

def build_nima(input_shape=(224, 224, 3), backbone_name="MobileNetV2", dropout=0.25):
    if backbone_name == "MobileNetV2":
        base = keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        base = keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )

    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(dropout)(x)
    logits = keras.layers.Dense(10)(x)
    probs = keras.layers.Softmax(name="score_dist")(logits)
    model = keras.Model(base.input, probs)

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=emd_loss)
    return model, base   


    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(dropout)(x)
    logits = keras.layers.Dense(10)(x)
    probs = keras.layers.Softmax(name="score_dist")(logits)
    model = keras.Model(base.input, probs)


    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=emd_loss)
    return model

if __name__ == "__main__":
    m = build_nima()
    m.summary()