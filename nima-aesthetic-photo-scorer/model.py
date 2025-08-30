import tensorflow as tf
from tensorflow import keras
from .loss import emd_loss

def build_nima(input_shape=(224, 224, 3), backbone_name="MobileNetV2", dropout=0.25):
    """
    Builds a NIMA model: backbone (imagenet) + GAP + Dropout + Dense(10) + Softmax.
    Returns:
        model: keras.Model taking (input_shape) -> score distribution (10,)
        backbone: the feature extractor (so you can freeze/unfreeze cleanly)
    """
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
    model = keras.Model(inputs=base.input, outputs=probs)

    # default compile with EMD (paper-style). train.py can re-compile to KL if requested.
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=emd_loss)
    return model, base

if __name__ == "__main__":
    m, _ = build_nima()
    m.summary()
