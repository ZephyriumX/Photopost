# photopost/nima/train.py
import argparse
import os
from tensorflow import keras
from .data import make_dataset
from .model import build_nima


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ava_csv", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", default="./models")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone", default="MobileNetV2")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset from {args.ava_csv} and {args.images_dir}")
    ds = make_dataset(args.ava_csv, args.images_dir,
                      batch_size=args.batch_size, shuffle=True)

    print(f"Building NIMA with backbone {args.backbone}")
    model = build_nima(backbone_name=args.backbone)

    # ---- Phase 1: Freeze backbone (fast training) ----
    for layer in model.layers[0].layers:  # model.layers[0] = backbone
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="kullback_leibler_divergence")

    ckpt = keras.callbacks.ModelCheckpoint(
        os.path.join(args.out_dir, "nima_best_head.h5"),
        save_best_only=True, monitor="loss"
    )
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

    print("Training head only (frozen backbone)...")
    model.fit(ds, epochs=min(3, args.epochs),
              callbacks=[ckpt, es, lr])

    # ---- Phase 2: Fine-tune some backbone layers (optional if GPU) ----
    print("Unfreezing backbone for fine-tuning...")
    for layer in model.layers[0].layers[-30:]:  # last ~30 layers
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="kullback_leibler_divergence")

    ckpt_ft = keras.callbacks.ModelCheckpoint(
        os.path.join(args.out_dir, "nima_best_finetune.h5"),
        save_best_only=True, monitor="loss"
    )

    model.fit(ds, epochs=args.epochs, callbacks=[ckpt_ft, es, lr])
    model.save(os.path.join(args.out_dir, "nima_final.h5"))
    print("✅ Training complete, model saved.")


if __name__ == "__main__":
    main()
