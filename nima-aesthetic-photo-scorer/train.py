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
    p.add_argument("--limit", type=int, default=None,
                   help="Use only first N samples for faster debugging.")
    p.add_argument("--steps_per_epoch", type=int, default=None,
                   help="Optional: cap steps per epoch for faster iterations.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset from {args.ava_csv} and {args.images_dir}")
    ds = make_dataset(
        args.ava_csv,
        args.images_dir,
        batch_size=args.batch_size,
        shuffle=True,
        limit=args.limit,               # << new
    )

    print(f"Building NIMA with backbone {args.backbone}")
    model, backbone = build_nima(backbone_name=args.backbone)  # return both

    # ---- Phase 1: Freeze backbone (fast training) ----
    for layer in backbone.layers:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.KLDivergence(),   # << avoid string alias issues
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        os.path.join(args.out_dir, "nima_best_head.h5"),
        save_best_only=True, monitor="loss"
    )
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

    print("Training head only (frozen backbone)...")
    model.fit(
        ds,
        epochs=min(3, args.epochs),
        steps_per_epoch=args.steps_per_epoch,   # << optional speed-up
        callbacks=[ckpt, es, lr]
    )

    # ---- Phase 2: Fine-tune some backbone layers (last ~30) ----
    print("Unfreezing backbone for fine-tuning...")
    for layer in backbone.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.KLDivergence(),
    )

    ckpt_ft = keras.callbacks.ModelCheckpoint(
        os.path.join(args.out_dir, "nima_best_finetune.h5"),
        save_best_only=True, monitor="loss"
    )

    model.fit(
        ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=[ckpt_ft, es, lr]
    )

    model.save(os.path.join(args.out_dir, "nima_final.h5"))
    print("✅ Training complete, model saved.")


if __name__ == "__main__":
    main()
