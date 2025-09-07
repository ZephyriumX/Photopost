import argparse
import os
import tensorflow as tf
from tensorflow import keras

from .data import make_datasets
from .model import build_nima
from .loss import emd_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ava_csv", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", default="./models")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--backbone", default="MobileNetV2")
    p.add_argument("--val_split", type=float, default=0.2, help="Validation fraction (0..1)")
    p.add_argument("--limit", type=int, default=None, help="Use only first N samples for quick runs.")
    p.add_argument("--steps_per_epoch", type=int, default=None, help="Cap steps/epoch for faster cycles.")
    p.add_argument("--val_steps", type=int, default=None, help="Cap validation steps.")
    p.add_argument("--loss", choices=["emd", "kl"], default="emd", help="Training loss.")
    p.add_argument("--freeze_backbone_epochs", type=int, default=3, help="Epochs to train head with frozen backbone.")
    p.add_argument("--finetune_layers", type=int, default=30, help="Unfreeze last N layers for fine-tuning.")
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision (fp16 on GPU).")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling/splits.")
    return p.parse_args()


def _maybe_enable_mixed_precision(enable: bool):
    if not enable:
        return
    try:
        from tensorflow.keras import mixed_precision as mp  # tf<2.15
    except Exception:
        from keras import mixed_precision as mp  # keras3
    policy = mp.Policy("mixed_float16")
    mp.set_global_policy(policy)
    print("⚙️  Mixed precision enabled:", policy)


def _export_savedmodel(model: keras.Model, export_dir: str):
    """
    Export a TF SavedModel in a version-compatible way.
    Keras 3: model.export(dir)
    Keras 2.x fallback: tf.saved_model.save(model, dir)
    """
    os.makedirs(export_dir, exist_ok=True)
    if hasattr(model, "export"):
        model.export(export_dir)
    else:
        tf.saved_model.save(model, export_dir)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Seed for reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    _maybe_enable_mixed_precision(args.mixed_precision)

    print(f"Loading datasets from {args.ava_csv} and {args.images_dir}")
    train_ds, val_ds, n_train, n_val = make_datasets(
    ava_csv=args.ava_csv,
    images_dir=args.images_dir,
    batch_size=args.batch_size,
    val_split=args.val_split,
    shuffle=True,
    limit=args.limit,
)

    print(f"🟩 train samples: {n_train} | 🟦 val samples: {n_val}")

    print(f"Building NIMA with backbone {args.backbone}")
    model, backbone = build_nima(backbone_name=args.backbone)

    # Freeze whole backbone for head warmup
    for layer in backbone.layers:
        layer.trainable = False

    # Choose loss
    loss_fn = emd_loss if args.loss == "emd" else keras.losses.KLDivergence()

    # Common callbacks (apply to both phases)
    val_monitor = "val_loss" if val_ds is not None else "loss"

    csv_log = keras.callbacks.CSVLogger(os.path.join(args.out_dir, "train_log.csv"), append=True)
    tonan = keras.callbacks.TerminateOnNaN()
    lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1, monitor=val_monitor)

    # Auto-resume training if interrupted mid-epoch
    backup = keras.callbacks.BackupAndRestore(
        backup_dir=os.path.join(args.out_dir, "backup_state"), save_freq="epoch"
    )

    # Phase 1: head-only training
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn)

    ckpt_head_path = os.path.join(args.out_dir, "nima_best_head.weights.h5")
    ckpt_head = keras.callbacks.ModelCheckpoint(
        ckpt_head_path,
        save_best_only=True,
        save_weights_only=True,
        monitor=val_monitor,
        mode="min",
        verbose=1,
    )
    es_head = keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor=val_monitor, mode="min"
    )

    print("🔒 Training head only (frozen backbone)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=min(args.freeze_backbone_epochs, args.epochs),
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        callbacks=[ckpt_head, es_head, lr, csv_log, tonan, backup],
        verbose=1,
    )

    # Phase 2: unfreeze last N layers for fine-tuning
    remaining_epochs = max(0, args.epochs - min(args.freeze_backbone_epochs, args.epochs))
    if remaining_epochs > 0:
        print(f"🔓 Unfreezing last {args.finetune_layers} backbone layers for fine-tuning...")
        for layer in backbone.layers[-args.finetune_layers:]:
            layer.trainable = True

        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=loss_fn)

        ckpt_ft_path = os.path.join(args.out_dir, "nima_best_finetune.weights.h5")
        ckpt_ft = keras.callbacks.ModelCheckpoint(
            ckpt_ft_path,
            save_best_only=True,
            save_weights_only=True,
            monitor=val_monitor,
            mode="min",
            verbose=1,
        )
        es_ft = keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor=val_monitor, mode="min"
        )

        print("🎯 Fine-tuning...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining_epochs,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.val_steps,
            callbacks=[ckpt_ft, es_ft, lr, csv_log, tonan, backup],
            verbose=1,
        )

        # Prefer the best finetune weights if they exist; else fall back to best head
        if os.path.exists(ckpt_ft_path):
            print("⬇️  Loading best fine-tune weights before final save...")
            model.load_weights(ckpt_ft_path)
        elif os.path.exists(ckpt_head_path):
            print("⬇️  Loading best head weights before final save (no finetune ckpt found)...")
            model.load_weights(ckpt_head_path)
    else:
        # No finetune phase; ensure we save best head weights
        if os.path.exists(ckpt_head_path):
            print("⬇️  Loading best head weights before final save...")
            model.load_weights(ckpt_head_path)

    # Save final model in multiple formats
    final_keras = os.path.join(args.out_dir, "nima_final.keras")   # native Keras format (recommended)
    final_h5 = os.path.join(args.out_dir, "nima_final.h5")         # legacy H5
    export_dir = os.path.join(args.out_dir, "nima_savedmodel")     # TF SavedModel directory

    print("💾 Saving models...")
    model.save(final_keras)
    model.save(final_h5)
    try:
        _export_savedmodel(model, export_dir)
    except Exception as e:
        print(f"⚠️ SavedModel export failed with: {e}")

    print("✅ Training complete")
    print(f"  • {final_keras}")
    print(f"  • {final_h5}")
    print(f"  • {export_dir} (if export succeeded)")


if __name__ == "__main__":
    main()
