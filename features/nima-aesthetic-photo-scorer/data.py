import os
import tensorflow as tf
import pandas as pd
import numpy as np

DEFAULT_IMG_SIZE = (224, 224)

def load_ava_csv(csv_path):
    """
    Expects a table where the first column is image id or name,
    and the next 10 columns are vote counts for 1..10.
    """
    return pd.read_csv(csv_path)

def counts_to_dist(counts_row):
    counts = np.array(counts_row, dtype=np.float32)
    s = counts.sum()
    if s == 0:
        return np.ones_like(counts) / len(counts)
    return counts / s

def _id_to_filename(images_dir, x):
    x = str(x)
    if not x.lower().endswith(".jpg"):
        x += ".jpg"
    return os.path.join(images_dir, x)

def _build_tf_dataset(paths, targets, batch_size, img_size, shuffle):
    """Pure-TF input pipeline; corrupt JPEGs are dropped via ignore_errors()."""
    paths = tf.constant(paths)
    targets = tf.constant(targets, dtype=tf.float32)

    def _load_image(path, target):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_ACCURATE")
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, target

    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, len(paths)))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def make_datasets(
    ava_csv,
    images_dir,
    batch_size=32,
    img_size=DEFAULT_IMG_SIZE,
    val_split=0.2,
    shuffle=True,
    limit=None,
):
    """
    Returns (train_ds, val_ds, n_train, n_val).
    - Filters missing files up-front.
    - Converts counts -> normalized distributions.
    - Optional `limit` to take first N for quick debug.
    """
    df = load_ava_csv(ava_csv)
    cols = df.columns.tolist()
    if len(cols) < 11:
        raise ValueError("AVA CSV must have image_name/image_num + 10 count columns.")
    img_col = cols[0]
    count_cols = cols[1:11]

    # Build aligned arrays
    paths = [_id_to_filename(images_dir, name) for name in df[img_col].values]
    targets = np.stack([counts_to_dist(df[count_cols].iloc[i].values) for i in range(len(df))]).astype("float32")

    # Drop paths that don't exist
    valid_idx = [i for i, p in enumerate(paths) if os.path.exists(p)]
    if len(valid_idx) != len(paths):
        print(f"⚠️ Skipping {len(paths) - len(valid_idx)} missing files.")
    paths = [paths[i] for i in valid_idx]
    targets = targets[valid_idx]

    # Optional subset for speed
    if limit is not None:
        n = min(limit, len(paths))
        paths = paths[:n]
        targets = targets[:n]
        print(f"➡️ Using subset: {n} samples (limit={limit})")

    # Split BEFORE tf.data so lengths are exact
    n_total = len(paths)
    n_val = int(round(val_split * n_total)) if val_split and n_total > 0 else 0
    n_train = n_total - n_val

    # Keep validation deterministic: take last n_val items
    train_paths, train_targets = paths[:n_train], targets[:n_train]
    val_paths, val_targets = paths[n_train:], targets[n_train:]

    train_ds = _build_tf_dataset(train_paths, train_targets, batch_size, img_size, shuffle=shuffle)
    val_ds = _build_tf_dataset(val_paths, val_targets, batch_size, img_size, shuffle=False) if n_val > 0 else None

    return train_ds, val_ds, n_train, n_val
