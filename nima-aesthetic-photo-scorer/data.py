import os
import tensorflow as tf
import pandas as pd
import numpy as np


def load_ava_csv(csv_path):
    return pd.read_csv(csv_path)


def counts_to_dist(counts_row):
    counts = np.array(counts_row, dtype=np.float32)
    s = counts.sum()
    if s == 0:
        return np.ones_like(counts) / len(counts)
    return counts / s


def make_dataset(ava_csv, images_dir, batch_size=32,
                 shuffle=True, img_size=(224, 224), limit=None):
    """
    Build tf.data.Dataset from AVA CSV.
    - Skips non-existing files up front (Python-side).
    - Uses pure TF ops (no py_function).
    - Applies ignore_errors() so corrupt JPEGs are dropped.
    - Optional `limit` to use first N samples for faster debugging.
    """
    df = load_ava_csv(ava_csv)

    cols = df.columns.tolist()
    if len(cols) < 11:
        raise ValueError("AVA CSV must have image_name/image_num + 10 count columns.")
    img_col = cols[0]
    count_cols = cols[1:11]

    # Build file list
    def id_to_filename(x):
        x = str(x)
        if not x.lower().endswith(".jpg"):
            x += ".jpg"
        return os.path.join(images_dir, x)

    paths = [id_to_filename(name) for name in df[img_col].values]
    targets = np.stack([counts_to_dist(df[count_cols].iloc[i].values)
                        for i in range(len(df))]).astype("float32")

    # Filter out missing files BEFORE tf.data
    valid = [i for i, p in enumerate(paths) if os.path.exists(p)]
    if len(valid) != len(paths):
        missing = len(paths) - len(valid)
        print(f"⚠️ Skipping {missing} missing files (do not exist on disk).")
    paths = [paths[i] for i in valid]
    targets = targets[valid]

    # Optional subset for fast runs
    if limit is not None:
        n = min(limit, len(paths))
        paths = paths[:n]
        targets = targets[:n]
        print(f"➡️ Using subset: {n} samples (limit={limit})")

    # Pure TF loader
    def _load_image(path, target):
        img = tf.io.read_file(path)
        # decode_jpeg will raise if corrupt; ignore_errors() will drop sample
        img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_ACCURATE")
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, target

    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, len(paths)))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # If a decode_jpeg fails (corrupt image), silently skip that example
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
