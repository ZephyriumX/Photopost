# photopost/nima/data.py
import os
import tensorflow as tf
import pandas as pd
import numpy as np


def load_ava_csv(csv_path):
    """
    Reads AVA CSV with columns including image_num (or image_name) and votes for 1..10.
    Expected format:
        image_num,vote_1,vote_2,...,vote_10
    or:
        image_name,count1,...,count10
    """
    df = pd.read_csv(csv_path)
    return df


def counts_to_dist(counts_row):
    """
    Convert raw vote counts to a probability distribution (10 values).
    """
    counts = np.array(counts_row, dtype=np.float32)
    s = counts.sum()
    if s == 0:
        return np.ones_like(counts) / len(counts)
    return counts / s


def make_dataset(ava_csv, images_dir, batch_size=32, shuffle=True, img_size=(224, 224)):
    df = load_ava_csv(ava_csv)

    cols = df.columns.tolist()
    if len(cols) >= 11:
        img_col = cols[0]       # "image_num" or "image_name"
        count_cols = cols[1:11] # votes 1..10
    else:
        raise ValueError("AVA CSV format not as expected. Need image_name/image_num + 10 count columns.")

    # Robust filename handling
    def id_to_filename(x):
        x = str(x)
        if not x.lower().endswith(".jpg"):
            x += ".jpg"
        return os.path.join(images_dir, x)

    image_paths = [id_to_filename(name) for name in df[img_col].values]

    # Convert each row of 10 votes → normalized distribution
    targets = np.stack([counts_to_dist(df[count_cols].iloc[i].values) for i in range(len(df))])

    def _load_image(path, target):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, target

    ds = tf.data.Dataset.from_tensor_slices((image_paths, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=4096)
    ds = ds.map(lambda p, t: tf.py_function(_load_image, [p, t], [tf.float32, tf.float32]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.ensure_shape(x, img_size + (3,)), tf.ensure_shape(y, (10,))))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
