# photopost/nima/eval.py
import numpy as np
import argparse
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
from .data import load_ava_csv
from .model import build_nima
from .utils import dist_to_mos

def emd_between(p, q):
    return np.sum((np.cumsum(p) - np.cumsum(q)) ** 2)

def evaluate(keras_model_path, csv_path, images_dir, max_samples=None):
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    df = load_ava_csv(csv_path)
    cols = df.columns.tolist()
    img_col = cols[0]
    count_cols = cols[1:11]

    mos_preds = []
    mos_gts = []
    emd_list = []

    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break
        img_path = f"{images_dir}/{row[img_col]}"
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img, tf.float32))
        p = model(tf.expand_dims(x, 0)).numpy()[0]
        gt_counts = row[count_cols].values.astype(np.float32)
        gt_dist = gt_counts / (gt_counts.sum() + 1e-8)

        pred_mos, _ = dist_to_mos(p)
        gt_mos = float((np.arange(1, 11) * gt_dist).sum())
        mos_preds.append(pred_mos)
        mos_gts.append(gt_mos)
        emd_list.append(emd_between(p, gt_dist))

    srcc = spearmanr(mos_gts, mos_preds).correlation
    plcc = pearsonr(mos_gts, mos_preds)[0]
    mean_emd = float(np.mean(emd_list))
    return dict(srcc=srcc, plcc=plcc, mean_emd=mean_emd)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--max", type=int, default=500)
    args = p.parse_args()
    print(evaluate(args.model, args.csv, args.images_dir, max_samples=args.max))