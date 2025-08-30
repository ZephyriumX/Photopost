#!/bin/bash
python -m nima-aesthetic-photo-scorer.train \
  --ava_csv "nima-aesthetic-photo-scorer/data/ground_truth_dataset.csv" \
  --images_dir "nima-aesthetic-photo-scorer/data/images" \
  --out_dir "models" \
  --epochs 15 \
  --batch_size 64 \
  --backbone MobileNetV2 \
  --val_split 0.2
