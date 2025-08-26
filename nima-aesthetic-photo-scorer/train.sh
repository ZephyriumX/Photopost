#!/bin/bash

python -m nima-aesthetic-photo-scorer.train \
  --ava_csv nima-aesthetic-photo-scorer/data/ground_truth_dataset.csv \
  --images_dir nima-aesthetic-photo-scorer/data/images \
  --epochs 2 \
  --batch_size 16 \
  --backbone MobileNetV2 \
  --limit 500 \
  --steps_per_epoch 100
