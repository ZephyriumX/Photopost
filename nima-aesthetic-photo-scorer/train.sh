#!/bin/bash
python -m nima-aesthetic-photo-scorer.train \
  --ava_csv nima-aesthetic-photo-scorer/data/ground_truth_dataset.csv \
  --images_dir nima-aesthetic-photo-scorer/data/images \
  --out_dir nima-aesthetic-photo-scorer/models \
  --epochs 10 \
  --batch_size 32