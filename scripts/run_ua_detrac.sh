#!/usr/bin/env bash
set -euo pipefail

python train_baseline_yolov8.py --data data/ua_detrac_vehicle.yaml --name ua_detrac_baseline
python train_ablation.py --config configs/ablation_be_only.yaml --data data/ua_detrac_vehicle.yaml --name ua_detrac_be_only
python train_ablation.py --config configs/ablation_fr_only.yaml --data data/ua_detrac_vehicle.yaml --name ua_detrac_fr_only
python train_befr.py --model configs/yolov8n_befr.yaml --data data/ua_detrac_vehicle.yaml --name ua_detrac_befr

python tools/make_motion_blur_val.py \
  --images datasets/UA-DETRAC/images/val \
  --labels datasets/UA-DETRAC/labels/val \
  --out datasets/UA-DETRAC_blur \
  --level all \
  --dataset-name ua_detrac

python eval_all_blur_levels.py \
  --weights runs/detect/ua_detrac_befr/weights/best.pt \
  --data-root datasets/UA-DETRAC_blur \
  --dataset-name ua_detrac
