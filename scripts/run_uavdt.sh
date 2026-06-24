#!/usr/bin/env bash
set -euo pipefail

python train_baseline_yolov8.py --data data/uavdt_vehicle.yaml --name uavdt_baseline
python train_ablation.py --config configs/ablation_be_only.yaml --data data/uavdt_vehicle.yaml --name uavdt_be_only
python train_ablation.py --config configs/ablation_fr_only.yaml --data data/uavdt_vehicle.yaml --name uavdt_fr_only
python train_befr.py --model configs/yolov8n_befr.yaml --data data/uavdt_vehicle.yaml --name uavdt_befr

python tools/make_motion_blur_val.py \
  --images datasets/UAVDT/images/val \
  --labels datasets/UAVDT/labels/val \
  --out datasets/UAVDT_blur \
  --level all \
  --dataset-name uavdt

python eval_all_blur_levels.py \
  --weights runs/detect/uavdt_befr/weights/best.pt \
  --data-root datasets/UAVDT_blur \
  --dataset-name uavdt
