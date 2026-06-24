#!/usr/bin/env bash
set -euo pipefail

: "${BDD100K_IMAGES:?Set BDD100K_IMAGES to the BDD100K validation image directory}"
: "${BDD100K_ANNOTATIONS:?Set BDD100K_ANNOTATIONS to the BDD100K detection JSON}"
BDD100K_REALBLUR_LIST="${BDD100K_REALBLUR_LIST:-data_splits/bdd100k_realblur_final_500.txt}"

python tools/prepare_bdd100k_realblur_vehicle.py \
  --images "$BDD100K_IMAGES" \
  --annotations "$BDD100K_ANNOTATIONS" \
  --image-list "$BDD100K_REALBLUR_LIST" \
  --out datasets/BDD100K_realblur

python val_befr.py \
  --weights runs/detect/befr/weights/best.pt \
  --data data/bdd100k_realblur_vehicle.yaml \
  --name bdd100k_realblur_befr \
  --output results/metrics/bdd100k_realblur_befr.csv
