# BE-FR YOLO

Reproducible implementation of **BE-FR YOLO: Progressive Blur Evolution and
Frequency Refinement for Robust Vehicle Detection**. The project is built on the
official Ultralytics YOLO package and YOLOv8n, without modifying an installed
Ultralytics package.

Reported manuscript values in `results/organized_data/` are preserved as
reported data. They are not regenerated unless the training and evaluation
commands are run on the corresponding datasets, splits, seeds, and hardware.

## Overview

BE-FR YOLO combines training-time blur evolution with inference-time frequency
refinement for robust vehicle detection under motion blur.

- **BE-Net** is a training-only blur evolution module. It uses a linear motion
  PSF, optional Gaussian noise, Bernoulli blur injection, a lightweight PGN, and
  curriculum schedules for blur length and injection probability.
- **FR-Net** remains active during training and inference. It refines P3, P4,
  and P5 features immediately before the official YOLOv8 Detect head.
- **Spectral consistency loss** compares normalized amplitude spectra from
  detached sharp-domain neck features and refined blurred-domain features.
- **Total loss:** `L_total = L_det + lambda_spec * L_spec`.

BE-Net is excluded from inference checkpoints. FR-Net is injected through a
Detect pre-forward hook so YOLOv8 Detect weight keys remain compatible with
official Ultralytics weights.

## Main Contributions

- Progressive blur evolution for training YOLOv8n vehicle detectors under
  increasing motion blur.
- Lightweight frequency refinement for multi-scale YOLO neck features.
- Public metadata for the BDD100K real motion-blur subset: 800 candidate
  manifest, 500 final manifest, index-to-filename mapping, and manual-review
  record.
- Data conversion and manifest scripts for UA-DETRAC, UAVDT, BDD100K, synthetic
  blur validation, complexity measurement, and statistical testing.

## Repository Structure

```text
befr/blur/                 # Motion PSF and blur-evolution utilities.
befr/modules/              # FR-Net implementation.
befr/losses/               # Spectral consistency loss.
befr/integration/          # Ultralytics model/trainer integration boundary.
befr/data/                 # Shared preprocessing utilities.
configs/                   # BE-FR defaults, ablations, and sensitivity sweeps.
data/                      # Ultralytics dataset YAML files.
data_splits/               # Public split and manifest metadata.
tools/                     # Dataset, visualization, complexity, and audit scripts.
scripts/                   # End-to-end shell recipes.
tests/                     # Unit and integration tests.
results/organized_data/    # Manuscript-derived tables and workbook.
```

Main scripts:

- `train_baseline_yolov8.py`: train an unmodified Ultralytics YOLOv8 baseline.
- `train_ablation.py`: train BE-only, FR-only, or BE-FR ablations.
- `train_befr.py`: train the full BE-FR YOLO variant.
- `val_befr.py`: validate one checkpoint and save a metrics CSV.
- `eval_all_blur_levels.py`: evaluate clear/light/medium/heavy generated sets.
- `run_sensitivity.py`: execute sensitivity sweeps from `configs/sensitivity_*.yaml`.
- `tools/prepare_ua_detrac_vehicle.py`: convert common UA-DETRAC XML releases
  to one-class YOLO vehicle data.
- `tools/prepare_uavdt_vehicle.py`: convert common UAVDT MOT-style annotations
  to one-class YOLO vehicle data.
- `tools/map_bdd100k_indices_to_filenames.py`: map manual BDD100K review
  indices to original candidate filenames.
- `tools/build_bdd100k_final_subset.py`: rebuild the final 500-image BDD100K
  real motion-blur YOLO subset from the 800 candidates.
- `tools/validate_bdd100k_curation.py`: validate the 800/500/300 BDD100K
  curation metadata.
- `tools/export_yolo_dataset_manifest.py`: export train/val split files from a
  YOLO dataset YAML.
- `tools/statistical_significance.py`: compute paired sequence-level tests.
- `tools/make_motion_blur_val.py`: generate synthetic motion-blur validation
  sets without overwriting source data.
- `tools/measure_complexity.py`: measure parameters, FLOPs, FPS, and latency.
- `tools/visualize_predictions.py`, `tools/gradcam_yolo.py`, and
  `tools/visualize_fft_response.py`: generate qualitative visualizations.

## Environment Requirements

Recommended environment:

| Component | Recommended version |
|---|---|
| Python | 3.10 |
| PyTorch | `>=2.1,<2.6` |
| Torchvision | `>=0.16,<0.21`, compatible with the installed PyTorch version |
| Ultralytics | `>=8.3,<8.5` |
| OpenCV | `>=4.8,<5` |
| NumPy | `>=1.24,<2.0` |
| SciPy | `>=1.10,<2` |
| PyYAML | `>=6.0,<7` |
| Pillow | `>=9.5,<13` |
| Matplotlib | `>=3.7,<4` |
| THOP | `>=0.1.1` |

Repository validation environment used for the metadata and preprocessing
checks in this release:

| Component | Observed value |
|---|---|
| Operating system | Windows 11 10.0.22631-SP0 |
| Python | 3.12.13 audit runtime |
| NumPy | 2.3.5 audit runtime |
| SciPy | 1.18.0 audit runtime |
| PyYAML | 6.0.3 audit runtime |
| Pillow | 12.2.0 audit runtime |
| pytest | 9.1.1 audit runtime |
| GPU noted in manuscript | NVIDIA GeForce RTX 3060 |
| CUDA/cuDNN | CUDA-compatible PyTorch environment |

Use `requirements.txt` for portable installs and `requirements-lock.txt` or
`environment.yml` for a bounded reproducibility environment.

## Installation

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
python tools/smoke_test.py
pytest
```

Conda:

```bash
conda env create -f environment.yml
conda activate befr-yolo
pip install -e .
```

## Third-Party Dataset Sources

This repository does not redistribute original third-party images or
annotations. Download datasets from official sources and comply with their
licenses and terms.

| Dataset | Source reference | Use in this repository |
|---|---|---|
| UA-DETRAC | Wen et al., "UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking," Computer Vision and Image Understanding, 2020, DOI `10.1016/j.cviu.2020.102907`; dataset page https://detrac-db.rit.albany.edu/ | Source vehicle dataset converted to YOLO format. |
| UAVDT | Du et al., "The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking," ECCV 2018; project page https://sites.google.com/view/grli-uavdt/ | Source UAV vehicle dataset converted to YOLO format. |
| BDD100K | Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning," CVPR 2020; project page https://bdd-data.berkeley.edu/ | Source validation images and annotations for the real motion-blur subset. |

## Dataset Directory Structure

The committed YAML files assume YOLO-format datasets:

```text
datasets/
  UA-DETRAC/
    images/train/
    images/val/
    labels/train/
    labels/val/
  UAVDT/
    images/train/
    images/val/
    labels/train/
    labels/val/
  BDD100K_realblur/
    images/val/
    labels/val/
```

YOLO labels use normalized rows:

```text
class_id center_x center_y width height
```

The supplied dataset YAMLs merge vehicle categories into class `0: vehicle`.

## Dataset Preparation

### UA-DETRAC

`tools/prepare_ua_detrac_vehicle.py` supports common UA-DETRAC layouts such as:

```text
Insight-MVT_Annotation_Train/*.xml
Insight-MVT_Annotation_Test/*.xml
Insight-MVT_Train/MVI_20011/img00001.jpg
Insight-MVT_Test/MVI_20011/img00001.jpg
```

It parses XML frame numbers, target IDs, boxes, vehicle attributes, occlusion,
truncation, and ignored regions. Valid vehicle targets are merged to class `0`.
Boxes are clipped to image boundaries and invalid boxes are removed.

```bash
python tools/prepare_ua_detrac_vehicle.py \
  --train-images /path/to/Insight-MVT_Train \
  --train-annotations /path/to/Insight-MVT_Annotation_Train \
  --val-images /path/to/Insight-MVT_Test \
  --val-annotations /path/to/Insight-MVT_Annotation_Test \
  --output-root datasets/UA-DETRAC \
  --splits-output-dir data_splits \
  --mode copy
```

Optional sequence lists:

```bash
python tools/prepare_ua_detrac_vehicle.py \
  --train-images /path/to/Insight-MVT_Train \
  --train-annotations /path/to/Insight-MVT_Annotation_Train \
  --val-images /path/to/Insight-MVT_Test \
  --val-annotations /path/to/Insight-MVT_Annotation_Test \
  --train-sequences /path/to/train_sequences.txt \
  --val-sequences /path/to/val_sequences.txt \
  --output-root datasets/UA-DETRAC \
  --splits-output-dir data_splits
```

The script writes:

```text
datasets/UA-DETRAC/images/train/
datasets/UA-DETRAC/images/val/
datasets/UA-DETRAC/labels/train/
datasets/UA-DETRAC/labels/val/
data_splits/ua_detrac_train.txt
data_splits/ua_detrac_val.txt
```

### UAVDT

`tools/prepare_uavdt_vehicle.py` supports common UAVDT layouts such as:

```text
UAV-benchmark-M/M0101/img000001.jpg
UAV-benchmark-MOTD_v1.0/M0101/img000001.jpg
GT/M0101_gt_whole.txt
GT/M0101_gt.txt
```

It parses common comma-, space-, or tab-separated fields:
`frame_id`, `target_id`, `bbox_left`, `bbox_top`, `bbox_width`,
`bbox_height`, `out_of_view`, `occlusion`, and `object_category`. Valid vehicle
targets are merged to class `0`. Boxes are clipped to image boundaries and
invalid boxes are removed.

```bash
python tools/prepare_uavdt_vehicle.py \
  --images-root /path/to/UAV-benchmark-M \
  --annotations-root /path/to/GT \
  --train-sequences /path/to/uavdt_train_sequences.txt \
  --val-sequences /path/to/uavdt_val_sequences.txt \
  --output-root datasets/UAVDT \
  --splits-output-dir data_splits \
  --mode copy
```

The script writes:

```text
datasets/UAVDT/images/train/
datasets/UAVDT/images/val/
datasets/UAVDT/labels/train/
datasets/UAVDT/labels/val/
data_splits/uavdt_train.txt
data_splits/uavdt_val.txt
```

### BDD100K Real Motion-Blur Subset

The real BDD100K curation process is:

1. 1,000 candidate traffic images were initially collected from the BDD100K validation set.
2. A rule-based screening procedure retained 800 motion-blur candidates.
3. The 800 candidates were manually reviewed to reduce confounding degradations.
4. Images dominated by severe underexposure, rain or snow, severe occlusion, overexposure, or other unrelated degradation factors were excluded.
5. The remaining 500 images constituted the final real motion-blur evaluation subset.

Original BDD100K images are not redistributed in this repository. Users must
obtain BDD100K from the official source and follow the BDD100K license. The
paper's real motion-blur results use the final 500-image evaluation subset, not
the full 800-image rule-filtered candidate pool.

The repository provides:

- `data_splits/bdd100k_realblur_rule_filtered_800.txt`
- `data_splits/bdd100k_realblur_rule_filtered_800.csv`
- `data_splits/bdd100k_realblur_final_500.txt`
- `data_splits/bdd100k_realblur_final_500.csv`
- `data_splits/bdd100k_candidate_index_mapping.csv`
- `data_splits/bdd100k_realblur_manual_review.csv`
- `tools/map_bdd100k_indices_to_filenames.py`
- `tools/build_bdd100k_final_subset.py`
- `tools/validate_bdd100k_curation.py`

Map the manual review index file to filenames:

```bat
python tools\map_bdd100k_indices_to_filenames.py ^
  --candidate-root "D:\BDD100K\bdd100k_realblur_vehicle_clean" ^
  --index-file "D:\BDD100K\最终序号.txt" ^
  --output-dir data_splits
```

Linux/macOS equivalent:

```bash
python tools/map_bdd100k_indices_to_filenames.py \
  --candidate-root /path/to/bdd100k_realblur_vehicle_clean \
  --index-file /path/to/final_indices.txt \
  --output-dir data_splits
```

Validate the curation metadata:

```bash
python tools/validate_bdd100k_curation.py \
  --candidate-manifest data_splits/bdd100k_realblur_rule_filtered_800.csv \
  --final-manifest data_splits/bdd100k_realblur_final_500.csv \
  --review-record data_splits/bdd100k_realblur_manual_review.csv
```

Rebuild the final 500-image YOLO subset from the 800 candidates:

```bat
python tools\build_bdd100k_final_subset.py ^
  --source-root "D:\BDD100K\bdd100k_realblur_vehicle_clean" ^
  --manifest data_splits\bdd100k_realblur_final_500.txt ^
  --output-root "D:\BDD100K\bdd100k_realblur_final_500" ^
  --mode copy
```

Rebuild directly from official BDD100K validation images and labels:

```bash
python tools/prepare_bdd100k_realblur_vehicle.py \
  --annotations /path/to/bdd100k_labels_images_val.json \
  --images /path/to/bdd100k/images/100k/val \
  --image-list data_splits/bdd100k_realblur_final_500.txt \
  --out datasets/BDD100K_realblur
```

`car`, `bus`, `truck`, and `motor` are merged into class `0: vehicle`. Boxes are
clipped to image boundaries and invalid or tiny boxes are removed.

### Synthetic Motion-Blur Generation

Generate clear, light, medium, and heavy validation sets. Labels are copied
unchanged and source data is never overwritten.

```bash
python tools/make_motion_blur_val.py \
  --images datasets/UA-DETRAC/images/val \
  --labels datasets/UA-DETRAC/labels/val \
  --out datasets/UA-DETRAC_blur \
  --level all \
  --dataset-name ua_detrac
```

Kernel sets:

- Clear: original validation images.
- Blur-Light: `{9, 11, 13}`.
- Blur-Medium: `{15, 17, 19}`.
- Blur-Heavy: `{21, 23, 25}`.
- Direction: uniform in `[0, 180)`.

Generated YAMLs are written under `data/`.

## Dataset Splits and Metadata

Committed metadata:

| File | Description |
|---|---|
| `data_splits/bdd100k_realblur_rule_filtered_800.txt` | 800 rule-filtered candidate image names. |
| `data_splits/bdd100k_realblur_rule_filtered_800.csv` | Candidate index, image path, label path, label status, and vehicle count. |
| `data_splits/bdd100k_realblur_final_500.txt` | 500 final manually reviewed evaluation image names. |
| `data_splits/bdd100k_realblur_final_500.csv` | Final 500-image manifest with candidate indices and label checks. |
| `data_splits/bdd100k_candidate_index_mapping.csv` | Complete 800-row candidate index to filename mapping. |
| `data_splits/bdd100k_realblur_manual_review.csv` | 800-row manual review inclusion/exclusion record. |
| `data_splits/sequence_level_metrics_template.csv` | Blank template for statistical significance input. |

Export split metadata from any YOLO dataset YAML:

```bash
python tools/export_yolo_dataset_manifest.py \
  --data-yaml data/ua_detrac_vehicle.yaml \
  --output-dir data_splits
```

## Model Configuration

Default settings are in `configs/befr_default.yaml` and
`configs/yolov8n_befr.yaml`:

- Base detector: `yolov8n.pt`.
- Input size: `640`.
- Epochs: `100`.
- Batch size: `16`.
- Optimizer: `SGD`, `lr0=0.01`, `momentum=0.937`, `weight_decay=0.0005`.
- BE-Net: `l_min=9`, `l_max=25`, `p_min=0.2`, `p_max=0.8`,
  `alpha=2.0`, `beta=2.0`, `gaussian_noise_sigma=0.01`.
- FR-Net: P3/P4/P5 insertion, `lambda_spec=0.1`, L1 spectral distance,
  residual fusion.

Sensitivity grids are in `configs/sensitivity_lambda.yaml`,
`configs/sensitivity_alpha_beta.yaml`, `configs/sensitivity_p_blur.yaml`, and
`configs/sensitivity_l_range.yaml`.

## Training

YOLOv8 baseline:

```bash
python train_baseline_yolov8.py \
  --data data/ua_detrac_vehicle.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name ua_detrac_baseline
```

BE-only:

```bash
python train_ablation.py \
  --config configs/ablation_be_only.yaml \
  --data data/ua_detrac_vehicle.yaml \
  --use-be true \
  --use-fr false \
  --name ua_detrac_be_only
```

FR-only:

```bash
python train_ablation.py \
  --config configs/ablation_fr_only.yaml \
  --data data/ua_detrac_vehicle.yaml \
  --use-be false \
  --use-fr true \
  --lambda-spec 0.1 \
  --name ua_detrac_fr_only
```

Full BE-FR YOLO:

```bash
python train_befr.py \
  --data data/ua_detrac_vehicle.yaml \
  --model configs/yolov8n_befr.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name ua_detrac_befr
```

Sensitivity dry run:

```bash
python run_sensitivity.py \
  --config configs/sensitivity_lambda.yaml \
  --data data/ua_detrac_vehicle.yaml \
  --device 0 \
  --dry-run
```

## Evaluation

Evaluate all generated blur levels:

```bash
python eval_all_blur_levels.py \
  --weights runs/detect/ua_detrac_befr/weights/best.pt \
  --data-root datasets/UA-DETRAC_blur \
  --dataset-name ua_detrac \
  --imgsz 640 \
  --device 0 \
  --output results/metrics/ua_detrac_befr_all_levels.csv
```

Evaluate BDD100K final real blur:

```bash
python val_befr.py \
  --weights runs/detect/befr/weights/best.pt \
  --data data/bdd100k_realblur_vehicle.yaml \
  --name bdd100k_realblur_befr \
  --output results/metrics/bdd100k_realblur_befr.csv \
  --device 0
```

## Reproducing Manuscript Tables

The organized manuscript data are in `results/organized_data/`. These files
summarize values already present in the manuscript.

| Manuscript item | Configuration | Command | Input data | Output file |
|---|---|---|---|---|
| Tables 2-4: main comparisons | Baseline, external methods, BE-FR YOLO | Train/evaluate this repo for YOLOv8 and BE-FR; obtain external method code separately | UA-DETRAC, UAVDT, BDD100K final real blur | `results/organized_data/tables/table2_ua_detrac_comparison.csv`, `table3_uavdt_comparison.csv`, `table4_bdd100k_realblur.csv` |
| Table 5: ablation | `configs/ablation_be_only.yaml`, `configs/ablation_fr_only.yaml`, `configs/ablation_befr.yaml` | `python train_ablation.py ...` and `python eval_all_blur_levels.py ...` | UA-DETRAC clear/synthetic blur | `results/organized_data/tables/table5_ablation_ua_detrac.csv` |
| Table 6: statistical significance | Sequence-level mAP50 CSVs | `python tools/statistical_significance.py --model YOLOv8=/path/to/yolov8.csv --model BE-FR=/path/to/befr.csv --baseline YOLOv8 --metric map50 --output results/metrics/significance.csv` | Sequence-level evaluation CSVs with `sequence_id,map50` | New run CSV in `results/metrics/`; manuscript values in `results/organized_data/tables/table6_statistical_significance.csv` |
| Tables 7-10: sensitivity | `configs/sensitivity_*.yaml` | `python run_sensitivity.py --config configs/sensitivity_lambda.yaml --data data/ua_detrac_vehicle.yaml --device 0` | UA-DETRAC and generated blur levels | `results/organized_data/tables/table7_sensitivity_lambda.csv` through `table10_sensitivity_l_range.csv` |
| Table 11: complexity and efficiency | Trained checkpoints | `python tools/measure_complexity.py --weights runs/detect/befr/weights/best.pt --imgsz 640 --batch 1 --warmup 50 --repeat 300 --device 0 --output results/metrics/befr_complexity.csv` | Trained checkpoints | `results/organized_data/tables/table11_complexity_efficiency.csv` |

Third-party comparison implementations such as Deblur-YOLO, Adaptive Deblurring,
Feature-Level Deblurring, and RT-DETR-R18 are not redistributed here.

## Visualization

Prediction comparison:

```bash
python tools/visualize_predictions.py \
  --images /path/to/your/test_image.jpg \
  --weights YOLOv8=runs/detect/baseline/weights/best.pt \
  --weights BE-only=runs/detect/be_only/weights/best.pt \
  --weights FR-only=runs/detect/fr_only/weights/best.pt \
  --weights BE-FR=runs/detect/befr/weights/best.pt
```

Grad-CAM-like heatmap and FR-Net spectra:

```bash
python tools/gradcam_yolo.py \
  --weights runs/detect/befr/weights/best.pt \
  --image /path/to/your/test_image.jpg \
  --layer -2

python tools/visualize_fft_response.py \
  --weights runs/detect/befr/weights/best.pt \
  --image /path/to/your/test_image.jpg \
  --level P3
```

## Complexity and Efficiency

```bash
python tools/measure_complexity.py \
  --weights runs/detect/befr/weights/best.pt \
  --imgsz 640 \
  --batch 1 \
  --warmup 50 \
  --repeat 300 \
  --device 0 \
  --output results/metrics/befr_complexity.csv
```

FFT operations may not be counted by all FLOP profilers.

## Statistical Significance Analysis

`tools/statistical_significance.py` computes paired differences from
sequence-level CSV files and does not hard-code manuscript p-values.

Required input columns:

```text
sequence_id,map50
```

Create a blank input template:

```bash
python tools/statistical_significance.py \
  --write-template data_splits/sequence_level_metrics_template.csv
```

Run a paired comparison:

```bash
python tools/statistical_significance.py \
  --model YOLOv8=/path/to/yolov8_sequence_metrics.csv \
  --model BE-FR=/path/to/befr_sequence_metrics.csv \
  --baseline YOLOv8 \
  --metric map50 \
  --output results/metrics/significance_befr_vs_yolov8.csv
```

The script aligns by `sequence_id`, checks missing sequences, runs a
Shapiro-Wilk normality test when possible, selects a paired t-test or Wilcoxon
signed-rank test, and writes mean difference, test name, statistic, p-value, and
sample count.

## Citation

`CITATION.cff` contains the software citation metadata.

```bibtex
@software{befr_yolo_2026,
  title = {BE-FR YOLO: Progressive Blur Evolution and Frequency Refinement for Robust Vehicle Detection},
  author = {Li, Xiaodan and Li, Ran and Sun, Xiaohui},
  year = {2026},
  version = {0.1.0},
  note = {Software repository accompanying the submitted manuscript}
}
```

## License

Code in this repository is licensed under **GNU Affero General Public License
v3.0 only (AGPL-3.0-only)**. See `LICENSE`.

Third-party datasets, original images, annotations, and external model weights
are not covered by this repository license. See `THIRD_PARTY_NOTICES.md`.

## Contribution Guidelines

See `CONTRIBUTING.md`. Issues should include reproducible commands and
environment details. Pull requests should be scoped, include tests for new
tooling, and avoid committing restricted UA-DETRAC, UAVDT, or BDD100K data.
