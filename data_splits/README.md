# Dataset Splits and Manifests

This directory stores public, text-based metadata that helps reconstruct the
datasets used by BE-FR YOLO without redistributing restricted third-party
images.

## BDD100K Real Motion-Blur Curation

| File | Purpose |
|---|---|
| `bdd100k_realblur_rule_filtered_800.txt` | 800 rule-filtered candidate image names in stable Windows-style natural order. |
| `bdd100k_realblur_rule_filtered_800.csv` | 800-row candidate manifest with candidate index, image path, label path, label status, and vehicle count. |
| `bdd100k_realblur_final_500.txt` | 500 final manually reviewed evaluation image names. |
| `bdd100k_realblur_final_500.csv` | 500-row final manifest with candidate index, image path, label path, label status, vehicle count, and `final_included=true`. |
| `bdd100k_candidate_index_mapping.csv` | Complete mapping from manual-review candidate indices to original image names. |
| `bdd100k_realblur_manual_review.csv` | 800-row manual review record. Included images have a blank exclusion reason; excluded images use the shared reason `confounding_degradation_removed_during_manual_review`. |

The final indices were interpreted as 1-based because the index file contains no
zero and all values fall within `1..800`. The same values also fall within the
0-based numeric range, so this file records the explicit convention used for
reconstruction.

Regenerate these files with:

```bash
python tools/map_bdd100k_indices_to_filenames.py \
  --candidate-root /path/to/bdd100k_realblur_vehicle_clean \
  --index-file /path/to/final_indices.txt \
  --output-dir data_splits
```

Validate them with:

```bash
python tools/validate_bdd100k_curation.py \
  --candidate-manifest data_splits/bdd100k_realblur_rule_filtered_800.csv \
  --final-manifest data_splits/bdd100k_realblur_final_500.csv \
  --review-record data_splits/bdd100k_realblur_manual_review.csv
```

## UA-DETRAC and UAVDT Splits

When the original datasets are converted with `tools/prepare_ua_detrac_vehicle.py`
and `tools/prepare_uavdt_vehicle.py`, this directory receives:

```text
ua_detrac_train.txt
ua_detrac_val.txt
uavdt_train.txt
uavdt_val.txt
```

Each split file contains one dataset-root-relative image path per line.

## Statistical Testing

`sequence_level_metrics_template.csv` is a blank input template for
`tools/statistical_significance.py` with columns:

```text
sequence_id,map50
```
