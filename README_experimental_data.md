# Reported Experimental Data

The files in `results/organized_data/` summarize the experimental values
reported in the manuscript. They are intended for reproducibility
documentation, plotting, and consistency checking. They should not be
interpreted as newly reproduced results unless the corresponding training and
evaluation scripts are executed.

## Source and Scope

The extraction uses the manuscript titled **BE-FR YOLO: Progressive Blur
Evolution and Frequency Refinement for Robust Vehicle Detection**.

Configuration values come from:

- Sections 3.3-3.5 for BE-Net, FR-Net, and spectral-consistency settings.
- Section 4.2 for datasets, hardware, training settings, and synthetic-blur
  protocol.

Quantitative values come from manuscript Tables 2-11:

| File | Manuscript content |
|---|---|
| `table2_ua_detrac_comparison.csv` | UA-DETRAC main comparison |
| `table3_uavdt_comparison.csv` | UAVDT main comparison |
| `table4_bdd100k_realblur.csv` | BDD100K real motion-blur subset |
| `table5_ablation_ua_detrac.csv` | UA-DETRAC ablation study |
| `table6_statistical_significance.csv` | Sequence-level significance analysis |
| `table7_sensitivity_lambda.csv` | Spectral consistency weight sensitivity |
| `table8_sensitivity_alpha_beta.csv` | Blur evolution parameter sensitivity |
| `table9_sensitivity_p_blur.csv` | Blur injection probability sensitivity |
| `table10_sensitivity_l_range.csv` | Blur kernel range sensitivity |
| `table11_complexity_efficiency.csv` | Complexity and inference efficiency |

## Run Extraction

Install the Python extraction dependency:

```bash
pip install python-docx
```

Run against a DOCX manuscript:

```bash
python tools/extract_experimental_data.py --paper /path/to/BE-FR_YOLO_manuscript.docx
```

The extractor also supports Markdown and text manuscripts containing
pipe-delimited tables. PDF extraction deliberately stops with a conversion
recommendation because merged scientific tables cannot be parsed reliably enough
for this release.

Workbook creation uses `tools/build_experimental_workbook.mjs`. Set `BEFR_NODE`
or pass `--node` when Node.js is not on `PATH`. Use `--skip-xlsx` only when
CSV/JSON/YAML outputs are sufficient.

## Output Structure

```text
results/organized_data/
  experiment_config.yaml
  befr_experimental_results.json
  befr_experimental_results.xlsx
  derived_summary.csv
  data_validation_report.md
  missing_fields_report.md
  tables/
    table2_ua_detrac_comparison.csv
    ...
    table11_complexity_efficiency.csv
  plot_data/
    plot_main_comparison_ua_detrac_map50.csv
    ...
    plot_complexity_accuracy_tradeoff.csv
```

The JSON file combines configuration and every extracted table. The Excel
workbook places configuration and each manuscript table on separate sheets.
Plot-data CSVs provide long-format main-comparison and ablation data plus
simplified sensitivity and complexity tables.

## Validation and Interpretation

`data_validation_report.md` compares the requested manuscript claims with the
extracted tables and marks each check as `PASS`, `WARNING`, or `FAIL`.
`derived_summary.csv` contains only direct arithmetic differences between
extracted values.

Table 3 in the DOCX uses incorrect three-column grid spans for several
two-metric groups. The extraction script reads physical DOCX XML cells rather
than duplicated `python-docx` display cells, preserving the intended method plus
eight metrics.

These reported values complement the repository's training, blur-generation, and
evaluation scripts. Reproduction runs should write new measurements to separate
run/result locations and should not overwrite the organized manuscript data.
