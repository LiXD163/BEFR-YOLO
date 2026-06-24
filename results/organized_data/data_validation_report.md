# Experimental Data Validation Report

- Manuscript: local DOCX manuscript used during extraction; original absolute path omitted for portability.
- Scope: consistency between requested textual claims and extracted manuscript tables.
- Status rule: PASS for exact agreement within 0.0005; WARNING within 0.0015; otherwise FAIL.

| Status | Claim | Table value | Claimed value | Difference |
|---|---|---:|---:|---:|
| PASS | UA-DETRAC Blur-Heavy YOLOv8 mAP50 | 0.714000 | 0.714000 | 0.000000 |
| PASS | UA-DETRAC Blur-Heavy BE-FR YOLO mAP50 | 0.818000 | 0.818000 | 0.000000 |
| PASS | UA-DETRAC Blur-Heavy mAP50 improvement | 0.104000 | 0.104000 | -0.000000 |
| PASS | UAVDT Blur-Heavy YOLOv8 mAP50 | 0.171000 | 0.171000 | 0.000000 |
| PASS | UAVDT Blur-Heavy BE-FR YOLO mAP50 | 0.286000 | 0.286000 | 0.000000 |
| PASS | UAVDT Blur-Heavy mAP50 improvement | 0.115000 | 0.115000 | -0.000000 |
| PASS | BDD100K real-blur YOLOv8 mAP50 | 0.446000 | 0.446000 | 0.000000 |
| PASS | BDD100K real-blur BE-FR YOLO mAP50 | 0.556000 | 0.556000 | 0.000000 |
| PASS | BDD100K real-blur mAP50 improvement | 0.110000 | 0.110000 | 0.000000 |
| PASS | YOLOv8 parameters (M) | 3.200000 | 3.200000 | 0.000000 |
| PASS | BE-FR YOLO parameters (M) | 4.400000 | 4.400000 | 0.000000 |
| PASS | Parameter increase (M) | 1.200000 | 1.200000 | 0.000000 |
| PASS | YOLOv8 FLOPs (G) | 8.700000 | 8.700000 | 0.000000 |
| PASS | BE-FR YOLO FLOPs (G) | 11.600000 | 11.600000 | 0.000000 |
| PASS | FLOPs increase (G) | 2.900000 | 2.900000 | 0.000000 |
| PASS | YOLOv8 FPS | 112.400000 | 112.400000 | 0.000000 |
| PASS | BE-FR YOLO FPS | 109.700000 | 109.700000 | 0.000000 |

## Summary

- PASS: 17
- WARNING: 0
- FAIL: 0

## Extraction Notes

- UAVDT Table 3 contains visually merged cells with three-column grid spans. The extractor reads physical DOCX XML cells, yielding the intended method plus eight metric values.
- Statistical significance values are sequence-level mean differences and are preserved separately from differences computed from aggregate Table 5 values.
