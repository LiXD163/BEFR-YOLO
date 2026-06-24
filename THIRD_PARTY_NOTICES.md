# Third-Party Notices

This repository integrates with third-party software and datasets. The BE-FR
YOLO source code is licensed under AGPL-3.0-only. Dataset images, original
annotations, and external comparison-method implementations remain governed by
their own licenses and terms.

## Software

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
  - BE-FR YOLO uses the official Ultralytics package rather than vendoring a
    modified copy.
  - Ultralytics YOLO is distributed under AGPL-3.0 with separate enterprise
    licensing options. This repository therefore uses AGPL-3.0-only.
- PyTorch and Torchvision: https://pytorch.org/
- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- PyYAML: https://pyyaml.org/
- Pillow: https://python-pillow.org/
- Matplotlib: https://matplotlib.org/
- THOP: https://github.com/Lyken17/pytorch-OpCounter

Consult each upstream project for its authoritative license text.

## Datasets

- UA-DETRAC: Wen et al., "UA-DETRAC: A new benchmark and protocol for
  multi-object detection and tracking," Computer Vision and Image
  Understanding, 2020, DOI: `10.1016/j.cviu.2020.102907`.
  Dataset page: https://detrac-db.rit.albany.edu/
- UAVDT: Du et al., "The Unmanned Aerial Vehicle Benchmark: Object Detection and
  Tracking," ECCV 2018. Project page: https://sites.google.com/view/grli-uavdt/
- BDD100K: Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous
  Multitask Learning," CVPR 2020. Project page: https://bdd-data.berkeley.edu/

This repository does not redistribute original UA-DETRAC, UAVDT, or BDD100K
images or annotations. Users must obtain each dataset from its official source.

## Comparison Methods

The manuscript compares BE-FR YOLO with external methods including Deblur-YOLO,
Adaptive Deblurring, Feature-Level Deblurring, and RT-DETR-R18. Their full
upstream implementations are not redistributed here. Users should obtain those
projects from their authors or official repositories when reproducing
cross-method tables.
