from __future__ import annotations

import csv

from tools.export_bdd100k_realblur_manifest import build_manifest
from tools.statistical_significance import read_metric_csv


def test_bdd_manifest_counts_valid_labels(tmp_path):
    root = tmp_path / "bdd"
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / "sample.jpg").write_bytes(b"not decoded by manifest")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    rows, summary, warnings = build_manifest(root)

    assert warnings == []
    assert summary["images"] == 1
    assert summary["labels"] == 1
    assert summary["total_vehicles"] == 1
    assert rows[0].image_name == "sample.jpg"
    assert rows[0].relative_image_path == "images/val/sample.jpg"


def test_sequence_metric_reader_requires_unique_sequence_ids(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "map50"])
        writer.writeheader()
        writer.writerow({"sequence_id": "seq1", "map50": "0.5"})

    assert read_metric_csv(csv_path, "sequence_id", "map50") == {"seq1": 0.5}
