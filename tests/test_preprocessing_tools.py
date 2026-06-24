from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(128, 128, 128)).save(path)


def test_prepare_ua_detrac_vehicle_minimal(tmp_path):
    train_images = tmp_path / "Insight-MVT_Train" / "MVI_00001"
    val_images = tmp_path / "Insight-MVT_Test" / "MVI_00002"
    write_image(train_images / "img00001.jpg")
    write_image(val_images / "img00001.jpg")
    train_ann = tmp_path / "Insight-MVT_Annotation_Train"
    val_ann = tmp_path / "Insight-MVT_Annotation_Test"
    train_ann.mkdir()
    val_ann.mkdir()
    xml_template = """<sequence name="{sequence}">
  <ignored_region />
  <frame num="1">
    <target_list>
      <target id="1">
        <box left="10" top="12" width="30" height="20" />
        <attribute vehicle_type="car" truncation_ratio="0" occlusion="0" />
      </target>
    </target_list>
  </frame>
</sequence>
"""
    (train_ann / "MVI_00001.xml").write_text(xml_template.format(sequence="MVI_00001"), encoding="utf-8")
    (val_ann / "MVI_00002.xml").write_text(xml_template.format(sequence="MVI_00002"), encoding="utf-8")
    out = tmp_path / "datasets" / "UA-DETRAC"
    splits = tmp_path / "splits"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "prepare_ua_detrac_vehicle.py"),
            "--train-images",
            str(tmp_path / "Insight-MVT_Train"),
            "--train-annotations",
            str(train_ann),
            "--val-images",
            str(tmp_path / "Insight-MVT_Test"),
            "--val-annotations",
            str(val_ann),
            "--output-root",
            str(out),
            "--splits-output-dir",
            str(splits),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "labels" / "train" / "MVI_00001" / "img00001.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (splits / "ua_detrac_train.txt").read_text(encoding="utf-8").strip() == "images/train/MVI_00001/img00001.jpg"


def test_prepare_uavdt_vehicle_minimal(tmp_path):
    images_root = tmp_path / "UAV-benchmark-M"
    annotations_root = tmp_path / "GT"
    write_image(images_root / "M0101" / "img000001.jpg")
    write_image(images_root / "M0201" / "img000001.jpg")
    annotations_root.mkdir()
    (annotations_root / "M0101_gt_whole.txt").write_text("1,1,10,12,30,20,0,0,1\n", encoding="utf-8")
    (annotations_root / "M0201_gt_whole.txt").write_text("1,1,5,6,20,15,0,0,1\n", encoding="utf-8")
    train_seq = tmp_path / "train.txt"
    val_seq = tmp_path / "val.txt"
    train_seq.write_text("M0101\n", encoding="utf-8")
    val_seq.write_text("M0201\n", encoding="utf-8")
    out = tmp_path / "datasets" / "UAVDT"
    splits = tmp_path / "splits"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "prepare_uavdt_vehicle.py"),
            "--images-root",
            str(images_root),
            "--annotations-root",
            str(annotations_root),
            "--train-sequences",
            str(train_seq),
            "--val-sequences",
            str(val_seq),
            "--output-root",
            str(out),
            "--splits-output-dir",
            str(splits),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "labels" / "train" / "M0101" / "img000001.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (splits / "uavdt_val.txt").read_text(encoding="utf-8").strip() == "images/val/M0201/img000001.jpg"
