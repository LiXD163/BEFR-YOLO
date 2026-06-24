"""Map manually reviewed BDD100K candidate indices to original file names."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from befr.data.utils import (
    find_images,
    image_label_relative_path,
    infer_index_base,
    parse_integer_indices,
    posix,
    validate_yolo_label,
    write_csv,
    write_lines,
)


EXPECTED_CANDIDATES = 800
EXPECTED_FINAL = 500
EXCLUDED_REASON = "confounding_degradation_removed_during_manual_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map BDD100K manual-review indices to candidate image names.")
    parser.add_argument("--candidate-root", required=True, type=Path, help="Prepared 800-image candidate dataset root.")
    parser.add_argument("--index-file", required=True, type=Path, help="Text file containing retained candidate indices.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated manifests.")
    parser.add_argument("--expected-candidates", type=int, default=EXPECTED_CANDIDATES)
    parser.add_argument("--expected-final", type=int, default=EXPECTED_FINAL)
    return parser.parse_args()


def image_root(candidate_root: Path) -> Path:
    root = candidate_root / "images"
    return root if root.is_dir() else candidate_root


def label_root(candidate_root: Path) -> Path:
    root = candidate_root / "labels"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {root}")
    return root


def build_candidate_rows(candidate_root: Path, base: int) -> list[dict[str, object]]:
    images_base = image_root(candidate_root)
    labels_base = label_root(candidate_root)
    images = find_images(images_base)
    if not images:
        raise FileNotFoundError(f"No candidate images found under {images_base}")
    rows = []
    seen_names: set[str] = set()
    duplicate_names = []
    for offset, image_path in enumerate(images):
        relative_under_images = image_path.relative_to(images_base)
        image_name = image_path.name
        if image_name in seen_names:
            duplicate_names.append(image_name)
        seen_names.add(image_name)
        relative_image_path = Path("images") / relative_under_images
        relative_label_path = image_label_relative_path(relative_image_path)
        label_path = labels_base / relative_under_images.with_suffix(".txt")
        validation = validate_yolo_label(label_path)
        rows.append(
            {
                "candidate_index": offset + base,
                "image_name": image_name,
                "relative_image_path": posix(relative_image_path),
                "relative_label_path": posix(relative_label_path),
                "label_exists": str(validation.exists).lower(),
                "vehicle_count": validation.valid_count,
                "invalid_label_count": validation.invalid_count,
            }
        )
    if duplicate_names:
        raise ValueError(f"Duplicate candidate image names: {duplicate_names[:10]}")
    return rows


def main() -> int:
    args = parse_args()
    try:
        candidate_root = args.candidate_root.expanduser().resolve()
        if not candidate_root.is_dir():
            raise FileNotFoundError(f"Candidate root not found: {candidate_root}")
        if not args.index_file.is_file():
            raise FileNotFoundError(f"Index file not found: {args.index_file}")

        raw_indices = parse_integer_indices(args.index_file)
        duplicate_indices = sorted(index for index, count in Counter(raw_indices).items() if count > 1)
        base, base_note = infer_index_base(raw_indices, args.expected_candidates)
        out_of_bounds = sorted(index for index in raw_indices if index < base or index >= args.expected_candidates + base)
        if duplicate_indices:
            raise ValueError(f"Duplicate manual-review indices: {duplicate_indices[:20]}")
        if out_of_bounds:
            raise ValueError(f"Out-of-bounds manual-review indices for {base}-based candidates: {out_of_bounds[:20]}")

        candidate_rows = build_candidate_rows(candidate_root, base)
        if len(candidate_rows) != args.expected_candidates:
            raise ValueError(f"Expected {args.expected_candidates} candidate images, found {len(candidate_rows)}")
        index_to_row = {int(row["candidate_index"]): row for row in candidate_rows}
        final_indices = sorted(raw_indices)
        final_rows = []
        for index in final_indices:
            row = dict(index_to_row[index])
            row["final_included"] = "true"
            final_rows.append(row)
        if len(final_rows) != args.expected_final:
            raise ValueError(f"Expected {args.expected_final} final indices, found {len(final_rows)}")
        if len({row["image_name"] for row in final_rows}) != args.expected_final:
            raise ValueError("Final index file does not map to 500 unique images.")

        final_set = {int(row["candidate_index"]) for row in final_rows}
        review_rows = []
        for row in candidate_rows:
            index = int(row["candidate_index"])
            included = index in final_set
            review_rows.append(
                {
                    "candidate_index": index,
                    "image_name": row["image_name"],
                    "final_included": str(included).lower(),
                    "review_stage": "manual_review",
                    "exclusion_reason": "" if included else EXCLUDED_REASON,
                }
            )

        output_dir = args.output_dir
        candidate_fields = [
            "candidate_index",
            "image_name",
            "relative_image_path",
            "relative_label_path",
            "label_exists",
            "vehicle_count",
        ]
        final_fields = candidate_fields + ["final_included"]
        mapping_fields = ["candidate_index", "image_name", "relative_image_path", "relative_label_path"]
        review_fields = ["candidate_index", "image_name", "final_included", "review_stage", "exclusion_reason"]

        write_lines(output_dir / "bdd100k_realblur_rule_filtered_800.txt", [str(row["image_name"]) for row in candidate_rows])
        write_csv(output_dir / "bdd100k_realblur_rule_filtered_800.csv", candidate_fields, candidate_rows)
        write_lines(output_dir / "bdd100k_realblur_final_500.txt", [str(row["image_name"]) for row in final_rows])
        write_csv(output_dir / "bdd100k_realblur_final_500.csv", final_fields, final_rows)
        write_csv(output_dir / "bdd100k_candidate_index_mapping.csv", mapping_fields, candidate_rows)
        write_csv(output_dir / "bdd100k_realblur_manual_review.csv", review_fields, review_rows)

        missing_labels = sum(1 for row in candidate_rows if row["label_exists"] != "true")
        invalid_labels = sum(1 for row in candidate_rows if int(row["invalid_label_count"]) > 0)
        print(f"Candidate images: {len(candidate_rows)}")
        print(f"Manual-review indices: {len(raw_indices)}")
        print(f"Unique final images: {len(final_rows)}")
        print(f"Excluded images: {len(candidate_rows) - len(final_rows)}")
        print(f"Index base: {base}-based ({base_note})")
        print(f"Duplicate indices: {len(duplicate_indices)}")
        print(f"Out-of-bounds indices: {len(out_of_bounds)}")
        print(f"Missing labels: {missing_labels}")
        print(f"Invalid label files: {invalid_labels}")
        print(f"Output directory: {output_dir}")
        return 0 if missing_labels == 0 and invalid_labels == 0 else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
