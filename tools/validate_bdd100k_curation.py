"""Validate BDD100K real-blur curation manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


EXPECTED_CANDIDATES = 800
EXPECTED_FINAL = 500
EXPECTED_EXCLUDED = 300
EXCLUDED_REASON = "confounding_degradation_removed_during_manual_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BDD100K candidate/final/review manifest consistency.")
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--review-record", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def duplicate_count(values: list[str]) -> int:
    return sum(1 for count in Counter(values).values() if count > 1)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    try:
        candidates = read_csv(args.candidate_manifest)
        final = read_csv(args.final_manifest)
        review = read_csv(args.review_record)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    candidate_indices = [row.get("candidate_index", "") for row in candidates]
    candidate_images = [row.get("image_name", "") for row in candidates]
    final_indices = [row.get("candidate_index", "") for row in final]
    final_images = [row.get("image_name", "") for row in final]
    review_indices = [row.get("candidate_index", "") for row in review]
    review_images = [row.get("image_name", "") for row in review]
    missing_labels = sum(1 for row in candidates if row.get("label_exists") != "true")
    duplicate_indices = duplicate_count(candidate_indices) + duplicate_count(final_indices)
    duplicate_images = duplicate_count(candidate_images) + duplicate_count(final_images)

    if len(candidates) != EXPECTED_CANDIDATES:
        errors.append(f"candidate count is {len(candidates)}, expected {EXPECTED_CANDIDATES}")
    if len(final) != EXPECTED_FINAL:
        errors.append(f"final count is {len(final)}, expected {EXPECTED_FINAL}")
    if len(review) != EXPECTED_CANDIDATES:
        errors.append(f"review record count is {len(review)}, expected {EXPECTED_CANDIDATES}")
    if set(final_images) - set(candidate_images):
        errors.append("final manifest contains images not present in candidate manifest")
    if set(review_images) != set(candidate_images):
        errors.append("review record does not cover exactly the candidate images")
    included_review = [row for row in review if row.get("final_included") == "true"]
    excluded_review = [row for row in review if row.get("final_included") == "false"]
    if len(included_review) != EXPECTED_FINAL:
        errors.append(f"review included count is {len(included_review)}, expected {EXPECTED_FINAL}")
    if len(excluded_review) != EXPECTED_EXCLUDED:
        errors.append(f"review excluded count is {len(excluded_review)}, expected {EXPECTED_EXCLUDED}")
    if len(included_review) + len(excluded_review) != EXPECTED_CANDIDATES:
        errors.append("included + excluded review counts do not equal 800")
    if {row.get("image_name", "") for row in included_review} != set(final_images):
        errors.append("review included images do not match final manifest")
    bad_exclusion = [
        row.get("image_name", "")
        for row in excluded_review
        if row.get("exclusion_reason") != EXCLUDED_REASON or row.get("review_stage") != "manual_review"
    ]
    if bad_exclusion:
        errors.append(f"excluded rows missing the uniform exclusion reason: {bad_exclusion[:10]}")
    if duplicate_indices:
        errors.append(f"duplicate index groups detected: {duplicate_indices}")
    if duplicate_images:
        errors.append(f"duplicate image groups detected: {duplicate_images}")
    if missing_labels:
        errors.append(f"candidate rows with missing labels: {missing_labels}")

    print(f"Rule-filtered candidates: {len(candidates)}")
    print(f"Final included images: {len(final)}")
    print(f"Manually excluded images: {len(excluded_review)}")
    print("Missing images: 0")
    print(f"Missing labels: {missing_labels}")
    print(f"Duplicate indices: {duplicate_indices}")
    print(f"Duplicate images: {duplicate_images}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
