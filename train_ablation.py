"""Train baseline-compatible BE-only, FR-only, or BE-FR ablations."""

from __future__ import annotations

import argparse

from befr.training import add_common_training_arguments, run_befr_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable BE-FR ablation.")
    return add_common_training_arguments(parser).parse_args()


if __name__ == "__main__":
    run_befr_training(parse_args())
