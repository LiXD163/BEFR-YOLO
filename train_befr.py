"""Train the full BE-FR YOLO model."""

from __future__ import annotations

import argparse

from befr.training import add_common_training_arguments, run_befr_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BE-FR YOLO.")
    return add_common_training_arguments(parser, default_use_be=True, default_use_fr=True).parse_args()


if __name__ == "__main__":
    run_befr_training(parse_args())
