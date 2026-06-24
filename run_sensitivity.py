"""Run grid experiments described by configs/sensitivity_*.yaml."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any

from befr.utils.config import load_yaml


FLAG_MAP = {
    "fr_net.lambda_spec": "--lambda-spec",
    "be_net.alpha": "--alpha",
    "be_net.beta": "--beta",
    "be_net.p_min": "--p-min",
    "be_net.p_max": "--p-max",
    "be_net.l_min": "--l-min",
    "be_net.l_max": "--l-max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BE-FR sensitivity grid.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/sensitivity")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def value_flags(key: str, value: Any) -> tuple[list[str], str]:
    if key == "be_net.p_range":
        return ["--p-min", str(value[0]), "--p-max", str(value[1])], f"p{value[0]}-{value[1]}"
    if key == "be_net.l_range":
        return ["--l-min", str(value[0]), "--l-max", str(value[1])], f"l{value[0]}-{value[1]}"
    if key not in FLAG_MAP:
        raise KeyError(f"Unsupported sweep key: {key}")
    return [FLAG_MAP[key], str(value)], f"{key.split('.')[-1]}{value}"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    base_config = config.get("base_config")
    sweep = config.get("sweep")
    if not base_config or not isinstance(sweep, dict) or not sweep:
        raise ValueError("Sensitivity YAML must define base_config and a non-empty sweep mapping.")
    keys = list(sweep)
    for values in itertools.product(*(sweep[key] for key in keys)):
        command = [
            sys.executable,
            "train_ablation.py",
            "--config",
            str(base_config),
            "--data",
            args.data,
            "--project",
            args.project,
        ]
        labels = []
        for key, value in zip(keys, values):
            flags, label = value_flags(key, value)
            command.extend(flags)
            labels.append(label)
        command.extend(["--name", "_".join(labels)])
        if args.device is not None:
            command.extend(["--device", args.device])
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
