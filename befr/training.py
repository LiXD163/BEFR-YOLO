"""Shared training command-line and configuration logic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from befr.integration.trainer import BEFRDetectionTrainer
from befr.utils.config import deep_update, load_yaml, str2bool


DEFAULT_CONFIG = Path("configs/befr_default.yaml")


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_use_be: bool | None = None,
    default_use_fr: bool | None = None,
) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="BE-FR experiment YAML.")
    parser.add_argument(
        "--model",
        default=None,
        help="Base YOLO model/weights or a BE-FR project YAML such as configs/yolov8n_befr.yaml.",
    )
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--use-be", type=str2bool, default=default_use_be)
    parser.add_argument("--use-fr", type=str2bool, default=default_use_fr)
    parser.add_argument("--lambda-spec", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--p-min", type=float, default=None)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--l-min", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--gaussian-noise-sigma", type=float, default=None)
    return parser


def _load_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    if args.model and Path(args.model).suffix.lower() in {".yaml", ".yml"} and Path(args.model).exists():
        candidate = load_yaml(args.model)
        if "be_net" in candidate or "fr_net" in candidate or "base_model" in candidate:
            config_path = Path(args.model)
    config = load_yaml(config_path)
    return config


def _normal_model_name(value: str) -> str:
    if value == "yolov8n":
        return "yolov8n.pt"
    return value


def resolved_training_settings(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_experiment_config(args)
    base_model = config.get("base_model", config.get("model", "yolov8n.pt"))
    if args.model and not (
        Path(args.model).suffix.lower() in {".yaml", ".yml"} and Path(args.model).exists()
    ):
        base_model = args.model
    base_model = _normal_model_name(str(base_model))

    be_config = dict(config.get("be_net", {}))
    fr_config = dict(config.get("fr_net", {}))
    be_overrides = {
        "enabled": args.use_be,
        "alpha": args.alpha,
        "beta": args.beta,
        "p_min": args.p_min,
        "p_max": args.p_max,
        "l_min": args.l_min,
        "l_max": args.l_max,
        "gaussian_noise_sigma": args.gaussian_noise_sigma,
        "seed": args.seed,
    }
    fr_overrides = {"enabled": args.use_fr, "lambda_spec": args.lambda_spec}
    be_config = deep_update(be_config, {key: value for key, value in be_overrides.items() if value is not None})
    fr_config = deep_update(fr_config, {key: value for key, value in fr_overrides.items() if value is not None})

    use_be = bool(be_config.get("enabled", True))
    use_fr = bool(fr_config.get("enabled", True))
    if not use_fr:
        fr_config["lambda_spec"] = 0.0
    default_name = "befr" if use_be and use_fr else "be_only" if use_be else "fr_only" if use_fr else "baseline"
    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    training_overrides = {
        "model": base_model,
        "data": args.data,
        "epochs": args.epochs if args.epochs is not None else int(config.get("epochs", 100)),
        "imgsz": args.imgsz if args.imgsz is not None else int(config.get("imgsz", 640)),
        "batch": args.batch if args.batch is not None else int(config.get("batch", 16)),
        "optimizer": args.optimizer or config.get("optimizer", "SGD"),
        "lr0": args.lr0 if args.lr0 is not None else float(config.get("lr0", 0.01)),
        "momentum": args.momentum if args.momentum is not None else float(config.get("momentum", 0.937)),
        "weight_decay": (
            args.weight_decay if args.weight_decay is not None else float(config.get("weight_decay", 0.0005))
        ),
        "device": device,
        "workers": args.workers,
        "seed": args.seed,
        "project": args.project,
        "name": args.name or default_name,
        "exist_ok": args.exist_ok,
    }
    return training_overrides, be_config, fr_config


def run_befr_training(args: argparse.Namespace) -> None:
    overrides, be_config, fr_config = resolved_training_settings(args)
    print(f"Training variant: BE={be_config.get('enabled', True)}, FR={fr_config.get('enabled', True)}")
    print(f"Base model: {overrides['model']}")
    trainer = BEFRDetectionTrainer(overrides=overrides, be_config=be_config, fr_config=fr_config)
    trainer.train()
