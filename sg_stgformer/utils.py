from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEVEL_NAMES = ["poor", "moderate", "good", "excellent"]


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (root / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def make_run_dir(base_dir: str | Path) -> Path:
    root = ensure_dir(base_dir)
    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def score_to_label(score: float) -> int:
    if score < 2.5:
        return 0
    if score < 5.0:
        return 1
    if score < 7.5:
        return 2
    return 3


def label_to_name(label: int) -> str:
    return LEVEL_NAMES[label]


def format_metrics(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "loss",
        "mae",
        "rmse",
        "spearman",
        "accuracy",
        "f1",
        "regression_loss",
        "classification_loss",
        "contrastive_loss",
    ]
    parts = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in metrics:
        if key not in ordered_keys:
            parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)
