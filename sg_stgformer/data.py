from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import resolve_path


class SkeletonSequenceDataset(Dataset):
    def __init__(self, npz_path: str | Path) -> None:
        payload = np.load(npz_path, allow_pickle=True)
        self.inputs = payload["x"].astype(np.float32)
        self.scores = payload["score"].astype(np.float32)
        self.labels = payload["label"].astype(np.int64)
        sample_id = payload["sample_id"] if "sample_id" in payload.files else None
        if sample_id is None:
            sample_id = np.array([f"{Path(npz_path).stem}_{idx}" for idx in range(len(self.inputs))])
        self.sample_id = sample_id.astype(str)

        if self.inputs.ndim != 4:
            raise ValueError(f"Expected x to have shape [N, T, V, C], got {self.inputs.shape}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "x": torch.from_numpy(self.inputs[index]),
            "score": torch.tensor(self.scores[index], dtype=torch.float32),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "sample_id": self.sample_id[index],
        }


def create_dataloaders(config: dict, base_dir: str | Path | None = None) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["train"]

    paths = {
        split: resolve_path(data_cfg[f"{split}_path"], base_dir)
        for split in ("train", "val", "test")
    }

    datasets = {split: SkeletonSequenceDataset(path) for split, path in paths.items()}
    common_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
        "pin_memory": False,
    }

    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common_kwargs),
        "val": DataLoader(datasets["val"], shuffle=False, **common_kwargs),
        "test": DataLoader(datasets["test"], shuffle=False, **common_kwargs),
    }
