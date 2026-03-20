from __future__ import annotations

import numpy as np
import torch

from .metrics import compute_metrics


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_items = 0
    sums = {"loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0, "contrastive_loss": 0.0}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        losses = criterion(outputs, batch["score"], batch["label"])
        losses["loss"].backward()
        optimizer.step()

        batch_size = batch["x"].size(0)
        total_items += batch_size
        for key in sums:
            sums[key] += float(losses[key].item()) * batch_size

    return {key: value / max(total_items, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    *,
    num_classes: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    total_items = 0
    sums = {"loss": 0.0, "regression_loss": 0.0, "classification_loss": 0.0, "contrastive_loss": 0.0}

    score_true = []
    score_pred = []
    label_true = []
    label_pred = []
    sample_ids = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["x"])
        losses = criterion(outputs, batch["score"], batch["label"])

        batch_size = batch["x"].size(0)
        total_items += batch_size
        for key in sums:
            sums[key] += float(losses[key].item()) * batch_size

        score_true.append(batch["score"].detach().cpu().numpy())
        score_pred.append(outputs["score"].detach().cpu().numpy())
        label_true.append(batch["label"].detach().cpu().numpy())
        label_pred.append(outputs["logits"].argmax(dim=-1).detach().cpu().numpy())
        sample_ids.extend(batch["sample_id"])

    score_true_np = np.concatenate(score_true) if score_true else np.empty(0)
    score_pred_np = np.concatenate(score_pred) if score_pred else np.empty(0)
    label_true_np = np.concatenate(label_true) if label_true else np.empty(0, dtype=np.int64)
    label_pred_np = np.concatenate(label_pred) if label_pred else np.empty(0, dtype=np.int64)

    metrics = {key: value / max(total_items, 1) for key, value in sums.items()}
    metrics.update(
        compute_metrics(
            score_true_np,
            score_pred_np,
            label_true_np,
            label_pred_np,
            num_classes=num_classes,
        )
    )
    return metrics, {
        "sample_id": np.array(sample_ids),
        "score_true": score_true_np,
        "score_pred": score_pred_np,
        "label_true": label_true_np,
        "label_pred": label_pred_np,
    }
