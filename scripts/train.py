from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sg_stgformer.data import create_dataloaders
from sg_stgformer.engine import evaluate, train_one_epoch
from sg_stgformer.losses import QualityAssessmentLoss
from sg_stgformer.model import SGSTGFormer
from sg_stgformer.utils import format_metrics, get_device, load_json, make_run_dir, set_seed


def write_predictions_csv(path: Path, payload: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "score_true", "score_pred", "label_true", "label_pred"])
        for row in zip(
            payload["sample_id"],
            payload["score_true"],
            payload["score_pred"],
            payload["label_true"],
            payload["label_pred"],
        ):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SG-STGFormer.")
    parser.add_argument("--config", default="configs/ttedu_default.json")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu, cuda, mps.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--run-root", default="runs", help="Directory for checkpoints and logs.")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    config = load_json(config_path)
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    set_seed(config["seed"])
    device = get_device(config["train"]["device"])
    print(f"Using device: {device}")

    loaders = create_dataloaders(config, ROOT)
    model = SGSTGFormer(config).to(device)
    criterion = QualityAssessmentLoss(
        regression_weight=config["loss"]["regression_weight"],
        classification_weight=config["loss"]["classification_weight"],
        contrastive_weight=config["loss"]["contrastive_weight"],
        temperature=config["loss"]["temperature"],
        label_smoothing=config["train"]["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    run_dir = make_run_dir(ROOT / args.run_root)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_mae = float("inf")
    best_epoch = 0
    patience = config["train"]["patience"]
    patience_counter = 0

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics, _ = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes=config["data"]["num_classes"],
        )
        print(f"Epoch {epoch:03d} | train: {format_metrics(train_metrics)}")
        print(f"Epoch {epoch:03d} |   val: {format_metrics(val_metrics)}")

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_val_metrics": val_metrics,
                    "epoch": epoch,
                },
                run_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_predictions = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        num_classes=config["data"]["num_classes"],
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "test_metrics": test_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_predictions_csv(run_dir / "test_predictions.csv", test_predictions)

    print(f"Best epoch: {best_epoch}")
    print(f"Test metrics: {format_metrics(test_metrics)}")
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
