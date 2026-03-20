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
from sg_stgformer.engine import evaluate
from sg_stgformer.losses import QualityAssessmentLoss
from sg_stgformer.model import SGSTGFormer
from sg_stgformer.utils import format_metrics


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
    parser = argparse.ArgumentParser(description="Evaluate a trained SG-STGFormer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Optional CSV file for predictions.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    loaders = create_dataloaders(config, ROOT)
    model = SGSTGFormer(config)
    model.load_state_dict(checkpoint["model_state"])
    criterion = QualityAssessmentLoss(
        regression_weight=config["loss"]["regression_weight"],
        classification_weight=config["loss"]["classification_weight"],
        contrastive_weight=config["loss"]["contrastive_weight"],
        temperature=config["loss"]["temperature"],
        label_smoothing=config["train"]["label_smoothing"],
    )

    metrics, predictions = evaluate(
        model,
        loaders[args.split],
        criterion,
        torch.device("cpu"),
        num_classes=config["data"]["num_classes"],
    )
    print(f"{args.split} metrics: {format_metrics(metrics)}")

    if args.output:
        output_path = Path(args.output).resolve()
        write_predictions_csv(output_path, predictions)
        print(f"Predictions written to {output_path}")
    else:
        summary_path = checkpoint_path.with_name(f"{args.split}_metrics.json")
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics written to {summary_path}")


if __name__ == "__main__":
    main()
