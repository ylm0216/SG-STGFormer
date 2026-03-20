from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sg_stgformer.model import SGSTGFormer
from sg_stgformer.utils import LEVEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-sample prediction with SG-STGFormer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", default="data/demo/test.npz", help="NPZ file that contains x/score/label arrays.")
    parser.add_argument("--index", type=int, default=0, help="Sample index inside the NPZ file.")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location="cpu")
    config = checkpoint["config"]
    payload = np.load((ROOT / args.npz).resolve(), allow_pickle=True)

    x = torch.from_numpy(payload["x"][args.index]).float().unsqueeze(0)
    true_score = float(payload["score"][args.index]) if "score" in payload.files else None
    true_label = int(payload["label"][args.index]) if "label" in payload.files else None

    model = SGSTGFormer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        outputs = model(x)
        pred_score = float(outputs["score"].item())
        pred_label = int(outputs["logits"].argmax(dim=-1).item())

    print(f"Predicted score: {pred_score:.3f}")
    print(f"Predicted level: {LEVEL_NAMES[pred_label]}")
    if true_score is not None:
        print(f"Ground truth score: {true_score:.3f}")
    if true_label is not None:
        print(f"Ground truth level: {LEVEL_NAMES[true_label]}")


if __name__ == "__main__":
    main()
