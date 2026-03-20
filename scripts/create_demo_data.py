from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "thorax",
    "upper_neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]

ACTION_NAMES = ["forehand_drive", "backhand_push", "forehand_loop"]


def base_skeleton() -> np.ndarray:
    return np.array(
        [
            [0.62, 0.95],
            [0.60, 0.80],
            [0.58, 0.62],
            [0.42, 0.62],
            [0.40, 0.80],
            [0.38, 0.95],
            [0.50, 0.60],
            [0.50, 0.42],
            [0.50, 0.28],
            [0.50, 0.15],
            [0.72, 0.48],
            [0.66, 0.42],
            [0.58, 0.38],
            [0.42, 0.38],
            [0.34, 0.44],
            [0.28, 0.50],
        ],
        dtype=np.float32,
    )


def score_to_label(score: float) -> int:
    if score < 2.5:
        return 0
    if score < 5.0:
        return 1
    if score < 7.5:
        return 2
    return 3


def smooth_peak(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-((phase - center) ** 2) / (2 * width**2))


def generate_sample(rng: np.random.Generator, sequence_length: int = 81) -> tuple[np.ndarray, float, int, int]:
    skeleton = base_skeleton()
    phase = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    action_id = int(rng.integers(0, len(ACTION_NAMES)))
    action_name = ACTION_NAMES[action_id]

    raw_score = np.clip(rng.normal(loc=6.0, scale=2.4), 0.0, 10.0)
    label = score_to_label(float(raw_score))
    quality_factor = 0.35 + 0.18 * label
    jitter_scale = 0.018 - 0.0035 * label

    prep = smooth_peak(phase, 0.25, 0.10)
    strike = smooth_peak(phase, 0.55, 0.11)
    follow = smooth_peak(phase, 0.77, 0.12)

    sequence = np.repeat(skeleton[None, :, :], sequence_length, axis=0)
    confidence = np.full((sequence_length, skeleton.shape[0], 1), 0.95, dtype=np.float32)

    torso_shift = 0.025 * quality_factor * (prep + strike)
    lower_drive = 0.015 * quality_factor * prep
    recovery = 0.015 * quality_factor * follow

    sequence[:, 6, 0] += lower_drive - recovery * 0.4
    sequence[:, 7, 0] += torso_shift
    sequence[:, 8, 0] += torso_shift * 0.6
    sequence[:, 9, 0] += torso_shift * 0.4
    sequence[:, 0, 1] -= lower_drive * 0.5
    sequence[:, 5, 1] -= lower_drive * 0.5

    if action_name == "forehand_drive":
        sequence[:, 10, 0] += 0.18 * quality_factor * strike
        sequence[:, 10, 1] -= 0.08 * quality_factor * strike
        sequence[:, 11, 0] += 0.12 * quality_factor * strike
        sequence[:, 12, 0] += 0.06 * quality_factor * strike
    elif action_name == "backhand_push":
        sequence[:, 10, 0] -= 0.11 * quality_factor * strike
        sequence[:, 10, 1] -= 0.03 * quality_factor * strike
        sequence[:, 11, 0] -= 0.08 * quality_factor * strike
        sequence[:, 13, 0] += 0.04 * quality_factor * prep
    else:
        sequence[:, 10, 0] += 0.16 * quality_factor * strike + 0.03 * follow
        sequence[:, 10, 1] -= 0.12 * quality_factor * strike
        sequence[:, 11, 1] -= 0.05 * quality_factor * strike
        sequence[:, 12, 0] += 0.05 * quality_factor * prep
        sequence[:, 0, 1] -= 0.01 * strike
        sequence[:, 5, 1] -= 0.01 * strike

    rhythm_noise = rng.normal(0.0, jitter_scale, size=sequence[:, :, :2].shape).astype(np.float32)
    torso_noise = rng.normal(0.0, jitter_scale * 1.6, size=(sequence_length, 4, 2)).astype(np.float32)
    sequence[:, :, :2] += rhythm_noise
    sequence[:, 6:10, :2] += torso_noise

    confidence -= np.abs(rng.normal(0.0, 0.015 + (3 - label) * 0.01, size=confidence.shape)).astype(np.float32)
    confidence = np.clip(confidence, 0.55, 0.99)

    features = np.concatenate([sequence, confidence], axis=-1).astype(np.float32)
    return features, float(raw_score), label, action_id


def save_split(
    output_path: Path,
    count: int,
    rng: np.random.Generator,
    *,
    sequence_length: int,
    split_name: str,
) -> None:
    samples = []
    scores = []
    labels = []
    actions = []
    sample_ids = []

    for index in range(count):
        x, score, label, action_id = generate_sample(rng, sequence_length=sequence_length)
        samples.append(x)
        scores.append(score)
        labels.append(label)
        actions.append(action_id)
        sample_ids.append(f"{split_name}_{index:04d}")

    np.savez_compressed(
        output_path,
        x=np.stack(samples).astype(np.float32),
        score=np.array(scores, dtype=np.float32),
        label=np.array(labels, dtype=np.int64),
        action_id=np.array(actions, dtype=np.int64),
        sample_id=np.array(sample_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic demo dataset for SG-STGFormer.")
    parser.add_argument("--output-dir", default="data/demo", help="Directory for generated NPZ files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=81)
    parser.add_argument("--train-size", type=int, default=240)
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    save_split(output_dir / "train.npz", args.train_size, rng, sequence_length=args.sequence_length, split_name="train")
    save_split(output_dir / "val.npz", args.val_size, rng, sequence_length=args.sequence_length, split_name="val")
    save_split(output_dir / "test.npz", args.test_size, rng, sequence_length=args.sequence_length, split_name="test")

    metadata = {
        "joint_names": JOINT_NAMES,
        "action_names": ACTION_NAMES,
        "label_names": ["poor", "moderate", "good", "excellent"],
        "sequence_length": args.sequence_length,
        "note": "Synthetic demonstration data generated to verify the code path. Replace with TTEdu skeleton sequences for paper experiments.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Demo dataset written to {output_dir}")


if __name__ == "__main__":
    main()
