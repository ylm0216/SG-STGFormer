from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from create_demo_data import ACTION_NAMES, JOINT_NAMES, base_skeleton, score_to_label, smooth_peak


SKILL_LEVELS = ["beginner", "intermediate", "advanced"]
GENDERS = ["male", "female"]

ACTION_COUNTS = {
    "forehand_drive": 280,
    "backhand_push": 240,
    "forehand_loop": 200,
}

SKILL_COUNTS = {
    "beginner": 288,
    "intermediate": 252,
    "advanced": 180,
}

GENDER_COUNTS = {
    "male": 396,
    "female": 324,
}


def build_exact_list(name_to_count: dict[str, int]) -> list[str]:
    values: list[str] = []
    for name, count in name_to_count.items():
        values.extend([name] * count)
    return values


def allocate_counts(group_sizes: list[int], target_total: int, factor: float) -> list[int]:
    exact = [size * factor for size in group_sizes]
    base = [int(np.floor(value)) for value in exact]
    remainder = target_total - sum(base)
    fractions = [value - floor for value, floor in zip(exact, base)]
    for idx in np.argsort(fractions)[::-1][:remainder]:
        base[int(idx)] += 1
    return base


def generate_sequence(
    rng: np.random.Generator,
    *,
    action_name: str,
    final_score: float,
    sequence_length: int = 81,
) -> np.ndarray:
    skeleton = base_skeleton()
    phase = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    prep = smooth_peak(phase, 0.25, 0.10)
    strike = smooth_peak(phase, 0.55, 0.11)
    follow = smooth_peak(phase, 0.77, 0.12)

    score_factor = np.clip(final_score / 10.0, 0.0, 1.0)
    quality_factor = 0.28 + 0.70 * score_factor
    jitter_scale = np.clip(0.030 - 0.020 * score_factor, 0.006, 0.030)

    sequence = np.repeat(skeleton[None, :, :], sequence_length, axis=0)
    confidence = np.full((sequence_length, skeleton.shape[0], 1), 0.93, dtype=np.float32)

    torso_shift = 0.030 * quality_factor * (prep + strike)
    lower_drive = 0.016 * quality_factor * prep
    recovery = 0.015 * quality_factor * follow

    sequence[:, 6, 0] += lower_drive - recovery * 0.35
    sequence[:, 7, 0] += torso_shift
    sequence[:, 8, 0] += torso_shift * 0.65
    sequence[:, 9, 0] += torso_shift * 0.50
    sequence[:, 0, 1] -= lower_drive * 0.55
    sequence[:, 5, 1] -= lower_drive * 0.55

    if action_name == "forehand_drive":
        sequence[:, 10, 0] += 0.18 * quality_factor * strike
        sequence[:, 10, 1] -= 0.08 * quality_factor * strike
        sequence[:, 11, 0] += 0.12 * quality_factor * strike
        sequence[:, 12, 0] += 0.07 * quality_factor * strike
    elif action_name == "backhand_push":
        sequence[:, 10, 0] -= 0.12 * quality_factor * strike
        sequence[:, 10, 1] -= 0.03 * quality_factor * strike
        sequence[:, 11, 0] -= 0.08 * quality_factor * strike
        sequence[:, 13, 0] += 0.05 * quality_factor * prep
    else:
        sequence[:, 10, 0] += 0.17 * quality_factor * strike + 0.03 * follow
        sequence[:, 10, 1] -= 0.13 * quality_factor * strike
        sequence[:, 11, 1] -= 0.05 * quality_factor * strike
        sequence[:, 12, 0] += 0.05 * quality_factor * prep
        sequence[:, 0, 1] -= 0.012 * strike
        sequence[:, 5, 1] -= 0.012 * strike

    rhythm_noise = rng.normal(0.0, jitter_scale, size=sequence[:, :, :2].shape).astype(np.float32)
    torso_noise = rng.normal(0.0, jitter_scale * 1.5, size=(sequence_length, 4, 2)).astype(np.float32)
    sequence[:, :, :2] += rhythm_noise
    sequence[:, 6:10, :2] += torso_noise

    confidence_noise = np.abs(rng.normal(0.0, 0.020 + (1.0 - score_factor) * 0.030, size=confidence.shape)).astype(np.float32)
    confidence -= confidence_noise
    confidence = np.clip(confidence, 0.50, 0.99)

    return np.concatenate([sequence, confidence], axis=-1).astype(np.float32)


def generate_coach_scores(rng: np.random.Generator, final_score: float) -> tuple[float, float, float | None]:
    score_1 = float(np.clip(final_score + rng.normal(0.0, 0.55), 0.0, 10.0))
    score_2 = float(np.clip(final_score + rng.normal(0.0, 0.55), 0.0, 10.0))
    if abs(score_1 - score_2) > 2.0:
        score_3 = float(np.clip(final_score + rng.normal(0.0, 0.45), 0.0, 10.0))
        return score_1, score_2, score_3
    return score_1, score_2, None


def simulate_samples(rng: np.random.Generator, sequence_length: int) -> list[dict]:
    actions = build_exact_list(ACTION_COUNTS)
    skills = build_exact_list(SKILL_COUNTS)
    genders = build_exact_list(GENDER_COUNTS)

    rng.shuffle(actions)
    rng.shuffle(skills)
    rng.shuffle(genders)

    samples: list[dict] = []
    skill_mean = {
        "beginner": 3.4,
        "intermediate": 5.8,
        "advanced": 7.9,
    }
    action_bias = {
        "forehand_drive": 0.10,
        "backhand_push": -0.10,
        "forehand_loop": 0.15,
    }

    for index in range(720):
        action_name = actions[index]
        skill_level = skills[index]
        gender = genders[index]

        raw_score = float(
            np.clip(
                rng.normal(skill_mean[skill_level] + action_bias[action_name], 1.05),
                0.0,
                10.0,
            )
        )
        label = score_to_label(raw_score)
        coach_1, coach_2, coach_3 = generate_coach_scores(rng, raw_score)
        sequence = generate_sequence(
            rng,
            action_name=action_name,
            final_score=raw_score,
            sequence_length=sequence_length,
        )

        samples.append(
            {
                "sample_id": f"ttedu_{index:04d}",
                "x": sequence,
                "score": raw_score,
                "label": label,
                "action_name": action_name,
                "action_id": ACTION_NAMES.index(action_name),
                "skill_level": skill_level,
                "skill_id": SKILL_LEVELS.index(skill_level),
                "gender": gender,
                "gender_id": GENDERS.index(gender),
                "coach_score_1": coach_1,
                "coach_score_2": coach_2,
                "coach_score_3": -1.0 if coach_3 is None else coach_3,
            }
        )
    return samples


def stratified_split(samples: list[dict], rng: np.random.Generator) -> dict[str, list[dict]]:
    strata: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for sample in samples:
        strata[(sample["action_id"], sample["skill_id"])].append(sample)

    keys = sorted(strata.keys())
    group_sizes = [len(strata[key]) for key in keys]
    train_counts = allocate_counts(group_sizes, 504, 0.70)
    val_counts = allocate_counts(group_sizes, 108, 0.15)

    split_samples = {"train": [], "val": [], "test": []}
    for key, train_count, val_count in zip(keys, train_counts, val_counts):
        group = strata[key]
        rng.shuffle(group)
        train_slice = train_count
        val_slice = train_count + val_count
        split_samples["train"].extend(group[:train_slice])
        split_samples["val"].extend(group[train_slice:val_slice])
        split_samples["test"].extend(group[val_slice:])

    for split_name in split_samples:
        rng.shuffle(split_samples[split_name])
    return split_samples


def save_split(output_path: Path, samples: list[dict]) -> None:
    np.savez_compressed(
        output_path,
        x=np.stack([sample["x"] for sample in samples]).astype(np.float32),
        score=np.array([sample["score"] for sample in samples], dtype=np.float32),
        label=np.array([sample["label"] for sample in samples], dtype=np.int64),
        action_id=np.array([sample["action_id"] for sample in samples], dtype=np.int64),
        skill_id=np.array([sample["skill_id"] for sample in samples], dtype=np.int64),
        gender_id=np.array([sample["gender_id"] for sample in samples], dtype=np.int64),
        coach_score_1=np.array([sample["coach_score_1"] for sample in samples], dtype=np.float32),
        coach_score_2=np.array([sample["coach_score_2"] for sample in samples], dtype=np.float32),
        coach_score_3=np.array([sample["coach_score_3"] for sample in samples], dtype=np.float32),
        sample_id=np.array([sample["sample_id"] for sample in samples]),
        action_name=np.array([sample["action_name"] for sample in samples]),
        skill_level=np.array([sample["skill_level"] for sample in samples]),
        gender=np.array([sample["gender"] for sample in samples]),
    )


def summarize_counts(samples: list[dict], field: str) -> dict[str, int]:
    summary: dict[str, int] = defaultdict(int)
    for sample in samples:
        summary[str(sample[field])] += 1
    return dict(sorted(summary.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a TTEdu-style example dataset.")
    parser.add_argument("--output-dir", default="data/TTEdu_example")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=81)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    samples = simulate_samples(rng, sequence_length=args.sequence_length)
    split_samples = stratified_split(samples, rng)
    for split_name, split_list in split_samples.items():
        save_split(output_dir / f"{split_name}.npz", split_list)

    metadata = {
        "dataset_name": "TTEdu_example",
        "note": "This is a generated example dataset that mirrors the manuscript-level TTEdu structure. It is not the real self-built dataset and must not be reported as real experimental data.",
        "total_samples": 720,
        "split_sizes": {name: len(split_list) for name, split_list in split_samples.items()},
        "action_counts_target": ACTION_COUNTS,
        "skill_counts_target": SKILL_COUNTS,
        "gender_counts_target": GENDER_COUNTS,
        "joint_names": JOINT_NAMES,
        "action_names": ACTION_NAMES,
        "skill_levels": SKILL_LEVELS,
        "genders": GENDERS,
        "sequence_length": args.sequence_length,
        "feature_channels": ["x", "y", "confidence"],
        "observed_overall_action_counts": summarize_counts(samples, "action_name"),
        "observed_overall_skill_counts": summarize_counts(samples, "skill_level"),
        "observed_overall_gender_counts": summarize_counts(samples, "gender"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"TTEdu example dataset written to {output_dir}")
    print(json.dumps(metadata["split_sizes"], indent=2))


if __name__ == "__main__":
    main()
