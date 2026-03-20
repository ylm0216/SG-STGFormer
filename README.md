# SG-STGFormer Code Package

This folder is a paper-aligned code package for the manuscript:

`A Skeleton-Guided Spatiotemporal Graph Transformer for Automated Stroke Quality Assessment in College-Level Table Tennis Education`

It includes:

- an implementation of the SG-STGFormer architecture in PyTorch
- adaptive hybrid graph learning
- spatial graph Transformer blocks
- causal temporal Transformer blocks
- cross-attention spatiotemporal fusion
- multi-task regression + classification + supervised contrastive learning
- a synthetic demo dataset so the pipeline can be exercised before plugging in the real TTEdu data
- a generated `TTEdu_example` dataset that mirrors the manuscript-level dataset scale and split

## Folder Layout

- `sg_stgformer/`: model, loss, metrics, graph utilities, data loader, train/eval engine
- `scripts/create_demo_data.py`: generate synthetic skeleton sequences for a full dry run
- `scripts/create_ttedu_example.py`: generate a manuscript-shaped TTEdu example dataset
- `scripts/train.py`: training entry point
- `scripts/evaluate.py`: split evaluation for a saved checkpoint
- `scripts/predict.py`: single-sample inference
- `configs/ttedu_default.json`: paper-style default settings
- `configs/ttedu_example.json`: training config for the TTEdu example split
- `MANUSCRIPT_SNIPPET.md`: ready-to-paste wording for code availability and reviewer response

## Quick Start

1. Create the environment and install dependencies:

```bash
./setup_env.sh
```

2. Run a short demo training:

```bash
./run_demo.sh
```

3. Run a full training session:

```bash
./.venv/bin/python scripts/train.py --config configs/ttedu_default.json
```

To train on the TTEdu example package:

```bash
./.venv/bin/python scripts/train.py --config configs/ttedu_example.json
```

4. Evaluate a checkpoint:

```bash
./.venv/bin/python scripts/evaluate.py --checkpoint runs/<run_id>/best.pt --split test
```

5. Predict one sample:

```bash
./.venv/bin/python scripts/predict.py --checkpoint runs/<run_id>/best.pt --npz data/demo/test.npz --index 0
```

## Expected Data Format

Each split is stored as an `.npz` file with:

- `x`: shape `[N, T, V, C]`, where `T=81`, `V=16`, `C=3`
- `score`: continuous score in `[0, 10]`
- `label`: discrete quality label in `{0, 1, 2, 3}`
- optional: `sample_id`, `action_id`

The code assumes the feature channels are:

- channel `0`: x coordinate
- channel `1`: y coordinate
- channel `2`: keypoint confidence

## Paper-Aligned Defaults

The following values were recovered directly from the manuscript text:

- `16` keypoints per sample
- `81` frames per sample
- `4` spatial graph Transformer layers
- `3` temporal Transformer layers
- `8` attention heads
- hidden dimension `256`
- batch size `32`
- training epochs `100`
- AdamW learning rate `1e-4`
- weight decay `1e-2`
- loss weights `lambda_1=1.0`, `lambda_2=0.5`, `lambda_3=0.3`
- temperature `tau=0.07`
- hybrid graph coefficient initialized to `0.5`
- spatial gating coefficient initialized to `0.5`

## Important Notes

- The included `data/demo/` files are synthetic and only meant to validate the code path.
- The included `data/TTEdu_example/` files are generated example data and are only for structure validation and internal testing.
- Replace the demo data with your real TTEdu skeleton sequences before reporting paper results.
- The manuscript states that the skeletons were extracted with `HRNet-W48`; this repository starts from those extracted skeleton sequences, not from raw RGB videos.
- The 16-joint order is configurable in `configs/ttedu_default.json`. If your exported keypoint order differs, update `joint_names` and `physical_edges`.

## Suggested Release Checklist

- replace synthetic demo data with anonymized TTEdu skeleton files
- confirm the final joint order matches your pose-export pipeline
- rerun training on the real train/val/test split
- update the GitHub URL in `MANUSCRIPT_SNIPPET.md`
- upload `runs/<best_run>/best.pt` only if journal policy allows weight sharing
