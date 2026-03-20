#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/create_demo_data.py --output-dir data/demo
python scripts/create_ttedu_example.py --output-dir data/TTEdu_example

echo "Environment ready."
echo "Demo data generated under data/demo."
echo "TTEdu example data generated under data/TTEdu_example."
echo "Train with: ./.venv/bin/python scripts/train.py --config configs/ttedu_default.json"
