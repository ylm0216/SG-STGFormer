#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Environment not found. Run ./setup_env.sh first."
  exit 1
fi

.venv/bin/python scripts/create_demo_data.py --output-dir data/demo
.venv/bin/python scripts/train.py --config configs/ttedu_default.json --epochs 5
