#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_gircse.yaml}"

python scripts/train_skeleton_gircse.py \
  --config "${CONFIG}" \
  --wandb_mode offline
