#!/usr/bin/env bash
set -eo pipefail
source /home/htw/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab-pace
cd "$(dirname "$(readlink -f "$0")")/../.."
export PYTHONUNBUFFERED=1
exec python scripts/np3o/train.py \
  --task DDT-Velocity-Flat-Andy-Pace-Fixed035-v0 \
  --num_envs "${1:-4096}" --max_iterations "${2:-20000}" \
  --device "${3:-cuda:0}" --headless
