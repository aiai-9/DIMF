# cscripts/train_mixed.sh

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

GPUS="${1:-0,1}"
CONFIG="${2:-configs/train_mixed_robust.yaml}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l | xargs)
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

echo "=== Mixed Robust Training ==="
echo "Config: $CONFIG"
echo "GPUs: $GPUS ($NUM_GPUS processes)"
echo "MASTER_PORT: $MASTER_PORT"

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  train_mixed.py -c "$CONFIG"




# bash scripts/train_mixed.sh 1,5
# bash scripts/train_mixed.sh 0,1,7,5