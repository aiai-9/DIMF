

#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Move to project root (ImageDifussionFake/)
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --------------------------------------------------
# Usage
#   bash scripts/train.sh
#   bash scripts/train.sh 0
#   bash scripts/train.sh 0,1
#   bash scripts/train.sh 4,5,6,7 configs/train_celebdf.yaml
#   MASTER_PORT=29517 bash scripts/train.sh 2,3 configs/train_mixed.yaml
# --------------------------------------------------

GPUS="${1:-7,3,6,0}"          # <---- configurable GPU list
# ================================================================================

# CONFIG="${2:-configs/train.yaml}"  
# CONFIG="${2:-configs/train_celebdf.yaml}" 
# CONFIG="${2:-configs/train_ffpp_celebdf.yaml}" 
# CONFIG="${2:-configs/train_mixed.yaml}" 

# single-domain configs
# CONFIG="${2:-configs/single/ffpp.yaml}"
CONFIG="${2:-configs/single/celebdf.yaml}"
# CONFIG="${2:-configs/single/dfd.yaml}"
# CONFIG="${2:-configs/single/wild_deepfake.yaml}"
# CONFIG="${2:-configs/single/diffswap.yaml}"

# multi-domain configs
# CONFIG="${2:-configs/multi/ffpp_celebdf.yaml}"
# CONFIG="${2:-configs/multi/ffpp_celebdf_wd.yaml}"
# CONFIG="${2:-configs/multi/ffpp_celebdf_wd_dfd.yaml}" 
# CONFIG="${2:-configs/multi/ffpp_celebdf_wd_dfd_ds.yaml}" 

# ================================================================================
 
# Validate config before launching — catch missing config BEFORE torchrun
if [[ ! -f "$CONFIG" ]]; then
    echo ""
    echo "ERROR: Config not found: $CONFIG"
    echo ""
    echo "Single-domain configs:"
    ls configs/single/*.yaml 2>/dev/null | sed 's/^/  /' || echo "  (run from project root)"
    echo "Multi-domain configs:"
    ls configs/multi/*.yaml 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo ""
    exit 1
fi
 
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
 
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | sed '/^\s*$/d' | wc -l | xargs)
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
 
# NCCL settings — see comments at top for rationale
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
 
echo "===================================================================="
echo "  GPUs        : $GPUS ($NUM_GPUS processes)"
echo "  Config      : $CONFIG"
echo "  Master port : $MASTER_PORT"
echo "  NCCL timeout: ${NCCL_TIMEOUT}s"
echo "  Val         : max_items=6000, bs=64 → ~45s per epoch"
echo "===================================================================="
 
if [[ "$NUM_GPUS" -le 1 ]]; then
    python train.py -c "$CONFIG"
else
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        train.py -c "$CONFIG"
fi

# bash scripts/train.sh
# bash scripts/train.sh 0,1,7,5
# MASTER_PORT=29517 bash scripts/train.sh 2,3
# bash code/ImageDifussionFake/scripts/train.sh 2,3
# bash scripts/train.sh 7,4,6,0
# bash scripts/train.sh 1,5