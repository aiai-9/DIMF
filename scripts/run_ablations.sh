# code/ImageDifussionFake/scripts/ablation.sh

#!/usr/bin/env bash
# =============================================================================
# scripts/run_ablations.sh  —  Run all ablation studies sequentially
#
# Configs live in:  configs/ablation/ablation_*.yaml
# Checkpoints in:   experiments/ablation/<variant>/ckpt/
# WandB project:    DFD_ablation   group: ablation_study
#
# Usage:
#   bash scripts/run_ablations.sh                      # all variants, GPU 0
#   bash scripts/run_ablations.sh 5,1,3              # all variants, 4 GPUs
#   bash scripts/run_ablations.sh 0 a b c              # only variants a, b, c
#   bash scripts/run_ablations.sh 0 full               # only Full model
#   bash scripts/run_ablations.sh 0 g h full           # design-choice ablations
# bash scripts/run_ablations.sh 5,1,3
# =============================================================================


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

GPUS="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l | xargs)

REQUESTED=("${@:2}")

declare -A CONFIGS
CONFIGS["a"]="configs/ablation/ablation_a_backbone_linear.yaml"
CONFIGS["b"]="configs/ablation/ablation_b_diffusion_guidance.yaml"
CONFIGS["c"]="configs/ablation/ablation_c_dual_upsample_ff.yaml"
CONFIGS["d"]="configs/ablation/ablation_d_global_head_smc.yaml"
CONFIGS["e"]="configs/ablation/ablation_e_dimf.yaml"
CONFIGS["f"]="configs/ablation/ablation_f_igs.yaml"
CONFIGS["g"]="configs/ablation/ablation_g_no_dual_dropout.yaml"
CONFIGS["h"]="configs/ablation/ablation_h_no_sbi.yaml"
CONFIGS["full"]="configs/ablation/ablation_full_model.yaml"

ORDER=("a" "b" "c" "d" "e" "f" "g" "h" "full")

if [ ${#REQUESTED[@]} -eq 0 ]; then
    TO_RUN=("${ORDER[@]}")
else
    TO_RUN=("${REQUESTED[@]}")
fi

run_variant() {
    local id="$1"
    local cfg="${CONFIGS[$id]:-}"

    if [ -z "$cfg" ]; then
        echo "[SKIP] Unknown variant: '$id'. Valid: ${!CONFIGS[*]}"
        return 0
    fi
    if [ ! -f "$cfg" ]; then
        echo "[SKIP] Config not found: $cfg"
        return 0
    fi

    MASTER_PORT=$((20000 + RANDOM % 20000))

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Ablation variant : ($id)"
    echo " Config           : $cfg"
    echo " GPUs             : $GPUS  (nproc=$NUM_GPUS)"
    echo " MASTER_PORT      : $MASTER_PORT"
    echo " Start time       : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_port="$MASTER_PORT" \
            train.py -c "$cfg"
    else
        python train.py -c "$cfg"
    fi

    echo "[DONE] Variant ($id) finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

echo ""
echo "Ablation runner starting"
echo "  Variants to run : ${TO_RUN[*]}"
echo "  GPUs            : $GPUS"
echo ""

FAILED=()
for variant in "${TO_RUN[@]}"; do
    run_variant "$variant" || FAILED+=("$variant")
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All ablation variants completed."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED variants: ${FAILED[*]}"
else
    echo "All variants succeeded."
fi
echo "Results in: experiments/ablation/<variant>/ckpt/"
echo "════════════════════════════════════════════════════════════"