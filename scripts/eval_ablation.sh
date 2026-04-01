#!/usr/bin/env bash
# =============================================================================
# scripts/eval_ablation.sh  —  Evaluate all ablation variants
#
# Runs eval_ablation.py for every variant and saves:
#   experiments/ablation/eval_results/
#     <variant_id>/
#       celeb_df_preds.csv
#       dfd_preds.csv
#       wild_deepfake_preds.csv
#       diffswap_preds.csv
#       ffpp_rela_preds.csv
#       metrics.json
#     ablation_table.csv          ← AUC% / EER% per variant per dataset
#     ablation_table_compact.csv  ← compact version for paper
#     ablation_metrics.json       ← raw numbers
#
# Usage:
#   bash scripts/eval_ablation.sh                        # all variants, GPU 0
#   GPU=1 bash scripts/eval_ablation.sh                  # different GPU
#   VARIANTS="a b c" bash scripts/eval_ablation.sh       # specific variants only
#   VARIANTS="full_model" GPU=0 bash scripts/eval_ablation.sh
#   USE_ALL_FRAMES=1 bash scripts/eval_ablation.sh       # all DFD frames
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Configuration ─────────────────────────────────────────────────────────────
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/ablation/eval_ablation.yaml}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_ALL_FRAMES="${USE_ALL_FRAMES:-1}"   # 1 = use all DFD frames, 0 = respect config
VARIANTS="${VARIANTS:-}"               # space-separated list, empty = all

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "════════════════════════════════════════════════════════════════"
echo " DiffusionFake Ablation Study Evaluation"
echo "════════════════════════════════════════════════════════════════"
echo " Config      : $CONFIG"
echo " GPU         : $GPU"
echo " Batch size  : $BATCH_SIZE"
echo " Num workers : $NUM_WORKERS"
echo " All frames  : $USE_ALL_FRAMES"
if [[ -n "$VARIANTS" ]]; then
  echo " Variants    : $VARIANTS"
else
  echo " Variants    : ALL"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
  python -u eval_ablation.py
  -c            "$CONFIG"
  --batch_size  "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
)

if [[ "$USE_ALL_FRAMES" == "1" ]]; then
  CMD+=(--use_all_frames)
fi

if [[ -n "$VARIANTS" ]]; then
  # shellcheck disable=SC2206
  CMD+=(--variants $VARIANTS)
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
if [[ "$EXIT_CODE" -eq 0 ]]; then
  echo " Ablation evaluation completed successfully."
  echo ""
  echo " Results:"
  echo "   experiments/ablation/eval_results/ablation_table.csv"
  echo "   experiments/ablation/eval_results/ablation_table_compact.csv"
  echo "   experiments/ablation/eval_results/ablation_metrics.json"
  echo "   experiments/ablation/eval_results/<variant_id>/  (per-variant CSVs)"
else
  echo " [ERROR] Evaluation exited with code $EXIT_CODE"
fi
echo "════════════════════════════════════════════════════════════════"

exit $EXIT_CODE

# ──────────────────────────────────────────────────────────────────────────────
# Quick reference:
#
#   # All variants, GPU 0:
#   bash scripts/eval_ablation.sh
#
#   # Run only variants a, b, c:
#   VARIANTS="a_backbone_linear b_diffusion_guidance c_dual_upsample_ff" \
#     bash scripts/eval_ablation.sh
#
#   # Run full model only on GPU 2:
#   GPU=2 VARIANTS="full_model" bash scripts/eval_ablation.sh
#
#   # Run design-choice ablations (g, h, full):
#   VARIANTS="g_no_dual_dropout h_no_sbi full_model" \
#     bash scripts/eval_ablation.sh
#
#   # Multi-GPU note: eval runs sequentially (one model at a time per GPU).
#   # To parallelize, run separate instances on different GPUs:
#   GPU=0 VARIANTS="a_backbone_linear b_diffusion_guidance c_dual_upsample_ff" \
#     bash scripts/eval_ablation.sh &
#   GPU=1 VARIANTS="d_global_head_smc e_dimf f_igs_igl_gap" \
#     bash scripts/eval_ablation.sh &
#   GPU=2 VARIANTS="g_no_dual_dropout h_no_sbi full_model" \
#     bash scripts/eval_ablation.sh &
#   wait
# ──────────────────────────────────────────────────────────────────────────────

# Example parallel execution:
# GPU=0 VARIANTS="a_backbone_linear b_diffusion_guidance c_dual_upsample_ff" bash scripts/eval_ablation.sh > logs_eval_gpu0.out 2>&1 &
# GPU=1 VARIANTS="d_global_head_smc e_dimf f_igs_igl_gap" bash scripts/eval_ablation.sh > logs_eval_gpu1.out 2>&1 &
# GPU=2 VARIANTS="g_no_dual_dropout h_no_sbi full_model" bash scripts/eval_ablation.sh > logs_eval_gpu2.out 2>&1 &
# wait

# ──────────────────────────────────────────────────────────────────────────────
# Note: If you want to run all variants sequentially on a single GPU, simply run:
# GPU=2 VARIANTS="a_backbone_linear b_diffusion_guidance c_dual_upsample_ff" bash scripts/eval_ablation.sh
# GPU=3 VARIANTS="d_global_head_smc e_dimf f_igs_igl_gap" bash scripts/eval_ablation.sh
# GPU=7 VARIANTS="g_no_dual_dropout h_no_sbi full_model" bash scripts/eval_ablation.sh
# ──────────────────────────────────────────────────────────────────────────────