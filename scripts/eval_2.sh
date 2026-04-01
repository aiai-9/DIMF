# scripts/eval_2.sh


#!/usr/bin/env bash
set -euo pipefail

cd /scratch/sahil/projects/img_deepfake/code/ImageDifussionFake

# ======================================================================================

# fp_celeb_wild_diffswap_2
# EXP_DIR="${EXP_DIR:-experiments/fp_celeb_wild_diffswap_2}"
EXP_DIR="${EXP_DIR:-experiments/FFpp_Celeb_Wild_2}"
# EXP_DIR="${EXP_DIR:-experiments/FFpp_v2}"
# EXP_DIR="${EXP_DIR:-experiments/FFpp_CelebDF_v6}"

# EXP_DIR="${EXP_DIR:-experiments/FFpp_CelebDF_v4}"
# EXP_DIR="${EXP_DIR:-experiments/AllDomains_1}"

# ======================================================================================

CKPT_DIR="${CKPT_DIR:-$EXP_DIR/ckpt}"
CONFIG="${CONFIG:-configs/eval_all_datasets.yaml}"
SPLIT="${SPLIT:-test}"
GPU="${GPU:-3,2}"
TOP_K="${TOP_K:-10}"
VIDEO_SCORE_MODE="${VIDEO_SCORE_MODE:-topk}"   # topk | mean | median
 
# FIX-S1: use all frames for DFD (overrides eval_k_frames_per_video in config)
USE_ALL_FRAMES="${USE_ALL_FRAMES:-1}"
 
# FIX-S2: TTA (horizontal flip averaging, off by default — adds ~40% eval time)
USE_TTA="${USE_TTA:-0}"
 
DEBUG="${DEBUG:-0}"
 
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
 
# ── Find best checkpoint ────────────────────────────────────────────────────
BEST_CKPT="$(ls -1 "$CKPT_DIR"/best-eer-epoch=*\.ckpt 2>/dev/null | \
  sed -E 's/.*best-eer-epoch=([0-9]+)\.ckpt/\1 &/' | \
  sort -n | tail -1 | awk '{print $2}' || true)"
 
if [[ -n "${BEST_CKPT:-}" ]]; then
  CKPT="$BEST_CKPT"
elif [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  CKPT="$CKPT_DIR/last.ckpt"
else
  echo "[ERR] No checkpoint found in: $CKPT_DIR"
  exit 1
fi
 
# ── Output directories ──────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$EXP_DIR/eval_2}"
mkdir -p "$OUT_DIR"
 
OUT_FRAME_CSV="${OUT_FRAME_CSV:-$OUT_DIR/frame_predictions_${SPLIT}.csv}"
OUT_VIDEO_CSV="${OUT_VIDEO_CSV:-$OUT_DIR/video_predictions_${SPLIT}.csv}"
OUT_METRICS="${OUT_METRICS:-$OUT_DIR/generalization_all_${SPLIT}_metrics.json}"
OUT_TABLE_CSV="${OUT_TABLE_CSV:-$OUT_DIR/generalization_table_${SPLIT}.csv}"
OUT_SUMMARY_CSV="${OUT_SUMMARY_CSV:-$OUT_DIR/summary_table_${SPLIT}.csv}"
 
echo "[EVAL-ALL] CKPT=$CKPT"
echo "[EVAL-ALL] CONFIG=$CONFIG  SPLIT=$SPLIT"
echo "[EVAL-ALL] GPU=$GPU"
echo "[EVAL-ALL] USE_ALL_FRAMES=$USE_ALL_FRAMES  USE_TTA=$USE_TTA"
echo "[EVAL-ALL] OUT_DIR=$OUT_DIR"
echo "[EVAL-ALL] TOP_K=$TOP_K  VIDEO_SCORE_MODE=$VIDEO_SCORE_MODE"
echo "[EVAL-ALL] DEBUG=$DEBUG"
 
# ── Build command ───────────────────────────────────────────────────────────
CMD=(
  python -u eval_generalization.py
  -c              "$CONFIG"
  --ckpt          "$CKPT"
  --out_frame_csv   "$OUT_FRAME_CSV"
  --out_video_csv   "$OUT_VIDEO_CSV"
  --out_table_csv   "$OUT_TABLE_CSV"
  --out_summary_csv "$OUT_SUMMARY_CSV"   # FIX-S3: was missing
  --out_metrics_json "$OUT_METRICS"
  --top_k           "$TOP_K"
  --video_score_mode "$VIDEO_SCORE_MODE"
)
 
# FIX-S1: pass --use_all_frames when enabled
if [[ "$USE_ALL_FRAMES" == "1" ]]; then
  CMD+=(--use_all_frames)
  echo "[EVAL-ALL] NOTE: --use_all_frames active — DFD will load all frames (not just 1/video)"
fi
 
# FIX-S2: pass --use_tta when enabled
if [[ "$USE_TTA" == "1" ]]; then
  CMD+=(--use_tta)
  echo "[EVAL-ALL] NOTE: --use_tta active — horizontal-flip TTA enabled"
fi
 
if [[ "$DEBUG" == "1" ]]; then
  CMD+=(--debug)
fi
 
"${CMD[@]}"
 
echo ""
echo "=============================="
echo "Evaluation complete"
# echo "Frame CSV:   $OUT_FRAME_CSV"
# echo "Video CSV:   $OUT_VIDEO_CSV"
# echo "Full table:  $OUT_TABLE_CSV"
# echo "Summary CSV: $OUT_SUMMARY_CSV"
# echo "Metrics JSON: $OUT_METRICS"
echo "=============================="

# chmod +x scripts/eval_2.sh
# bash scripts/eval_2.sh
# GPU=3,4 bash scripts/eval_2.sh
# GPU=3 DEBUG=1 bash scripts/eval_2.sh
# GPU=2 bash scripts/eval_2.sh
