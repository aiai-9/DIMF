# code/ImageDifussionFake/scripts/eval.sh
set -euo pipefail

cd /scratch/sahil/projects/img_deepfake/code/ImageDifussionFake

EXP_DIR="${EXP_DIR:-experiments/FFPP_10}"
CKPT_DIR="${CKPT_DIR:-$EXP_DIR/ckpt}"
CONFIG="${CONFIG:-configs/train.yaml}"
DATASET="${DATASET:-celeb_df}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# --- choose ckpt: best-eer-epoch=*.ckpt with max epoch, else last.ckpt ---
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


OUT_CSV="${OUT_CSV:-$EXP_DIR/eval/celeb_train/celebdf_v2_test_preds.csv}"
mkdir -p "$(dirname "$OUT_CSV")"

echo "[EVAL] EXP_DIR=$EXP_DIR"
echo "[EVAL] CKPT=$CKPT"
echo "[EVAL] CONFIG=$CONFIG DATASET=$DATASET SPLIT=$SPLIT BS=$BATCH_SIZE NW=$NUM_WORKERS"
echo "[EVAL] OUT_CSV=$OUT_CSV"

python -u eval_2.py \
  -c "$CONFIG" \
  --ckpt "$CKPT" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --out_csv "$OUT_CSV"

# chmod +x code/ImageDifussionFake/scripts/eval.sh
# chmod +x scripts/eval.sh
# bash scripts/eval.sh

# python utils/debug_dump_frames.py