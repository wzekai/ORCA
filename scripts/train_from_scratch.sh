#!/usr/bin/env bash
# Train an ORCA TTT-Probe from scratch on the released training corpora.
# Default: Qwen2.5-32B, no-QK linear variant, supervised label mode.

set -euo pipefail
DATA=./data
RESULTS=./results

LLM=qwen2.5-32b               # qwen2.5-32b | qwq-32b | llama-3.3-70b
LABEL=supervised              # supervised | consistent
VARIANT=no_kq                 # no_kq | qk
EPOCHS=20                     # paper: 20 for no_kq, 10 for qk
DH=128                        # only used for qk
SEED=42                       # random seed (matches paper)

case "$LLM" in
    qwen2.5-32b)   CFG=configs/qwen32b.yaml ;;
    qwq-32b)       CFG=configs/qwq32b.yaml ;;
    llama-3.3-70b) CFG=configs/llama70b.yaml ;;
esac

OUT_DIR=$RESULTS/${LLM}__${LABEL}__${VARIANT}
mkdir -p "$OUT_DIR"

EXTRA_FLAGS=""
[ "$VARIANT" = "no_kq" ] && EXTRA_FLAGS="--no_kq"
[ "$VARIANT" = "qk" ]    && EXTRA_FLAGS="--d_hidden $DH"

TRAIN_PATHS=(
    "$DATA/$LLM/s1k.pkl"
    "$DATA/$LLM/openr1_2k.pkl"
    "$DATA/$LLM/deepmath_2k.pkl"
)

# OOD evaluation only available for Qwen2.5-32B in the released datasets.
OOD_ARGS=()
if [ "$LLM" = "qwen2.5-32b" ]; then
    OOD_ARGS=(--ood_paths
        "$DATA/qwen2.5-32b/math500.pkl"
        "$DATA/qwen2.5-32b/gpqa_diamond.pkl"
        "$DATA/qwen2.5-32b/aime24.pkl"
        "$DATA/qwen2.5-32b/aime25.pkl"
        "$DATA/qwen2.5-32b/aime26.pkl"
    )
fi

python code/train.py \
    --config "$CFG" --method ttt --seed $SEED \
    --dataset_path "${TRAIN_PATHS[@]}" \
    --output_dir "$OUT_DIR" \
    --label_mode "$LABEL" \
    --epochs "$EPOCHS" --save_every 10 \
    $EXTRA_FLAGS

python code/calibrate.py \
    --config "$CFG" --method ttt --seed $SEED \
    --dataset_path "${TRAIN_PATHS[@]}" \
    --output_dir "$OUT_DIR" \
    --label_mode "$LABEL" --epochs "$EPOCHS" \
    --delta 0.05 0.1 0.15 0.2 --epsilon 0.05 \
    $EXTRA_FLAGS

python code/test.py \
    --config "$CFG" --method ttt --seed $SEED \
    --dataset_path "${TRAIN_PATHS[@]}" \
    "${OOD_ARGS[@]}" \
    --output_dir "$OUT_DIR" \
    --label_mode "$LABEL" --epochs "$EPOCHS" \
    --delta 0.05 0.1 0.15 0.2 --epsilon 0.05 \
    $EXTRA_FLAGS

echo "Done. Probe + metrics under $OUT_DIR/"
