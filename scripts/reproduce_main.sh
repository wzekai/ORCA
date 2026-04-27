#!/usr/bin/env bash
# Reproduce Qwen2.5-32B paper results: Tables 1, 2, sup_con, dhidden, ablation_arch.
# Per probe: calibrate.py -> lambdas.json, then test.py -> metrics.json.
#
# Prereqs:
#   bash scripts/download_artifacts.sh
#   source .venv/bin/activate

set -euo pipefail
PROBES=./probes
DATA=./data
RESULTS=./results

flags_from_config() {
    python3 -c "
import json
c = json.load(open('$1'))
out = []
for k in ('no_kq', 'use_ln', 'use_residual', 'share_kq', 'use_mlp', 'learnable_eta'):
    if c.get(k):
        out.append('--' + k)
out += ['--d_hidden', str(c.get('d_hidden', 128))]
out += ['--base_lr', str(c.get('base_lr', 0.01))]
out += ['--smooth_window', str(c.get('smooth_window', 10))]
print(' '.join(out))
"
}

run_probe() {
    local probe_dir=$1
    local label_mode=$2
    local deltas=$3
    local include_ood=${4:-1}

    local fl=$(flags_from_config "$probe_dir/config.json")
    local probe_name=$(basename "$probe_dir")
    local out_dir="$RESULTS/${label_mode}__${probe_name}"
    mkdir -p "$out_dir"

    local in_dist="$DATA/qwen2.5-32b/s1k.pkl $DATA/qwen2.5-32b/openr1_2k.pkl $DATA/qwen2.5-32b/deepmath_2k.pkl"
    local ood_args=""
    if [ "$include_ood" = "1" ]; then
        ood_args="--ood_paths
            $DATA/qwen2.5-32b/math500.pkl
            $DATA/qwen2.5-32b/gpqa_diamond.pkl
            $DATA/qwen2.5-32b/aime24.pkl
            $DATA/qwen2.5-32b/aime25.pkl
            $DATA/qwen2.5-32b/aime26.pkl"
    fi

    python code/calibrate.py \
        --method ttt \
        --dataset_path $in_dist \
        --probe_path "$probe_dir/probe.pt" \
        --output_dir "$out_dir" \
        --label_mode "$label_mode" \
        --delta $deltas --epsilon 0.05 \
        $fl

    python code/test.py \
        --method ttt \
        --dataset_path $in_dist \
        $ood_args \
        --probe_path "$probe_dir/probe.pt" \
        --output_dir "$out_dir" \
        --label_mode "$label_mode" \
        --delta $deltas --epsilon 0.05 \
        $fl
}

# Tables 1, 2 + tab:sup_con: 4 main probes
for VARIANT in no_kq qk_dh128; do
    for MODE in supervised consistent; do
        run_probe "$PROBES/qwen2.5-32b/$MODE/$VARIANT" "$MODE" "0.05 0.1 0.15 0.2" 1
    done
done

# tab:dhidden
for DH in qk_dh32 qk_dh64 qk_dh256 qk_dh512; do
    run_probe "$PROBES/qwen2.5-32b/supervised/$DH" supervised "0.1" 0
done

# tab:ablation_arch
for ARCH in qk_dh128_ln qk_dh128_ln_res qk_dh128_share_kq qk_dh128_eta_learn qk_dh128_mlp; do
    run_probe "$PROBES/qwen2.5-32b/supervised/$ARCH" supervised "0.1" 1
done

echo "Done. Per-probe metrics under $RESULTS/<label>__<probe>/<label>/<run_name>/metrics.json"
