#!/usr/bin/env bash
# Reproduce Table 3 (tab:cross_model): qwq-32b + llama-3.3-70b, supervised, delta=0.1.

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

for LLM in qwq-32b llama-3.3-70b; do
    for VARIANT in no_kq qk_dh128; do
        probe_dir="$PROBES/$LLM/supervised/$VARIANT"
        in_dist="$DATA/$LLM/s1k.pkl $DATA/$LLM/openr1_2k.pkl $DATA/$LLM/deepmath_2k.pkl"
        out_dir="$RESULTS/${LLM}__supervised__${VARIANT}"
        mkdir -p "$out_dir"

        fl=$(flags_from_config "$probe_dir/config.json")

        python code/calibrate.py \
            --method ttt \
            --dataset_path $in_dist \
            --probe_path "$probe_dir/probe.pt" \
            --output_dir "$out_dir" \
            --label_mode supervised \
            --delta 0.1 --epsilon 0.05 \
            $fl

        python code/test.py \
            --method ttt \
            --dataset_path $in_dist \
            --probe_path "$probe_dir/probe.pt" \
            --output_dir "$out_dir" \
            --label_mode supervised \
            --delta 0.1 --epsilon 0.05 \
            $fl
    done
done

echo "Done. Cross-model metrics under $RESULTS/<llm>__supervised__<variant>/supervised/<run_name>/metrics.json"
