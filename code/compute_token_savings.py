"""
compute_token_savings.py — Post-processing script to compute token-level savings

Given step-level savings (from test.py), converts to token-level savings
using step_limits from dataset.pkl.

Token savings = 1 - (tokens consumed up to stop_step) / (total tokens)

Usage:
  python compute_token_savings.py --config configs/qwen32b_5k.yaml --method ttt --no_kq --run_name ttt__no_kq__lr0.01__final_ep20
"""

import argparse
import os
import json
import pickle
import numpy as np

from utils import (
    load_datasets, get_data_splits, smooth,
    add_common_args, add_ttt_args, parse_args_with_config,
    get_run_dir, LABEL_FIELDS,
)


def compute_token_savings_for_run(args, delta="0.1"):
    """Compute token-level savings for a single run."""
    run_dir = get_run_dir(args)

    # Load lambdas
    lambdas_path = os.path.join(run_dir, "lambdas.json")
    with open(lambdas_path) as f:
        lambdas = json.load(f)

    if delta not in lambdas:
        print(f"  delta={delta} not in lambdas, skipping")
        return None

    threshold = lambdas[delta]

    # Load val_scores if available, otherwise we need to recompute
    scores_path = os.path.join(run_dir, "scores", "val_scores.pkl")
    if not os.path.exists(scores_path):
        print(f"  No val_scores.pkl in {run_dir}, skipping")
        return None

    with open(scores_path, "rb") as f:
        val_scores = pickle.load(f)

    # Load datasets to get step_limits and labels
    all_step_limits = []
    all_labels = []
    label_field = LABEL_FIELDS[args.label_mode]

    for path in args.dataset_path:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for prob in data["problems"]:
            all_step_limits.append(prob["step_limits"])
            lbl = list(prob[label_field])
            # to_cumulative
            if 1 in lbl:
                first = lbl.index(1)
                for j in range(first, len(lbl)):
                    lbl[j] = 1
            else:
                lbl = []
            all_labels.append(lbl)

    splits = get_data_splits(
        {"splits": json.load(open(os.path.join(run_dir, "config.json")))["splits"]}
        if os.path.exists(os.path.join(run_dir, "config.json")) else
        load_datasets(args.dataset_path, args.label_mode)[2]
    )

    # Actually, simpler: just load the full dataset metadata
    _, labels, metadata = load_datasets(args.dataset_path, args.label_mode)
    splits = get_data_splits(metadata)

    # Reload raw step_limits (not from utils, need raw data)
    raw_step_limits = []
    for path in args.dataset_path:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for prob in data["problems"]:
            raw_step_limits.append(prob["step_limits"])

    batch_size = args.batch_size

    step_savings_list = []
    token_savings_list = []
    errors_step = 0
    errors_token = 0  # should be same as step errors
    n = 0

    for i in splits["test"]:
        if len(labels[i]) == 0:
            continue
        if i not in val_scores:
            continue

        scores = val_scores[i]
        sl = raw_step_limits[i]

        # Expand labels
        lbl_expanded = []
        for lbl in labels[i]:
            lbl_expanded.extend([lbl] * batch_size)
        lbl_expanded = lbl_expanded[:len(scores)]
        n_steps = len(lbl_expanded)

        # Find stop step
        stop_step = n_steps - 1
        for t, s in enumerate(scores):
            if s >= threshold:
                stop_step = t
                break

        # Step savings
        step_sav = 1.0 - (stop_step + 1) / n_steps

        # Token savings: need to map step index to token count
        # Each label covers batch_size embeddings, and each embedding corresponds to one step_limit entry
        # stop_step is in embedding space (0 to len(scores)-1)
        # step_limits has one entry per label (not per embedding)
        # So stop_step // batch_size gives the label index, which maps to step_limits

        # Total tokens
        if len(sl) == 0:
            continue
        total_tokens = sl[-1][1] - sl[0][0]

        # Tokens at stop
        label_idx_at_stop = min(stop_step // batch_size, len(sl) - 1)
        tokens_at_stop = sl[label_idx_at_stop][1] - sl[0][0]

        token_sav = 1.0 - tokens_at_stop / total_tokens if total_tokens > 0 else 0.0

        step_savings_list.append(step_sav)
        token_savings_list.append(token_sav)

        # Error check
        stop_step_clipped = min(stop_step, n_steps - 1)
        if lbl_expanded[stop_step_clipped] == 0:
            errors_step += 1

        n += 1

    if n == 0:
        return None

    return {
        "n_problems": n,
        "step_savings": round(np.mean(step_savings_list), 4),
        "token_savings": round(np.mean(token_savings_list), 4),
        "error_rate": round(errors_step / n, 4),
        "step_savings_std": round(np.std(step_savings_list), 4),
        "token_savings_std": round(np.std(token_savings_list), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute token-level savings")
    add_common_args(parser)
    add_ttt_args(parser)
    parser.add_argument("--delta", type=str, default="0.1")
    args = parse_args_with_config(parser)

    result = compute_token_savings_for_run(args, delta=args.delta)
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
