"""
epoch_sweep.py — Evaluate each checkpoint to find optimal training epoch

For each checkpoint (probe_ep10.pt, probe_ep20.pt, ..., probe_ep200.pt),
runs calibrate + test inline and collects results into a single JSON.

Usage:
  python epoch_sweep.py --config configs/qwen32b_5k.yaml --method ttt --d_hidden 128 --epochs 200
  python epoch_sweep.py --config configs/qwen32b_5k.yaml --method ttt --no_kq --epochs 200
"""

import argparse
import os
import json
import logging
import pickle

import numpy as np
import torch

from utils import (
    load_dataset, load_datasets, get_data_splits,
    build_ttt_probe, smooth, run_test,
    add_common_args, add_ttt_args, parse_args_with_config,
    get_run_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DELTA_GRID = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(description="Epoch sweep evaluation")
    add_common_args(parser)
    add_ttt_args(parser)
    parser.add_argument("--delta", type=float, nargs="+", default=None,
                        help="Risk tolerance levels")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="FWER confidence level (default: 0.05)")
    parser.add_argument("--save_every", type=int, default=10)
    return parse_args_with_config(parser)


def score_problems(probe, step_embeddings, label, indices, batch_size, device, no_update=False):
    """Score all problems sequentially with score→update. Returns (preds, trues)."""
    preds = []
    trues = []
    for i in indices:
        if len(label[i]) == 0:
            continue
        embeds = step_embeddings[i]
        lbl_expanded = []
        for lbl in label[i]:
            lbl_expanded.extend([lbl] * batch_size)
        lbl_expanded = lbl_expanded[:len(embeds)]
        if len(lbl_expanded) == 0:
            continue

        probe.reset()
        scores = []
        with torch.no_grad():
            for t in range(len(lbl_expanded)):
                phi_t = torch.tensor(embeds[t], dtype=torch.float32, device=device)
                s_t = probe.score(phi_t).item()
                scores.append(s_t)
                if not no_update:
                    probe.update(phi_t, C_t=0.0)

        preds.append(scores)
        trues.append(lbl_expanded)
    return preds, trues


def evaluate_stops(preds, trues, lambdas, delta_grid, smooth_window):
    """Compute error_rate and savings for each delta using precomputed scores."""
    results = {}
    for delta in delta_grid:
        eps_str = str(delta)
        if eps_str not in lambdas:
            continue
        threshold = lambdas[eps_str]

        errors = 0
        total_savings = 0.0
        n = 0
        for scores, labels in zip(preds, trues):
            scores_sm = smooth(scores, window=smooth_window)
            n_steps = len(scores_sm)
            early_t = n_steps - 1
            for t, p in enumerate(scores_sm):
                if p >= threshold:
                    early_t = t
                    break
            if labels[early_t] == 0:
                errors += 1
            total_savings += 1.0 - (early_t + 1) / n_steps
            n += 1

        if n > 0:
            results[eps_str] = {
                "error_rate": round(errors / n, 4),
                "savings": round(total_savings / n, 4),
            }
    return results


def main():
    args = parse_args()
    with open(args.dataset_path[0], "rb") as f:
        meta = pickle.load(f)
    args.d_phi = meta["embed_dim"]

    run_dir = get_run_dir(args)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    delta_grid = args.delta or DELTA_GRID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_update = getattr(args, "no_online_update", False)

    # enumerate checkpoints
    checkpoints = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("probe_ep") and f.endswith(".pt")],
        key=lambda x: int(x.replace("probe_ep", "").replace(".pt", ""))
    )
    log.info(f"Found {len(checkpoints)} checkpoints in {ckpt_dir}")

    # load data once
    step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
    data = get_data_splits(metadata)

    # load PCA once if needed
    pca_scaler, pca_model = None, None
    if args.use_pca:
        pca_path = os.path.join(run_dir, "pca.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                pca_scaler, pca_model = pickle.load(f)
            log.info(f"Loaded PCA from {pca_path}")

    # load OOD data once
    ood_data = {}
    for ood_path in (args.ood_paths or []):
        ood_name = os.path.basename(os.path.dirname(ood_path))
        ood_emb, ood_lab, ood_meta = load_dataset(ood_path, args.label_mode)
        ood_bs = ood_meta.get("batch_size", args.batch_size)
        ood_data[ood_name] = (ood_emb, ood_lab, ood_bs, range(len(ood_emb)))

    sweep_results = {}

    for ckpt_name in checkpoints:
        ep = int(ckpt_name.replace("probe_ep", "").replace(".pt", ""))
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        log.info(f"\n=== Epoch {ep} ===")

        # build and load probe
        probe = build_ttt_probe(args, device)
        probe.load(ckpt_path, device)
        probe.eval()
        if pca_scaler is not None:
            probe.scaler = pca_scaler
            probe.pca = pca_model

        # 1. Calibrate on cal set
        cal_preds, cal_trues = score_problems(
            probe, step_embeddings, label, data["calibration"],
            args.batch_size, device, no_update
        )

        # smooth then run LTT
        cal_preds_sm = [smooth(s, window=args.smooth_window) for s in cal_preds]
        lambdas = {}
        for delta in delta_grid:
            lam = run_test(cal_preds_sm, cal_trues, delta, epsilon=args.epsilon)
            lambdas[str(delta)] = lam

        # 2. Test on test set
        test_preds, test_trues = score_problems(
            probe, step_embeddings, label, data["test"],
            args.batch_size, device, no_update
        )

        test_results = evaluate_stops(test_preds, test_trues, lambdas, delta_grid, args.smooth_window)

        entry = {"lambdas": lambdas, "in_dist": test_results}

        # 3. OOD evaluation
        for ood_name, (ood_emb, ood_lab, ood_bs, ood_idx) in ood_data.items():
            ood_preds, ood_trues = score_problems(
                probe, ood_emb, ood_lab, ood_idx,
                ood_bs, device, no_update
            )
            ood_results = evaluate_stops(ood_preds, ood_trues, lambdas, delta_grid, args.smooth_window)
            entry[ood_name] = ood_results

        sweep_results[ep] = entry

        # log key metrics
        r01 = test_results.get("0.1", {})
        r015 = test_results.get("0.15", {})
        log.info(f"  in-dist eps=0.1: err={r01.get('error_rate', '-')}, sav={r01.get('savings', '-')}")
        log.info(f"  in-dist eps=0.15: err={r015.get('error_rate', '-')}, sav={r015.get('savings', '-')}")

        for ood_name in ood_data:
            ood_r = entry.get(ood_name, {}).get("0.1", {})
            log.info(f"  {ood_name} eps=0.1: err={ood_r.get('error_rate', '-')}, sav={ood_r.get('savings', '-')}")

    # save results
    output_path = os.path.join(run_dir, "epoch_sweep.json")
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    log.info(f"\nSaved epoch sweep results to {output_path}")

    # print summary table
    header = f"{'Epoch':>6} {'in-dist err':>12} {'in-dist sav':>12}"
    for ood_name in ood_data:
        header += f" {ood_name+' sav':>12}"
    log.info("\n=== Epoch Sweep Summary (eps=0.1) ===")
    log.info(header)

    for ep in sorted(sweep_results.keys()):
        r = sweep_results[ep]
        indist = r.get("in_dist", {}).get("0.1", {})
        line = f"{ep:>6} {indist.get('error_rate', '-'):>12} {indist.get('savings', '-'):>12}"
        for ood_name in ood_data:
            ood_r = r.get(ood_name, {}).get("0.1", {})
            line += f" {ood_r.get('savings', '-'):>12}"
        log.info(line)

    # find best epoch for in-dist savings at eps=0.1
    best_ep = max(sweep_results.keys(),
                  key=lambda e: sweep_results[e].get("in_dist", {}).get("0.1", {}).get("savings", 0))
    best_sav = sweep_results[best_ep]["in_dist"]["0.1"]["savings"]
    best_err = sweep_results[best_ep]["in_dist"]["0.1"]["error_rate"]
    log.info(f"\nBest epoch (in-dist, eps=0.1): ep{best_ep} — sav={best_sav}, err={best_err}")


if __name__ == "__main__":
    main()
