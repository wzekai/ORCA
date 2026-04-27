"""
calibrate.py — LTT Calibration: determine threshold lambda*

Usage:
  python calibrate.py --config configs/qwen32b_5k.yaml --method ttt --d_hidden 128
"""

import argparse
import os
import json
import logging
import pickle

import numpy as np
import torch

from utils import (
    load_datasets, get_data_splits,
    BaseProbe, build_ttt_probe,
    smooth, run_test,
    add_common_args, add_ttt_args, parse_args_with_config,
    get_run_dir, save_config, save_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DELTA_GRID = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(description="LTT Calibration")
    add_common_args(parser)
    add_ttt_args(parser)
    parser.add_argument("--probe_path", type=str, default=None)
    parser.add_argument("--delta", type=float, nargs="+", default=None,
                        help="Risk tolerance levels (default: DELTA_GRID)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="FWER confidence level (default: 0.05)")
    return parse_args_with_config(parser)


def calibrate_base(args):
    """Base probe calibration: one-shot predict_proba on all steps."""
    run_dir = get_run_dir(args)
    delta_grid = args.delta or DELTA_GRID

    log.info(f"Calibrating Base probe: {args.dataset_path}")

    # load probe
    probe_path = args.probe_path or os.path.join(run_dir, "probe.pkl")
    probe = BaseProbe()
    probe.load(probe_path)
    log.info(f"Loaded probe from {probe_path}")

    # load data
    step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
    data = get_data_splits(metadata)

    # make predictions on calibration set
    preds = []
    trues = []
    for i in data["calibration"]:
        if len(label[i]) == 0:
            continue
        ebds = np.array(step_embeddings[i])
        probs = probe.predict_proba(ebds)
        probs = smooth(probs, window=args.smooth_window)
        preds.append(probs)

        # expand labels: each label covers batch_size embeddings
        lbl_expanded = []
        for lbl in label[i]:
            lbl_expanded.extend([lbl] * args.batch_size)
        lbl_expanded = lbl_expanded[:len(probs)]
        trues.append(lbl_expanded)

    log.info(f"Calibrating on {len(preds)} problems")

    # run LTT for each delta
    lambdas = {}
    for delta in delta_grid:
        lam = run_test(preds, trues, delta, epsilon=args.epsilon)
        lambdas[str(delta)] = lam
        log.info(f"  delta={delta:.3f} → lambda={lam:.4f}")

    # save
    lambdas_path = os.path.join(run_dir, "lambdas.json")
    with open(lambdas_path, "w") as f:
        json.dump(lambdas, f, indent=2)

    log.info(f"Saved lambdas to {lambdas_path}")


def calibrate_ttt(args):
    """TTT-Probe calibration: sequential score→update with C_t=0 (Algorithm 2)."""
    run_dir = get_run_dir(args)
    delta_grid = args.delta or DELTA_GRID

    log.info(f"Calibrating TTT probe: {args.dataset_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # load probe
    probe_path = args.probe_path or os.path.join(run_dir, "probe.pt")
    probe = build_ttt_probe(args, device)
    probe.load(probe_path, device)
    probe.eval()
    log.info(f"Loaded probe from {probe_path}")

    # load PCA if needed
    if args.use_pca:
        pca_path = os.path.join(os.path.dirname(probe_path), "pca.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                probe.scaler, probe.pca = pickle.load(f)
            log.info(f"Loaded PCA from {pca_path}")

    # load data
    step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
    data = get_data_splits(metadata)

    no_update = getattr(args, "no_online_update", False)

    # score each problem in calibration set sequentially
    preds = []
    trues = []
    cal_count = 0
    for i in data["calibration"]:
        if len(label[i]) == 0:
            continue
        embeds = step_embeddings[i]

        # expand labels
        lbl_expanded = []
        for lbl in label[i]:
            lbl_expanded.extend([lbl] * args.batch_size)
        lbl_expanded = lbl_expanded[:len(embeds)]
        if len(lbl_expanded) == 0:
            continue

        # sequential score→update
        probe.reset()
        scores = []
        with torch.no_grad():
            for t in range(len(lbl_expanded)):
                phi_t = torch.tensor(embeds[t], dtype=torch.float32, device=device)
                s_t = probe.score(phi_t).item()
                scores.append(s_t)
                if not no_update:
                    probe.update(phi_t, C_t=0.0)

        scores_smoothed = smooth(scores, window=args.smooth_window)
        preds.append(scores_smoothed)
        trues.append(lbl_expanded)
        cal_count += 1

    log.info(f"Calibrating on {cal_count} problems")

    # run LTT for each delta
    lambdas = {}
    for delta in delta_grid:
        lam = run_test(preds, trues, delta, epsilon=args.epsilon)
        lambdas[str(delta)] = lam
        log.info(f"  delta={delta:.3f} → lambda={lam:.4f}")

    # save
    lambdas_path = os.path.join(run_dir, "lambdas.json")
    with open(lambdas_path, "w") as f:
        json.dump(lambdas, f, indent=2)

    log.info(f"Saved lambdas to {lambdas_path}")


def main():
    args = parse_args()
    with open(args.dataset_path[0], "rb") as f:
        meta = pickle.load(f)
    args.d_phi = meta["embed_dim"]

    if args.method == "base":
        calibrate_base(args)
    elif args.method == "ttt":
        calibrate_ttt(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
