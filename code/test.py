"""
test.py — Evaluation: apply probe early stopping + compute metrics

Usage:
  python test.py --config configs/qwen32b_5k.yaml --method ttt --d_hidden 128
"""

import argparse
import os
import json
import logging
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_dataset, load_datasets, get_data_splits,
    BaseProbe, build_ttt_probe,
    smooth,
    add_common_args, add_ttt_args, parse_args_with_config,
    get_run_dir, save_config, save_metrics, append_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DELTA_GRID = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate probe")
    add_common_args(parser)
    add_ttt_args(parser)
    parser.add_argument("--probe_path", type=str, default=None)
    parser.add_argument("--lambdas_path", type=str, default=None)
    parser.add_argument("--delta", type=float, nargs="+", default=None,
                        help="Risk tolerance levels (default: DELTA_GRID)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="FWER confidence level (default: 0.05)")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--split_scheme", type=str, default="default",
                        choices=["default", "wu"])

    parser._option_string_actions["--method"].choices = ["base", "ttt", "compare"]

    args = parse_args_with_config(parser)
    return args


def get_stops_base(probe, step_embeddings, data, lambdas, delta, smooth_window, label):
    """Base probe early stopping on test split."""
    threshold = lambdas[str(delta)]
    stops = []
    val_scores = {}

    for i in data["test"]:
        ebds = np.array(step_embeddings[i])
        probs = probe.predict_proba(ebds)
        probs_smoothed = smooth(probs.tolist(), window=smooth_window)
        val_scores[i] = probs_smoothed

        early_t = len(probs_smoothed) - 1
        for t, p in enumerate(probs_smoothed):
            if p >= threshold:
                early_t = t
                break
        stops.append(early_t)

    return stops, val_scores


def get_stops_ttt(probe, step_embeddings, data, lambdas, delta, smooth_window,
                  label, batch_size, device, no_update=False):
    """TTT-Probe early stopping on test split (Algorithm 2)."""
    threshold = lambdas[str(delta)]
    stops = []
    val_scores = {}

    for i in data["test"]:
        embeds = step_embeddings[i]

        lbl_expanded = []
        for lbl in label[i]:
            lbl_expanded.extend([lbl] * batch_size)
        lbl_expanded = lbl_expanded[:len(embeds)]
        n_steps = len(lbl_expanded)

        probe.reset()
        raw_scores = []

        with torch.no_grad():
            for t in range(n_steps):
                phi_t = torch.tensor(embeds[t], dtype=torch.float32, device=device)
                s_t = probe.score(phi_t).item()
                raw_scores.append(s_t)
                if not no_update:
                    probe.update(phi_t, C_t=0.0)

        scores_smoothed = smooth(raw_scores, window=smooth_window)
        val_scores[i] = scores_smoothed

        early_t = len(scores_smoothed) - 1
        for t, p in enumerate(scores_smoothed):
            if p >= threshold:
                early_t = t
                break
        stops.append(early_t)

    return stops, val_scores


def evaluate(stops, step_embeddings, label, split_indices, batch_size):
    """Compute metrics from stopping points."""
    errors = 0
    total_savings = 0.0
    n = 0

    val_indices = list(split_indices)

    for k, i in enumerate(val_indices):
        if len(label[i]) == 0:
            continue

        lbl_expanded = []
        for lbl in label[i]:
            lbl_expanded.extend([lbl] * batch_size)
        lbl_expanded = lbl_expanded[:len(step_embeddings[i])]
        n_steps = len(lbl_expanded)

        stop_t = min(stops[k], n_steps - 1)
        if lbl_expanded[stop_t] == 0:
            errors += 1

        total_savings += 1.0 - (stop_t + 1) / n_steps
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0

    error_rate = errors / n
    avg_savings = total_savings / n
    accuracy = 1.0 - error_rate

    return error_rate, avg_savings, accuracy


def run_evaluation(args):
    """Run full evaluation for a single method."""
    run_dir = get_run_dir(args)
    delta_grid = args.delta or DELTA_GRID

    # load data
    step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
    data = get_data_splits(metadata, scheme=getattr(args, "split_scheme", "default"))

    # load lambdas
    lambdas_path = args.lambdas_path or os.path.join(run_dir, "lambdas.json")
    with open(lambdas_path) as f:
        lambdas = json.load(f)
    log.info(f"Loaded lambdas from {lambdas_path}")

    # setup probe
    if args.method == "base":
        probe_path = args.probe_path or os.path.join(run_dir, "probe.pkl")
        probe = BaseProbe()
        probe.load(probe_path)
        log.info(f"Loaded Base probe from {probe_path}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        probe_path = args.probe_path or os.path.join(run_dir, "probe.pt")
        probe = build_ttt_probe(args, device)
        probe.load(probe_path, device)
        probe.eval()
        if args.use_pca:
            pca_path = os.path.join(os.path.dirname(probe_path), "pca.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, "rb") as f:
                    probe.scaler, probe.pca = pickle.load(f)
        log.info(f"Loaded TTT probe from {probe_path}")

    no_update = getattr(args, "no_online_update", False)

    # evaluate at each delta
    delta_results = {}
    all_val_scores = None

    for delta in delta_grid:
        delta_str = str(delta)
        if delta_str not in lambdas:
            log.warning(f"delta={delta} not in lambdas, skipping")
            continue

        if args.method == "base":
            stops, val_scores = get_stops_base(
                probe, step_embeddings, data, lambdas, delta,
                args.smooth_window, label
            )
        else:
            stops, val_scores = get_stops_ttt(
                probe, step_embeddings, data, lambdas, delta,
                args.smooth_window, label, args.batch_size, device, no_update
            )

        if all_val_scores is None:
            all_val_scores = val_scores

        error_rate, avg_savings, accuracy = evaluate(
            stops, step_embeddings, label, data["test"], args.batch_size
        )

        delta_results[delta_str] = {
            "lambda": lambdas[delta_str],
            "error_rate": round(error_rate, 4),
            "savings": round(avg_savings, 4),
            "accuracy": round(accuracy, 4),
        }

        log.info(f"  delta={delta:.3f}: lambda={lambdas[delta_str]:.4f}, "
                 f"error={error_rate:.4f}, savings={avg_savings:.4f}")

    # load training metrics if available
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing_metrics = json.load(f)
    else:
        existing_metrics = {}

    existing_metrics["eps_results"] = delta_results
    save_metrics(existing_metrics, run_dir)
    append_summary(run_dir, existing_metrics, args)

    # save val scores
    scores_dir = os.path.join(run_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    with open(os.path.join(scores_dir, "val_scores.pkl"), "wb") as f:
        pickle.dump(all_val_scores, f)

    # print summary table
    log.info("\n=== Results Summary ===")
    log.info(f"{'delta':>8} {'lambda':>8} {'error':>8} {'savings':>8} {'accuracy':>8}")
    for delta_str, res in sorted(delta_results.items(), key=lambda x: float(x[0])):
        log.info(f"{delta_str:>8} {res['lambda']:>8.4f} {res['error_rate']:>8.4f} "
                 f"{res['savings']:>8.4f} {res['accuracy']:>8.4f}")

    # visualization
    if args.visualize and all_val_scores is not None:
        visualize_trajectories(args, all_val_scores, label, lambdas, data, run_dir)

    # OOD evaluation
    if args.ood_paths:
        evaluate_ood(args, run_dir, probe, lambdas, delta_grid)

    return delta_results


def evaluate_ood(args, run_dir, probe, lambdas, delta_grid):
    """Evaluate probe on OOD datasets. Uses ALL problems (no train/val/test split)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_update = getattr(args, "no_online_update", False)

    for ood_path in args.ood_paths:
        ood_name = os.path.basename(os.path.dirname(ood_path))
        log.info(f"\n=== OOD Evaluation: {ood_name} ===")

        ood_embeddings, ood_labels, ood_meta = load_dataset(ood_path, args.label_mode)
        ood_batch_size = ood_meta.get("batch_size", args.batch_size)
        all_indices = range(len(ood_embeddings))

        ood_results = {}
        for delta in delta_grid:
            delta_str = str(delta)
            if delta_str not in lambdas:
                continue
            threshold = lambdas[delta_str]

            stops = []
            for i in all_indices:
                embeds = ood_embeddings[i]
                if len(embeds) == 0 or len(ood_labels[i]) == 0:
                    stops.append(0)
                    continue

                lbl_exp = []
                for lbl in ood_labels[i]:
                    lbl_exp.extend([lbl] * ood_batch_size)
                lbl_exp = lbl_exp[:len(embeds)]
                n_steps = len(lbl_exp)

                if args.method == "base":
                    ebds = np.array(embeds[:n_steps])
                    probs = probe.predict_proba(ebds)
                    scores = smooth(probs.tolist(), window=args.smooth_window)
                else:
                    probe.reset()
                    raw_scores = []
                    with torch.no_grad():
                        for t in range(n_steps):
                            phi_t = torch.tensor(embeds[t], dtype=torch.float32, device=device)
                            s_t = probe.score(phi_t).item()
                            raw_scores.append(s_t)
                            if not no_update:
                                probe.update(phi_t, C_t=0.0)
                    scores = smooth(raw_scores, window=args.smooth_window)

                early_t = len(scores) - 1
                for t, p in enumerate(scores):
                    if p >= threshold:
                        early_t = t
                        break
                stops.append(early_t)

            error_rate, avg_savings, accuracy = evaluate(
                stops, ood_embeddings, ood_labels, all_indices, ood_batch_size
            )

            ood_results[delta_str] = {
                "lambda": lambdas[delta_str],
                "error_rate": round(error_rate, 4),
                "savings": round(avg_savings, 4),
                "accuracy": round(accuracy, 4),
            }

            log.info(f"  delta={delta:.3f}: error={error_rate:.4f}, savings={avg_savings:.4f}")

        # Save OOD results
        ood_metrics_path = os.path.join(run_dir, f"ood_{ood_name}.json")
        with open(ood_metrics_path, "w") as f:
            json.dump(ood_results, f, indent=2)

        log.info(f"  Saved -> {ood_metrics_path}")


def visualize_trajectories(args, val_scores, label, lambdas, data, run_dir):
    """Score trajectory visualization."""
    log.info("Generating trajectory visualizations...")

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    val_indices = [i for i in data["test"] if len(label[i]) > 0]
    n_show = min(args.n_examples, len(val_indices))

    problem_lengths = [(i, len(val_scores.get(i, []))) for i in val_indices if i in val_scores]
    problem_lengths.sort(key=lambda x: x[1])

    step = max(1, len(problem_lengths) // n_show)
    selected = [problem_lengths[j * step][0] for j in range(n_show) if j * step < len(problem_lengths)]

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 3 * len(selected)), squeeze=False)

    delta_for_line = "0.1" if "0.1" in lambdas else list(lambdas.keys())[len(lambdas) // 2]
    threshold = lambdas[delta_for_line]

    for ax_idx, prob_idx in enumerate(selected):
        ax = axes[ax_idx, 0]
        scores = val_scores[prob_idx]
        steps = range(len(scores))

        ax.plot(steps, scores, "b-", alpha=0.8, label=f"{args.method} scores")
        ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.5,
                    label=f"threshold (delta={delta_for_line})")

        lbl_expanded = []
        for lbl in label[prob_idx]:
            lbl_expanded.extend([lbl] * args.batch_size)
        lbl_expanded = lbl_expanded[:len(scores)]

        for t in range(1, len(lbl_expanded)):
            if lbl_expanded[t] == 1 and lbl_expanded[t - 1] == 0:
                ax.axvline(x=t, color="g", linestyle=":", alpha=0.7, label="correct transition")
                break

        ax.set_ylabel("Score")
        ax.set_title(f"Problem {prob_idx} ({len(scores)} steps)")
        ax.set_ylim(-0.05, 1.05)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Reasoning Step")
    plt.tight_layout()
    fp = os.path.join(plots_dir, "trajectories.pdf")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved trajectory plot to {fp}")


def compare(args):
    """Compare mode: run both base and ttt, output comparison table."""
    delta_grid = args.delta or DELTA_GRID

    log.info("=== Compare Mode ===")

    base_args = argparse.Namespace(**vars(args))
    base_args.method = "base"
    base_args.run_name = None
    base_run_dir = get_run_dir(base_args)

    ttt_args = argparse.Namespace(**vars(args))
    ttt_args.method = "ttt"
    ttt_args.run_name = None
    ttt_run_dir = get_run_dir(ttt_args)

    base_metrics_path = os.path.join(base_run_dir, "metrics.json")
    ttt_metrics_path = os.path.join(ttt_run_dir, "metrics.json")

    if not os.path.exists(base_metrics_path):
        log.info("Running Base evaluation...")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.method = "base"
        args_copy.run_name = None
        run_evaluation(args_copy)

    if not os.path.exists(ttt_metrics_path):
        log.info("Running TTT evaluation...")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.method = "ttt"
        args_copy.run_name = None
        run_evaluation(args_copy)

    with open(base_metrics_path) as f:
        base_metrics = json.load(f)
    with open(ttt_metrics_path) as f:
        ttt_metrics = json.load(f)

    log.info("\n=== Base vs TTT-Probe Comparison ===")
    log.info(f"{'delta':>8} {'Base err':>10} {'Base sav':>10} {'TTT err':>10} {'TTT sav':>10} {'diff':>8}")

    base_res = base_metrics.get("eps_results", {})
    ttt_res = ttt_metrics.get("eps_results", {})

    for delta in delta_grid:
        delta_str = str(delta)
        b = base_res.get(delta_str, {})
        t = ttt_res.get(delta_str, {})
        b_sav = b.get("savings", 0)
        t_sav = t.get("savings", 0)
        diff = t_sav - b_sav

        log.info(f"{delta_str:>8} {b.get('error_rate', '-'):>10} {b_sav:>10.4f} "
                 f"{t.get('error_rate', '-'):>10} {t_sav:>10.4f} {diff:>+8.4f}")

    # Generate risk-compute plot
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    base_x, base_y = [], []
    ttt_x, ttt_y = [], []

    for delta in delta_grid:
        delta_str = str(delta)
        if delta_str in base_res:
            base_x.append(delta)
            base_y.append(base_res[delta_str].get("savings", 0))
        if delta_str in ttt_res:
            ttt_x.append(delta)
            ttt_y.append(ttt_res[delta_str].get("savings", 0))

    ax.plot(base_x, base_y, "o--", color="gray", label="Base", markersize=6)
    ax.plot(ttt_x, ttt_y, "s-", color="steelblue", label="TTT-Probe", markersize=6)
    ax.set_xlabel("Risk tolerance (delta)")
    ax.set_ylabel("Average savings")
    dataset_name = os.path.basename(os.path.dirname(args.dataset_path[0]))
    ax.set_title(f"Risk-Compute Trade-off ({dataset_name}, {args.label_mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fp = os.path.join(plots_dir, f"risk_compute__{dataset_name}__{args.label_mode}.pdf")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved risk-compute plot to {fp}")


def main():
    args = parse_args()
    with open(args.dataset_path[0], "rb") as f:
        meta = pickle.load(f)
    args.d_phi = meta["embed_dim"]

    if args.method == "compare":
        compare(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
