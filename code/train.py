"""
train.py — Train probe (Base or TTT)

Usage:
  python train.py --config configs/qwen32b_5k.yaml --method ttt --d_hidden 128
  python train.py --config configs/qwen32b_5k.yaml --method base
"""

import argparse
import os
import json
import logging
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (
    load_datasets, get_data_splits,
    BaseProbe, build_ttt_probe,
    add_common_args, add_ttt_args, parse_args_with_config,
    get_run_dir, save_config, save_metrics, append_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train probe")
    add_common_args(parser)
    add_ttt_args(parser)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save probe checkpoint every N epochs (0 = disabled)")
    return parse_args_with_config(parser)


def train_base(args):
    """Train Base probe: StandardScaler → PCA → LogisticRegression."""
    run_dir = get_run_dir(args)

    # skip if already trained
    probe_path = os.path.join(run_dir, "probe.pkl")
    if os.path.exists(probe_path) and not args.force_retrain:
        log.info(f"Probe already exists at {probe_path}, skipping training.")
        return

    log.info(f"Training Base probe: {args.dataset_path}")

    # load data
    step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
    data = get_data_splits(metadata)

    # assemble training data: use embedding at batch boundary (matching Wu's get_last)
    # Wu: last_rep = (j+1)*batch_size, NO subtraction of 1; clamp if overshoots
    def _get_last(problem_idx, rel_idx):
        last_rep = (rel_idx + 1) * args.batch_size
        if last_rep >= len(step_embeddings[problem_idx]):
            last_rep = len(step_embeddings[problem_idx]) - 1
        return step_embeddings[problem_idx][last_rep]

    X_train, y_train = [], []
    for i in data["train"]:
        if len(label[i]) == 0:
            continue
        for j in range(len(label[i])):
            X_train.append(_get_last(i, j))
            y_train.append(label[i][j])

    X_val, y_val = [], []
    for i in data["calibration"]:
        if len(label[i]) == 0:
            continue
        np.random.seed(i)
        j = np.random.choice(len(label[i]))
        X_val.append(_get_last(i, j))
        y_val.append(label[i][j])

    X_test, y_test = [], []
    for i in data["test"]:
        if len(label[i]) == 0:
            continue
        for j in range(len(label[i])):
            X_test.append(_get_last(i, j))
            y_test.append(label[i][j])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    log.info(f"Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # fit probe
    probe = BaseProbe(n_components=256, max_iter=5000)
    probe.fit(X_train, y_train)

    # evaluate AUROC
    train_auroc = roc_auc_score(y_train, probe.predict_proba(X_train))
    val_auroc = roc_auc_score(y_val, probe.predict_proba(X_val))
    test_auroc = roc_auc_score(y_test, probe.predict_proba(X_test))

    log.info(f"AUROC — train: {train_auroc:.4f}, val: {val_auroc:.4f}, test: {test_auroc:.4f}")

    # save
    probe.save(probe_path)
    save_config(args, run_dir)

    metrics = {"train_auroc": train_auroc, "val_auroc": val_auroc, "test_auroc": test_auroc}
    save_metrics(metrics, run_dir)

    log.info(f"Saved probe to {probe_path}")


def train_ttt(args):
    """Meta-train TTTProbe (Algorithm 1)."""
    run_dir = get_run_dir(args)

    # skip if already trained
    probe_path = os.path.join(run_dir, "probe.pt")
    if os.path.exists(probe_path) and not args.force_retrain:
        log.info(f"Probe already exists at {probe_path}, skipping training.")
        return

    if args.no_meta_train:
        log.info("--no_meta_train: saving randomly initialized probe")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        probe = build_ttt_probe(args, device)

        # fit PCA if needed
        if args.use_pca:
            step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
            data = get_data_splits(metadata)
            all_embeds = []
            for i in data["train"]:
                if len(label[i]) == 0:
                    continue
                all_embeds.extend(step_embeddings[i])
            all_embeds = np.array(all_embeds)
            probe.scaler = StandardScaler().fit(all_embeds)
            probe.pca = PCA(n_components=args.pca_dim, random_state=0).fit(
                probe.scaler.transform(all_embeds)
            )

        probe.save(probe_path)
        if probe.scaler is not None:
            with open(os.path.join(run_dir, "pca.pkl"), "wb") as f:
                pickle.dump([probe.scaler, probe.pca], f)
        save_config(args, run_dir)
        save_metrics({"train_auroc": None, "val_auroc": None}, run_dir)
        log.info(f"Saved random-init probe to {probe_path}")
        return

    log.info(f"Meta-training TTTProbe: {args.dataset_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # set random seed before probe construction (for reproducible W0 init)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    step_embeddings, label, metadata = load_datasets(args.dataset_path, args.label_mode)
    data = get_data_splits(metadata)

    # build probe
    probe = build_ttt_probe(args, device)

    # fit PCA if needed
    if args.use_pca:
        all_embeds = []
        for i in data["train"]:
            if len(label[i]) == 0:
                continue
            all_embeds.extend(step_embeddings[i])
        all_embeds = np.array(all_embeds)
        probe.scaler = StandardScaler().fit(all_embeds)
        probe.pca = PCA(n_components=args.pca_dim, random_state=0).fit(
            probe.scaler.transform(all_embeds)
        )

    # build training sequences: for each problem, (phi_seq, C_seq)
    train_data = []
    for i in data["train"]:
        if len(label[i]) == 0:
            continue
        embeds = step_embeddings[i]
        # expand labels: each label covers batch_size embeddings
        C_expanded = []
        for lbl in label[i]:
            C_expanded.extend([lbl] * args.batch_size)
        # trim to actual number of embeddings
        C_expanded = C_expanded[:len(embeds)]
        if len(C_expanded) == 0:
            continue

        phi_seq = [torch.tensor(e, dtype=torch.float32, device=device) for e in embeds[:len(C_expanded)]]
        C_seq = [torch.tensor(c, dtype=torch.float32, device=device) for c in C_expanded]
        train_data.append((phi_seq, C_seq))


    # optimizer
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.outer_lr)

    # meta-training loop
    train_log = []

    for epoch in range(args.epochs):
        probe.train()
        epoch_loss = 0.0
        n_steps = 0

        # shuffle training order
        perm = np.random.permutation(len(train_data))

        for idx in perm:
            phi_seq, C_seq = train_data[idx]

            # forward trajectory
            scores, outer_loss = probe.forward_trajectory(
                phi_seq, C_seq
            )

            # normalize by sequence length
            outer_loss = outer_loss / len(phi_seq)

            optimizer.zero_grad()
            outer_loss.backward()

            # gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), args.grad_clip)

            optimizer.step()

            epoch_loss += outer_loss.item()
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)

        # logging every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_auroc = evaluate_ttt_auroc(probe, step_embeddings, label, data["train"],
                                             args.batch_size, device, max_problems=50)
            log.info(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}, "
                     f"train_auroc: {train_auroc:.4f}")
            train_log.append({
                "epoch": epoch + 1, "loss": avg_loss,
                "train_auroc": train_auroc,
            })
        else:
            log.info(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}")
            train_log.append({"epoch": epoch + 1, "loss": avg_loss})

        # checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            probe.save(os.path.join(ckpt_dir, f"probe_ep{epoch+1}.pt"))

    # final evaluation
    final_train_auroc = evaluate_ttt_auroc(probe, step_embeddings, label, data["train"],
                                           args.batch_size, device, max_problems=50)
    log.info(f"Final — train_auroc: {final_train_auroc:.4f}")

    # save
    probe.save(probe_path)
    if probe.scaler is not None:
        with open(os.path.join(run_dir, "pca.pkl"), "wb") as f:
            pickle.dump([probe.scaler, probe.pca], f)

    save_config(args, run_dir)

    metrics = {"train_auroc": final_train_auroc}
    save_metrics(metrics, run_dir)

    # save training log
    with open(os.path.join(run_dir, "train.log"), "w") as f:
        json.dump(train_log, f, indent=2)

    log.info(f"Saved probe to {probe_path}")


def evaluate_ttt_auroc(probe, step_embeddings, label, split_indices, batch_size, device,
                       max_problems=None):
    """Evaluate TTTProbe AUROC by running score→update on each problem."""
    probe.eval()
    all_scores = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for i in split_indices:
            if len(label[i]) == 0:
                continue
            if max_problems is not None and count >= max_problems:
                break

            embeds = step_embeddings[i]
            C_expanded = []
            for lbl in label[i]:
                C_expanded.extend([lbl] * batch_size)
            C_expanded = C_expanded[:len(embeds)]
            if len(C_expanded) == 0:
                continue

            probe.reset()
            for t in range(len(C_expanded)):
                phi_t = torch.tensor(embeds[t], dtype=torch.float32, device=device)
                s_t = probe.score(phi_t).item()
                all_scores.append(s_t)
                all_labels.append(C_expanded[t])
                # update with C_t=0 (inference mode) for evaluation
                probe.update(phi_t, C_t=0.0)

            count += 1

    if len(set(all_labels)) < 2:
        return 0.5  # undefined AUROC
    return roc_auc_score(all_labels, all_scores)


def main():
    args = parse_args()
    with open(args.dataset_path[0], "rb") as f:
        meta = pickle.load(f)
    args.d_phi = meta["embed_dim"]

    if args.method == "base":
        train_base(args)
    elif args.method == "ttt":
        train_ttt(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()
