"""
utils.py — Model definitions + utility functions

Models:
  BaseProbe:  StandardScaler → PCA → LogisticRegression (stateless)
  TTTProbe:   Test-Time Training probe with online adaptation (stateful)

Utilities:
  Data loading, YAML config, calibration (smooth, LTT),
  LayerNorm helpers, run naming, result saving.
"""

import os
import json
import math
import pickle
import yaml
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binom
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_config(args, config):
    """Apply YAML config as defaults — CLI args take priority."""
    # Data paths
    if "data" in config:
        if not args.dataset_path:
            args.dataset_path = config["data"].get("train", [])
        if not args.ood_paths:
            args.ood_paths = config["data"].get("ood", [])

    # Simple scalar fields: only set if not explicitly given on CLI
    defaults = {
        "output_dir": config.get("output_dir"),
        "label_mode": config.get("label_mode"),
        "batch_size": config.get("batch_size"),
        "smooth_window": config.get("smooth_window"),
        "seed": config.get("seed"),
    }
    for key, val in defaults.items():
        if val is not None and getattr(args, key, None) is None:
            setattr(args, key, val)


# ============================================================
# Data loading
# ============================================================

LABEL_FIELDS = {
    "supervised": "step_labels",
    "consistent": "step_labels_consistent",
}


def load_dataset(dataset_path, label_mode="supervised"):
    """Load one dataset.pkl → (step_embeddings, labels, metadata)."""
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    label_field = LABEL_FIELDS[label_mode]
    step_embeddings = []
    labels = []
    for prob in data["problems"]:
        step_embeddings.append(list(prob["step_embeddings"]))
        labels.append(list(prob[label_field]))

    to_cumulative(labels)
    return step_embeddings, labels, data


def load_datasets(dataset_paths, label_mode="supervised"):
    """Load and merge multiple dataset.pkl files.

    Concatenates problems and merges split indices with proper offsets.
    Returns (step_embeddings, labels, metadata) in the same format as load_dataset.
    """
    if isinstance(dataset_paths, str):
        return load_dataset(dataset_paths, label_mode)
    if len(dataset_paths) == 1:
        return load_dataset(dataset_paths[0], label_mode)

    all_embeddings, all_labels = [], []
    merged_splits = {"train": [], "calibration": [], "test": []}
    embed_dim = None
    batch_size = 10
    offset = 0

    for path in dataset_paths:
        step_embeddings, labels, data = load_dataset(path, label_mode)
        all_embeddings.extend(step_embeddings)
        all_labels.extend(labels)

        if embed_dim is None:
            embed_dim = data["embed_dim"]
            batch_size = data.get("batch_size", 10)

        if "splits" in data:
            for key in merged_splits:
                merged_splits[key].extend([i + offset for i in data["splits"].get(key, [])])

        offset += len(step_embeddings)

    metadata = {
        "embed_dim": embed_dim,
        "batch_size": batch_size,
        "splits": merged_splits,
    }
    return all_embeddings, all_labels, metadata


def to_cumulative(labels):
    """Enforce monotonicity: once correct, stays correct. Modifies in-place."""
    for i, lbl in enumerate(labels):
        if 1 not in lbl:
            labels[i] = []
            continue
        first = lbl.index(1)
        for j in range(first, len(lbl)):
            lbl[j] = 1


def get_data_splits(metadata, scheme="default"):
    """Read split indices from dataset metadata."""
    if scheme == "wu" and "splits_wu" in metadata:
        return metadata["splits_wu"]
    return metadata["splits"]


# ============================================================
# Calibration utilities
# ============================================================

def smooth(pred, window=1):
    """Rolling average smoothing."""
    queue = deque()
    pred_smooth = []
    for p in pred:
        queue.append(p)
        if len(queue) > window:
            queue.popleft()
        pred_smooth.append(np.mean(queue))
    return pred_smooth


def binom_p(loss, delta):
    """Binomial p-value for LTT hypothesis test.

    Tests H_j: R(lambda) > delta.
    Under the null, losses ~ Binom(n, delta).
    """
    return binom.cdf(k=np.sum(loss), n=len(loss), p=delta)


def get_loss(pred, true, lam):
    """LTT loss: 0 if stopped at correct step, 1 if incorrect."""
    pred_bin = [1 if p >= lam else 0 for p in pred]
    if 1 not in pred_bin:
        return 1 - true[-1]
    idx = min(pred_bin.index(1), len(true) - 1)
    return 1 - true[idx]


def run_test(preds, trues, delta, epsilon=0.05, bins=10000):
    """Fixed sequence testing: sweep lambda from conservative to aggressive.

    Args:
        delta: target risk upper bound (H_j: R > delta)
        epsilon: FWER confidence level (guarantee holds with prob >= 1-epsilon)
    """
    lambda_range = [1 - i / bins for i in range(bins)]
    lam = lambda_range[-1]
    for lam in lambda_range:
        loss = [get_loss(p, t, lam) for p, t in zip(preds, trues)]
        if binom_p(loss, delta) > epsilon:
            break
    return lam


# ============================================================
# LayerNorm helpers
# ============================================================

def gelu_bwd(x):
    """Analytical derivative of GELU (tanh approximation)."""
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


def ln_fwd(x, gamma, beta, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    return gamma * x_hat + beta


def ln_bwd(x, grad_output, gamma, eps=1e-6):
    """Analytical backward for LayerNorm."""
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    grad_x_hat = grad_output * gamma
    return (
        (1.0 / D)
        * (D * grad_x_hat
           - grad_x_hat.sum(dim=-1, keepdim=True)
           - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True))
        / std
    )


# ============================================================
# BaseProbe
# ============================================================

class BaseProbe:
    """Stateless probe: StandardScaler → PCA → LogisticRegression."""

    def __init__(self, n_components=256, max_iter=5000):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=0)
        self.lr = LogisticRegression(max_iter=max_iter)

    def fit(self, X_train, y_train):
        X = self.scaler.fit_transform(X_train)
        X = self.pca.fit_transform(X)
        self.lr.fit(X, y_train)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        return self.lr.predict_proba(X)[:, 1]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.lr, self.scaler, self.pca], f)

    def load(self, path):
        with open(path, "rb") as f:
            self.lr, self.scaler, self.pca = pickle.load(f)


# ============================================================
# TTTProbe
# ============================================================

class TTTProbe(nn.Module):
    """TTT-Probe: online-adaptive probe with fast weights.

    Two modes:
      - linear (default): fast weight W is [1, d_w], score = sigmoid(W·u + b)
      - mlp: fast weights W1 [d_w, 4*d_w] + W2 [4*d_w, 1], with GELU activation

    Slow weights (meta-learned): theta_K, theta_Q, W0/W10/W20, b0/b10/b20, eta
    Fast weights (updated per-step): W/W1/W2, b/b1/b2
    """

    def __init__(self, d_phi, d_hidden=64,
                 use_ln=False, use_residual=False,
                 learnable_eta=False, base_lr=0.01,
                 share_kq=False, no_kq=False, use_mlp=False,
                 use_pca=False, pca_dim=256):
        super().__init__()
        self.d_phi = d_phi
        self.d_hidden = d_hidden
        self.use_ln = use_ln
        self.use_residual = use_residual
        self.learnable_eta = learnable_eta
        self.base_lr = base_lr
        self.share_kq = share_kq
        self.no_kq = no_kq
        self.use_mlp = use_mlp
        self.use_pca = use_pca
        self.pca_dim = pca_dim

        # PCA preprocessing (fitted externally, not nn.Module)
        self.scaler = None
        self.pca = None

        d_in = pca_dim if use_pca else d_phi
        d_w = d_in if no_kq else d_hidden

        if not no_kq:
            self.theta_K = nn.Linear(d_in, d_hidden, bias=False)
            self.theta_Q = self.theta_K if share_kq else nn.Linear(d_in, d_hidden, bias=False)

        if use_mlp:
            d_mid = 4 * d_w
            self.W10 = nn.Parameter(torch.normal(0, 0.02, size=(d_w, d_mid)))
            self.b10 = nn.Parameter(torch.zeros(1, d_mid))
            self.W20 = nn.Parameter(torch.normal(0, 0.02, size=(d_mid, 1)))
            self.b20 = nn.Parameter(torch.zeros(1))
        else:
            self.W0 = nn.Parameter(torch.normal(0, 0.02, size=(1, d_w)))
            self.b0 = nn.Parameter(torch.zeros(1))

        if learnable_eta:
            self.lr_weight = nn.Parameter(torch.normal(0, 0.02, size=(1, d_in)))
            self.lr_bias = nn.Parameter(torch.zeros(1))
        else:
            self.log_eta = nn.Parameter(torch.tensor(math.log(base_lr)))

        if use_ln:
            self.ln_weight = nn.Parameter(torch.ones(d_w))
            self.ln_bias = nn.Parameter(torch.zeros(d_w))

        # fast weights (set at runtime by reset())
        self.W = self.W1 = self.W2 = self.b = self.b1 = self.b2 = None

    def get_eta(self, phi=None):
        if self.learnable_eta and phi is not None:
            return self.base_lr * torch.sigmoid(F.linear(phi, self.lr_weight) + self.lr_bias)
        return self.log_eta.exp()

    def reset(self):
        if self.use_mlp:
            self.W1 = self.W10.clone()
            self.b1 = self.b10.clone()
            self.W2 = self.W20.clone()
            self.b2 = self.b20.clone()
        else:
            self.W = self.W0.clone()
            self.b = self.b0.clone()

    def preprocess(self, phi):
        """PCA preprocessing (if enabled). Fitted externally during training."""
        if self.use_pca and self.scaler is not None:
            phi_np = phi.detach().cpu().numpy()
            phi_np = self.scaler.transform(phi_np.reshape(1, -1))
            phi_np = self.pca.transform(phi_np)
            return torch.tensor(phi_np.squeeze(0), dtype=phi.dtype, device=phi.device)
        return phi

    def score(self, phi_t):
        phi_t = self.preprocess(phi_t)
        u_Q = phi_t if self.no_kq else self.theta_Q(phi_t)
        if self.use_mlp:
            h = F.gelu(u_Q @ self.W1 + self.b1, approximate="tanh")
            logit = (h @ self.W2 + self.b2).squeeze(-1)
        else:
            logit = (self.W * u_Q).sum() + self.b
        return torch.sigmoid(logit)

    def update(self, phi_t, C_t=0.0):
        phi_t = self.preprocess(phi_t)
        u_K = phi_t if self.no_kq else self.theta_K(phi_t)
        eta = self.get_eta(phi_t if self.learnable_eta else None)

        if self.use_mlp:
            # Forward
            Z1 = u_K @ self.W1 + self.b1        # [1, 4*d_w]
            X2 = F.gelu(Z1, approximate="tanh")  # [1, 4*d_w]
            Z2 = X2 @ self.W2 + self.b2          # [1, 1]
            pred = torch.sigmoid(Z2.squeeze(-1))

            # Analytical backward
            d_Z2 = 2.0 * (pred - C_t) * pred * (1.0 - pred)  # scalar
            grad_W2 = X2.T * d_Z2                              # [4*d_w, 1]
            grad_b2 = d_Z2
            d_X2 = d_Z2 * self.W2.T                            # [1, 4*d_w]
            d_Z1 = d_X2 * gelu_bwd(Z1)                         # [1, 4*d_w]
            grad_W1 = u_K.unsqueeze(-1) @ d_Z1                   # [d_w, 1] @ [1, 4*d_w] → [d_w, 4*d_w]
            grad_b1 = d_Z1                                      # [1, 4*d_w]

            self.W1 = self.W1 - eta * grad_W1
            self.b1 = self.b1 - eta * grad_b1
            self.W2 = self.W2 - eta * grad_W2
            self.b2 = self.b2 - eta * grad_b2

        elif self.use_ln:
            Z = self.W * u_K
            Z_ln = ln_fwd(Z, self.ln_weight, self.ln_bias)
            Z_out = Z + Z_ln if self.use_residual else Z_ln
            logit = Z_out.sum() + self.b
            pred = torch.sigmoid(logit)

            d_logit = 2.0 * (pred - C_t) * pred * (1.0 - pred)
            d_Z_out = d_logit.expand_as(Z_out)
            if self.use_residual:
                d_Z = d_Z_out + ln_bwd(Z, d_Z_out, self.ln_weight)
            else:
                d_Z = ln_bwd(Z, d_Z_out, self.ln_weight)

            self.W = self.W - eta * (d_Z * u_K)
            self.b = self.b - eta * d_logit

        else:
            logit = (self.W * u_K).sum() + self.b
            pred = torch.sigmoid(logit)
            d_logit = 2.0 * (pred - C_t) * pred * (1.0 - pred)

            self.W = self.W - eta * (d_logit * u_K)
            self.b = self.b - eta * d_logit

    def forward_trajectory(self, phi_seq, C_seq):
        """Process full reasoning chain: score-then-update for each step."""
        self.reset()
        scores = []
        device = self.W10.device if self.use_mlp else self.W0.device
        outer_loss = torch.tensor(0.0, device=device)

        for t in range(len(phi_seq)):
            s_t = self.score(phi_seq[t])
            scores.append(s_t)
            outer_loss = outer_loss + (s_t - C_seq[t]) ** 2
            self.update(phi_seq[t], C_t=0.0)

        return scores, outer_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))


# ============================================================
# Run naming & result saving
# ============================================================

def make_run_name(args):
    """Generate run directory name from args."""
    if args.method == "base":
        return "base"

    parts = ["ttt"]
    if getattr(args, "no_kq", False):
        parts.append("no_kq")
    else:
        parts.append("dh%d" % args.d_hidden)
    parts.append("lr%s" % args.base_lr)
    if args.use_ln:
        parts.append("ln")
    if args.use_residual:
        parts.append("res")
    if args.learnable_eta:
        parts.append("eta_learn")
    if args.share_kq:
        parts.append("share_kq")
    if getattr(args, "use_mlp", False):
        parts.append("mlp")
    if getattr(args, "use_pca", False):
        parts.append("pca%d" % getattr(args, "pca_dim", 256))
    if getattr(args, "no_meta_train", False):
        parts.append("no_meta")
    if getattr(args, "no_online_update", False):
        parts.append("no_update")
    if getattr(args, "epochs", 100) != 100:
        parts.append("ep%d" % args.epochs)
    if getattr(args, "outer_lr", 1e-3) != 1e-3:
        parts.append("olr%s" % args.outer_lr)
    if getattr(args, "smooth_window", 10) != 10:
        parts.append("sw%d" % args.smooth_window)
    return "__".join(parts)


def get_run_dir(args):
    """Get or create the run directory: output_dir/label_mode/run_name."""
    run_name = getattr(args, "run_name", None) or make_run_name(args)
    label_mode = getattr(args, "label_mode", "supervised")
    run_dir = os.path.join(args.output_dir, label_mode, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(args, run_dir):
    """Save full hyperparameter snapshot."""
    config = vars(args).copy()
    config["run_name"] = os.path.basename(run_dir)
    config["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


def save_metrics(metrics, run_dir):
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def append_summary(run_dir, metrics, args, summary_path=None):
    """Append one row to summary.csv."""
    import pandas as pd

    if summary_path is None:
        summary_path = os.path.join(args.output_dir, "summary.csv")

    row = {
        "run_name": os.path.basename(run_dir),
        "method": args.method,
        "label_mode": getattr(args, "label_mode", "supervised"),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
    }

    if args.method == "ttt":
        row.update({
            "d_hidden": args.d_hidden,
            "base_lr": args.base_lr,
            "use_ln": args.use_ln,
            "learnable_eta": args.learnable_eta,
            "no_kq": getattr(args, "no_kq", False),
        })

    for k in ["train_auroc"]:
        if k in metrics:
            row[k] = metrics[k]

    if "eps_results" in metrics:
        for eps_str, res in metrics["eps_results"].items():
            eps_key = eps_str.replace(".", "")
            for metric_name in ["error_rate", "savings"]:
                if metric_name in res:
                    row["eps_%s_%s" % (eps_key, metric_name)] = res[metric_name]

    df_new = pd.DataFrame([row])
    if os.path.exists(summary_path):
        df_old = pd.read_csv(summary_path)
        df_old = df_old[df_old["run_name"] != row["run_name"]]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        df = df_new
    df.to_csv(summary_path, index=False)


# ============================================================
# Shared argparse helpers
# ============================================================

def add_common_args(parser):
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--method", type=str, choices=["base", "ttt"])
    parser.add_argument("--dataset_path", type=str, nargs="+",
                        help="Path(s) to dataset.pkl; multiple are merged")
    parser.add_argument("--ood_paths", type=str, nargs="*", default=[])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--label_mode", type=str, default=None, choices=list(LABEL_FIELDS.keys()))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smooth_window", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)


def add_ttt_args(parser):
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--use_ln", action="store_true")
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--learnable_eta", action="store_true")
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--share_kq", action="store_true")
    parser.add_argument("--use_mlp", action="store_true", help="2-layer MLP fast weight (TTT-MLP)")
    parser.add_argument("--use_pca", action="store_true", help="PCA preprocessing before TTT")
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--outer_lr", type=float, default=1e-3)
    parser.add_argument("--no_meta_train", action="store_true")
    parser.add_argument("--no_online_update", action="store_true")
    parser.add_argument("--no_kq", action="store_true")


def parse_args_with_config(parser):
    """Parse CLI args, then apply YAML config as defaults."""
    args, _ = parser.parse_known_args()

    if args.config:
        config = load_config(args.config)
        apply_config(args, config)

    # Fill remaining defaults
    if args.output_dir is None:
        args.output_dir = "results"
    if args.label_mode is None:
        args.label_mode = "supervised"
    if args.batch_size is None:
        args.batch_size = 10
    if args.seed is None:
        args.seed = 42
    if args.smooth_window is None:
        args.smooth_window = 10

    return args


def build_ttt_probe(args, device="cpu"):
    probe = TTTProbe(
        d_phi=args.d_phi,
        d_hidden=args.d_hidden,
        use_ln=args.use_ln,
        use_residual=args.use_residual,
        learnable_eta=args.learnable_eta,
        base_lr=args.base_lr,
        share_kq=args.share_kq,
        no_kq=getattr(args, "no_kq", False),
        use_mlp=getattr(args, "use_mlp", False),
        use_pca=getattr(args, "use_pca", False),
        pca_dim=getattr(args, "pca_dim", 256),
    ).to(device)
    return probe
