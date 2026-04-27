"""Step 4: Merge all stage outputs into a single dataset.pkl.

Pure CPU, no model needed. Reads the trajectories, embeddings, supervised
labels, and (optionally) consistent labels for one dataset, and writes a
single combined dataset.pkl.

Output formats:

  --release_format=False (default):
      Top-level: model, teacher_model, embed_dim, batch_size,
                 max_step_tokens, splits (training datasets only)
      Per-problem: problem_idx, original_index, question, answer, trajectory,
                   truncated_answers, step_embeddings, step_limits,
                   step_labels, step_labels_consistent

  --release_format:
      Same as default, but drops:
          original_index    (redundant with problem_idx)
          max_step_tokens   (gen-time only, not consumed by training/eval)

  --release_format --strip_text:
      Additionally drops the four plain-text fields:
          question, answer, trajectory, truncated_answers
      Required for GPQA-Diamond, whose upstream license forbids redistributing
      examples in plain text. Other datasets typically retain text under their
      respective upstream licenses.

Usage:
    python 4_merge.py --dataset s1k
    python 4_merge.py --dataset s1k         --release_format
    python 4_merge.py --dataset gpqa_diamond --release_format --strip_text
"""

import argparse
import json
import logging
import os
import pickle

from utils import DATASET_CONFIGS, MODEL_CONFIGS, load_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# Split definitions for each dataset
SPLITS = {
    # S1K: Wu-compatible test (500-549) included in our test split
    "s1k": {
        "splits": {
            "train": list(range(0, 500)) + list(range(550, 650)),
            "calibration": list(range(650, 850)),
            "test": list(range(500, 550)) + list(range(850, 1000)),
        },
        "splits_wu": {
            "train": list(range(0, 500)),
            "calibration": list(range(550, 1000)),
            "test": list(range(500, 550)),
        },
    },
    # Supplementary datasets: 3:1:1 sequential split
    "openr1_2k": {
        "splits": {
            "train": list(range(0, 1200)),
            "calibration": list(range(1200, 1600)),
            "test": list(range(1600, 2000)),
        },
    },
    "deepmath_2k": {
        "splits": {
            "train": list(range(0, 1200)),
            "calibration": list(range(1200, 1600)),
            "test": list(range(1600, 2000)),
        },
    },
    # OOD datasets (aime24/25/26, math500, gpqa_diamond): no splits
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--teacher_model", default="qwen3", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--max_step_tokens", type=int, default=500)
    p.add_argument("--output_dir", default="output")
    p.add_argument("--release_format", action="store_true",
                   help="Drop redundant fields (original_index, max_step_tokens) "
                        "for public release.")
    p.add_argument("--release_path", default=None,
                   help="Write output to this path instead of <output_dir>/<dataset>/dataset.pkl.")
    p.add_argument("--strip_text", action="store_true",
                   help="Additionally drop plain-text fields (question, answer, "
                        "trajectory, truncated_answers). Required for GPQA-Diamond.")
    args = p.parse_args()

    if args.strip_text and not args.release_format:
        p.error("--strip_text requires --release_format")

    dataset_dir = os.path.join(args.output_dir, args.dataset)
    out_path = args.release_path if args.release_path else os.path.join(dataset_dir, "dataset.pkl")
    if args.release_path:
        os.makedirs(os.path.dirname(args.release_path), exist_ok=True)

    # Load intermediate files
    trajectories = load_trajectories(args.dataset, args.output_dir)
    log.info("Loaded %d trajectories", len(trajectories))

    with open(os.path.join(dataset_dir, "truncated_answers.json")) as f:
        truncated_answers = json.load(f)
    log.info("Loaded truncated answers for %d problems", len(truncated_answers))

    with open(os.path.join(dataset_dir, "embeddings.pkl"), "rb") as f:
        embed_data = pickle.load(f)
    embed_problems = embed_data["problems"]
    log.info("Loaded embeddings for %d problems", len(embed_problems))

    with open(os.path.join(dataset_dir, "labels.json")) as f:
        labels_data = json.load(f)
    log.info("Loaded supervised labels for %d problems", len(labels_data))

    consistent_path = os.path.join(dataset_dir, "labels_consistent.json")
    labels_consistent = None
    if os.path.exists(consistent_path):
        with open(consistent_path) as f:
            labels_consistent = json.load(f)
        log.info("Loaded consistent labels for %d problems", len(labels_consistent))

    # Build problems list
    problems = []
    for i, traj in enumerate(trajectories):
        embed = embed_problems[i]
        label = labels_data[i]
        trunc = truncated_answers[i]

        prob = {
            "problem_idx": i,
            "step_embeddings": embed["step_embeddings"],
            "step_limits": embed["step_limits"],
            "step_labels": label["step_labels"],
        }

        if labels_consistent is not None:
            prob["step_labels_consistent"] = labels_consistent[i]["step_labels"]

        if not args.release_format:
            prob["original_index"] = traj.get("problem_idx", i)

        if not args.strip_text:
            prob["question"] = traj["question"]
            prob["answer"] = traj["answer"]
            prob["trajectory"] = traj["trajectory"]
            prob["truncated_answers"] = trunc["answers"]

        problems.append(prob)

    # Build top-level metadata
    dataset = {
        "model": embed_data["model"],
        "teacher_model": MODEL_CONFIGS[args.teacher_model]["hf_id"],
        "embed_dim": embed_data["embed_dim"],
        "batch_size": args.batch_size,
        "problems": problems,
    }

    if not args.release_format:
        dataset["max_step_tokens"] = args.max_step_tokens

    if args.dataset in SPLITS:
        for key, val in SPLITS[args.dataset].items():
            dataset[key] = val

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_steps = sum(len(prob["step_embeddings"]) for prob in problems)
    has_consistent = labels_consistent is not None
    has_splits = "splits" in dataset
    log.info("Saved -> %s (%d problems, %d steps, consistent=%s, splits=%s, "
             "release_format=%s, strip_text=%s)",
             out_path, len(problems), total_steps, has_consistent, has_splits,
             args.release_format, args.strip_text)


if __name__ == "__main__":
    main()
