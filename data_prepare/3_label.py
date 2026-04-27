"""Step 4: Judge truncated answers using Qwen3-32B teacher model.

Uses Qwen3-32B in vLLM generation mode. For each truncated answer,
asks the teacher whether the student's attempt is correct (Yes/No).

Usage:
    python 3_label.py --dataset s1k --model_path /path/to/Qwen3-32B --tp 2
"""

import argparse
import json
import logging
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import DATASET_CONFIGS, get_prompt_supervised, load_trajectories, parse_yes_no

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model_path", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=10, help="Truncation stride from 1_generate.py")
    p.add_argument("--output_dir", default="output")
    args = p.parse_args()

    out_path = os.path.join(args.output_dir, args.dataset, "labels.json")
    if os.path.exists(out_path):
        log.info("SKIP: %s exists", out_path)
        return

    # Load inputs
    trajectories = load_trajectories(args.dataset, args.output_dir)
    trunc_path = os.path.join(args.output_dir, args.dataset, "truncated_answers.json")
    with open(trunc_path) as f:
        truncated_answers = json.load(f)
    log.info("Loaded %d trajectories, %d truncated answer sets", len(trajectories), len(truncated_answers))

    # Build grading prompts
    all_messages = []
    index_map = []  # (problem_idx, step_idx)

    for prob in truncated_answers:
        pidx = prob["problem_idx"]
        traj = trajectories[pidx]

        for sidx, attempt in enumerate(prob["answers"]):
            msgs = get_prompt_supervised(traj["question"], attempt, traj["answer"])
            all_messages.append(msgs)
            index_map.append((pidx, sidx))

    log.info("Total grading prompts: %d", len(all_messages))

    # Apply Qwen3 chat template (no thinking mode)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    all_prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in all_messages
    ]

    # Run vLLM
    log.info("Loading Qwen3 model (tp=%d) ...", args.tp)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp)
    params = SamplingParams(temperature=0.7, top_p=0.80, min_p=0, top_k=20, max_tokens=4096)

    log.info("Generating ...")
    results = llm.generate(all_prompts, params)
    all_outputs = [r.outputs[0].text for r in results]
    log.info("Generation complete")

    # Parse labels and group by problem
    all_labels = [parse_yes_no(out) for out in all_outputs]

    problem_labels = {}
    for (pidx, sidx), label in zip(index_map, all_labels):
        problem_labels.setdefault(pidx, []).append((sidx, label))

    output_data = []
    for prob in truncated_answers:
        pidx = prob["problem_idx"]
        entries = sorted(problem_labels.get(pidx, []))
        output_data.append({
            "problem_idx": pidx,
            "step_labels": [lbl for _, lbl in entries],
            "batch_size": args.batch_size,
        })

    # Stats
    total = sum(len(d["step_labels"]) for d in output_data)
    yes = sum(sum(d["step_labels"]) for d in output_data)
    log.info("Labels: %d total, %d Yes (%.1f%%), %d No", total, yes, 100 * yes / max(total, 1), total - yes)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
