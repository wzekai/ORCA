"""Step 1-2: Generate trajectories and truncated answers.

Uses DeepSeek-R1-Distill-Qwen-32B in vLLM generation mode.
  Step 1: S1K extracts existing trajectories; OOD generates new ones.
  Step 2: Truncates at every batch_size steps, model continues to answer.

Usage:
    python 1_generate.py --dataset s1k --model_path /path/to/DeepSeek-32B --tp 2
"""

import argparse
import json
import logging
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import (
    DATASET_CONFIGS,
    convert,
    generate_truncated_prompts,
    get_step_limits,
    load_dataset_raw,
    separate_steps,
    split_long_steps,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def step1_trajectories(dataset, raw_dir, output_dir, llm, tokenizer):
    """Generate or extract trajectories → trajectories.jsonl."""
    out_path = os.path.join(output_dir, dataset, "trajectories.jsonl")
    if os.path.exists(out_path):
        log.info("Step 1 SKIP: %s exists", out_path)
        return

    data = load_dataset_raw(dataset, raw_dir)
    cfg = DATASET_CONFIGS[dataset]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if cfg["trajectory_key"] is not None:
        # S1K: extract existing trajectories
        with open(out_path, "w") as f:
            for idx, item in enumerate(data):
                record = {
                    "problem_idx": idx,
                    "question": item[cfg["question_key"]],
                    "answer": item[cfg["answer_key"]],
                    "trajectory": item[cfg["trajectory_key"]],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Extracted %d trajectories -> %s", len(data), out_path)
    else:
        # OOD: generate with vLLM
        prompts = [convert(
            f"{item[cfg['question_key']]} Please reason step by step, and put your final answer within \\boxed{{}}.",
            tokenizer,
        ) for item in data]

        log.info("Generating %d trajectories for %s ...", len(prompts), dataset)
        params = SamplingParams(top_p=0.9, temperature=0.6, max_tokens=8192, seed=0)
        results = llm.generate(prompts, params)

        with open(out_path, "w") as f:
            for idx, (item, r) in enumerate(zip(data, results)):
                record = {
                    "problem_idx": idx,
                    "question": item[cfg["question_key"]],
                    "answer": item[cfg["answer_key"]],
                    "trajectory": r.outputs[0].text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Generated %d trajectories -> %s", len(data), out_path)


def step2_truncated_answers(dataset, output_dir, llm, tokenizer, batch_size, max_step_tokens):
    """Generate truncated answers → truncated_answers.json + step_metadata.json."""
    ans_path = os.path.join(output_dir, dataset, "truncated_answers.json")
    if os.path.exists(ans_path):
        log.info("Step 2 SKIP: %s exists", ans_path)
        return

    # Load trajectories
    traj_path = os.path.join(output_dir, dataset, "trajectories.jsonl")
    trajectories = []
    with open(traj_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
    log.info("Loaded %d trajectories", len(trajectories))

    # Build truncated prompts for all problems
    all_prompts = []
    problem_n_prompts = []
    all_step_limits = []
    all_steps_info = []

    for traj in trajectories:
        steps = separate_steps(traj["trajectory"])
        limits = get_step_limits(traj, tokenizer)
        limits = split_long_steps(limits, max_step_tokens=max_step_tokens)
        all_step_limits.append(limits)

        prompts = generate_truncated_prompts(traj, tokenizer, steps, batch_size=batch_size)
        problem_n_prompts.append(len(prompts))
        all_prompts.extend(prompts)

        all_steps_info.append({
            "problem_idx": traj["problem_idx"],
            "n_steps": len(steps),
            "n_step_limits": len(limits),
            "n_truncated_prompts": len(prompts),
        })

    log.info("Total: %d truncated prompts across %d problems", len(all_prompts), len(trajectories))

    # Generate
    params = SamplingParams(top_p=0.9, temperature=0.6, max_tokens=8192, seed=0)
    results = llm.generate(all_prompts, params)
    all_outputs = [r.outputs[0].text for r in results]
    log.info("Generation complete")

    # Group by problem
    truncated_answers = []
    offset = 0
    for pidx, n in enumerate(problem_n_prompts):
        truncated_answers.append({
            "problem_idx": pidx,
            "answers": all_outputs[offset:offset + n],
        })
        offset += n

    # Save
    with open(ans_path, "w") as f:
        json.dump(truncated_answers, f, ensure_ascii=False, indent=2)
    log.info("Saved -> %s", ans_path)

    meta_path = os.path.join(output_dir, dataset, "step_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "steps_info": all_steps_info,
            "step_limits": all_step_limits,
            "batch_size": batch_size,
            "max_step_tokens": max_step_tokens,
        }, f, ensure_ascii=False, indent=2)
    log.info("Saved -> %s", meta_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model_path", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--max_step_tokens", type=int, default=500)
    p.add_argument("--raw_dir", default="raw")
    p.add_argument("--output_dir", default="output")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp)

    step1_trajectories(args.dataset, args.raw_dir, args.output_dir, llm, tokenizer)
    step2_truncated_answers(args.dataset, args.output_dir, llm, tokenizer, args.batch_size, args.max_step_tokens)

    log.info("Done.")


if __name__ == "__main__":
    main()
