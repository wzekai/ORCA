"""Step 3: Extract per-token hidden states and mean-pool into step embeddings.

Uses DeepSeek-R1-Distill-Qwen-32B in vLLM pooling mode (token_embed).
Returns post-norm last-layer hidden states for every token, equivalent
to lmdeploy's output_last_hidden_state="all".

Processes prompts in batches to avoid CPU memory OOM.

Usage:
    python 2_embed.py --dataset s1k --model_path /path/to/DeepSeek-32B --tp 4
"""

import argparse
import logging
import os
import pickle

import numpy as np
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM

from utils import (
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    format_prompt,
    get_step_limits,
    load_trajectories,
    separate_steps,
    split_long_steps,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--model_path", required=True)
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max_step_tokens", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=50, help="Prompts per batch to limit memory")
    p.add_argument("--output_dir", default="output")
    args = p.parse_args()

    out_path = os.path.join(args.output_dir, args.dataset, "embeddings.pkl")
    if os.path.exists(out_path):
        log.info("SKIP: %s exists", out_path)
        return

    trajectories = load_trajectories(args.dataset, args.output_dir)
    log.info("Loaded %d trajectories", len(trajectories))

    model_cfg = MODEL_CONFIGS[args.model]
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    embed_dim = config.hidden_size
    assert embed_dim == model_cfg["embed_dim"], (
        f"--model={args.model} expects hidden_size={model_cfg['embed_dim']}, "
        f"but model_path={args.model_path} has hidden_size={embed_dim}"
    )

    # Build all prompts
    prompts = []
    for item in trajectories:
        steps = separate_steps(item["trajectory"])
        prompts.append(format_prompt(item["question"], "\n\n".join(steps), tokenizer))
    log.info("Built %d prompts", len(prompts))

    # Load model
    log.info("Loading model (pooling mode, tp=%d) ...", args.tp)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp, runner="pooling")

    # Process in batches: encode → mean pool → discard raw hidden states
    problems = []
    bs = args.batch_size

    for batch_start in range(0, len(prompts), bs):
        batch_end = min(batch_start + bs, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        outputs = llm.encode(batch_prompts, pooling_task="token_embed")

        for i, output in enumerate(outputs):
            idx = batch_start + i
            item = trajectories[idx]
            hidden_states = output.outputs.data.float().cpu().numpy()
            assert hidden_states.shape[-1] == embed_dim, (
                f"vLLM returned hidden_size={hidden_states.shape[-1]}, expected {embed_dim}"
            )

            raw_limits = get_step_limits(item, tokenizer)
            step_limits = split_long_steps(raw_limits, max_step_tokens=args.max_step_tokens)

            embeddings = []
            for left, right in step_limits:
                segment = hidden_states[left:right]
                if len(segment) > 0:
                    embeddings.append(segment.mean(axis=0))

            step_embeddings = np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, embed_dim), dtype=np.float32)

            problems.append({
                "problem_idx": idx,
                "n_steps": len(step_embeddings),
                "step_embeddings": step_embeddings,
                "step_limits": step_limits,
            })

        log.info("Batch %d-%d done (%d/%d)", batch_start, batch_end, batch_end, len(prompts))

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": model_cfg["hf_id"],
            "embed_dim": embed_dim,
            "problems": problems,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_steps = sum(p["n_steps"] for p in problems)
    log.info("Saved -> %s (%d problems, %d steps)", out_path, len(problems), total_steps)


if __name__ == "__main__":
    main()
