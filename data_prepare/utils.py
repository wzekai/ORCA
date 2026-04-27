"""Shared utilities for data preparation pipeline."""

import json
import os

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "s1k": {
        "question_key": "question",
        "answer_key": "solution",
        "trajectory_key": "deepseek_thinking_trajectory",
    },
    "aime24": {
        "question_key": "Problem",
        "answer_key": "Answer",
        "trajectory_key": None,
    },
    "aime25": {
        "question_key": "Problem",
        "answer_key": "Answer",
        "trajectory_key": None,
    },
    "aime26": {
        "question_key": "Problem",
        "answer_key": "Answer",
        "trajectory_key": None,
    },
    "math500": {
        "question_key": "problem",
        "answer_key": "answer",
        "trajectory_key": None,
    },
    "gpqa_diamond": {
        "question_key": "Question",
        "answer_key": "Correct Answer",
        "trajectory_key": None,
    },
    "openr1_2k": {
        "question_key": "Problem",
        "answer_key": "Answer",
        "trajectory_key": "Trajectory",
    },
    "deepmath_2k": {
        "question_key": "Problem",
        "answer_key": "Answer",
        "trajectory_key": "Trajectory",
    },
}

MODEL_CONFIGS = {
    "qwen2.5": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "embed_dim": 5120,
    },
    "qwen3": {
        "hf_id": "Qwen/Qwen3-32B",
    },
    "llama3.3": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "embed_dim": 8192,
    },
    "qwq": {
        "hf_id": "Qwen/QwQ-32B-Preview",
        "embed_dim": 5120,
    },
    "deepseek_math_rl": {
        "hf_id": "deepseek-ai/deepseek-math-7b-rl",
        "embed_dim": 4096,
    },
}

GRADING_PROMPT = """You are an AI assistant for grading a science problem. The user will provide you with the question itself, the correct answer, and the student's attempt. Your job is to judge whether the attempt is correct by comparing it with the correct answer. If the correct answer is a number or choice, there should be no ambiguity, and you should directly compare the answer and the final result. If the attempt is incomplete, you should mark it as wrong. If the correct answer involves going through the entire reasoning process, you should judge the result based on whether the reasoning process is correct, compared to correct answer.

Do NOT try to solve the problem yourself. Only grade the attempt based on the correct answer.

The user will provide the attempt and the correct answer in the following format:

# Problem
{problem}

## Correct answer
{solution}

## Student attempt
{attempt}

Explain your reasoning concisely, and end your response on a new line with only "Yes" or "No" (without quotes).
"""

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_dataset_raw(dataset_name, raw_dir="raw"):
    return load_jsonl(os.path.join(raw_dir, f"{dataset_name}.jsonl"))


def load_trajectories(dataset_name, output_dir="output"):
    return load_jsonl(os.path.join(output_dir, dataset_name, "trajectories.jsonl"))


# ---------------------------------------------------------------------------
# Step splitting (Wu et al.)
# ---------------------------------------------------------------------------


def separate_steps(thoughts, delims=("wait", "Wait", "but", "But")):
    """Split thinking trajectory into steps at delimiter lines."""
    steps = [""]
    for line in thoughts.split("\n"):
        if not line:
            continue
        line = line + "\n"
        if any(d in line for d in delims):
            steps.append(line)
        else:
            steps[-1] += line
    return [s.strip() for s in steps]


def split_long_steps(limits, max_step_tokens=500):
    """Split steps exceeding max_step_tokens into fixed-size chunks."""
    result = []
    for left, right in limits:
        if right - left <= max_step_tokens:
            result.append((left, right))
        else:
            for start in range(left, right, max_step_tokens):
                end = min(start + max_step_tokens, right)
                result.append((start, end))
    return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def convert(messages, tokenizer):
    """Apply chat template to messages."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_prompt(question, thoughts, tokenizer):
    """Build full prompt: chat template + <think> block."""
    prompt = convert(
        f"{question} Please reason step by step, and put your final answer within \\boxed{{}}.",
        tokenizer,
    )
    return f"""{prompt}

<think>

{thoughts}

</think>

Final Answer:
"""


def generate_truncated_prompts(item, tokenizer, steps, batch_size=10):
    """Generate progressively longer prompts, truncated every batch_size steps."""
    prompts = []
    for i in range(0, len(steps), batch_size):
        thoughts = "\n\n".join(steps[:i + batch_size])
        prompt = format_prompt(item["question"], thoughts, tokenizer)
        if prompt.count("<think>\n\n") > 1:
            prompt = prompt.replace("<think>\n\n", "", 1)
        prompts.append(prompt)
    return prompts


def get_prompt_supervised(question, attempt, solution):
    """Build grading prompt for teacher model."""
    user_prompt = (
        f"## Problem\n{question}\n\n"
        f"## Correct answer\n{solution}\n\n"
        f"## Student attempt\n{attempt}\n"
    )
    return [
        {"role": "system", "content": GRADING_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Consistent / Novel / Leaf label prompts (Wu et al. Appendix A)
# ---------------------------------------------------------------------------

CONSISTENCY_GRADING_PROMPT = """You are an AI assistant for grading a science problem. The user will provide you with the question itself and two student attempts. Your job is to judge whether the two students arrive at the same answer. If question asks for a single numerical answer, there should be no ambiguity, and you should directly compare the two answers. If the question asks for multiple parts, the two attempts are identical if only if all of the parts arrive at the same conclusion.

Do NOT try to solve the problem yourself. Only grade whether the two attempts are the same.

The user will provide the problem and two attempts in the following format:

# Problem

{problem}

## Attempt 1

{attempt1}

## Attempt 2

{attempt2}

Explain your reasoning concisely, and end your response on a new line with only "Yes" or "No" (without quotes).
"""

NOVELTY_GRADING_PROMPT = """You are an AI assistant for assessing the quality of logical reasoning. The user will provide you with the question and an incomplete attempt, consisting of a series of reasoning steps. Your job is to judge whether current step appears to provide additional information, compared to the previous steps. If the current step is correct and novel, it is useful. If the current step is wrong or redundant, then it is not useful.

Do NOT try to solve the problem yourself. It does not matter if the attempt is not complete. Only comment on whether the current step is useful.

The user will provide the problem and reasoning steps in the following format:

# Problem

{problem}

# Reasoning

## step 1
{reasoning step 1}

## step 2
{reasoning step 2}

...

## step k
{reasoning step k}

Explain your reasoning, and end your response on a new line with only "Yes" if the current step provides new information or "No" otherwise (without quotes).
"""

LEAF_GRADING_PROMPT = """You are an AI assistant for parsing LLM outputs. The user will provide you with the question and an intermediate reasoning step. Your job is to judge whether the given step contains an attempt at a final answer.

Do NOT attempt to solve the problem yourself. It does not matter if the answer is correct. Only comment on whether an attempt has been made.

The user will provide the problem and reasoning step in the following format:

# Problem

{problem}

# Reasoning step

{reasoning step}

Explain your reasoning, and end your response on a new line with only "Yes" or "No" indicating whether the given step makes an attempt at providing the final answer.
"""


def parse_yes_no(text):
    """Parse grading output. Returns 1 for Yes, 0 otherwise."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip().lower()
        if line == "yes":
            return 1
        if line == "no":
            return 0
        if line:
            break
    return 0


def get_prompt_consistent(question, attempt, reference_attempt):
    """Build consistency grading prompt: do two attempts reach the same answer?"""
    user_prompt = (
        f"## Problem\n{question}\n\n"
        f"## Attempt 1\n\n{attempt}\n\n"
        f"## Attempt 2\n\n{reference_attempt}\n"
    )
    return [
        {"role": "system", "content": CONSISTENCY_GRADING_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def get_prompt_novelty(question, steps, current_idx):
    """Build novelty prompt: does the last step provide new information?

    Includes all steps up to current_idx as context.
    """
    formatted = [f"## step {i+1}\n{s}" for i, s in enumerate(steps[:current_idx + 1])]
    thoughts = "\n\n".join(formatted)
    user_prompt = (
        f"# Problem\n{question}\n\n"
        f"# Reasoning\n{thoughts}\n\n"
        f"Explain your reasoning, and end your response on a new line with only "
        f"\"Yes\" if the current step provides new information or \"No\" otherwise (without quotes)."
    )
    return [
        {"role": "system", "content": NOVELTY_GRADING_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def get_prompt_leaf(question, step):
    """Build leaf detection prompt: does this step attempt a final answer?"""
    user_prompt = (
        f"# Problem\n\n{question}\n\n"
        f"# Reasoning step\n\n{step}\n\n"
        f"Explain your reasoning, and end your response on a new line with only "
        f"\"Yes\" or \"No\" indicating whether the given step makes an attempt "
        f"at providing the final answer."
    )
    return [
        {"role": "system", "content": LEAF_GRADING_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Token-level step boundaries (BPE-aligned)
# ---------------------------------------------------------------------------


def get_step_limits(item, tokenizer):
    """Compute token-level step boundaries using offset_mapping.

    Uses full tokenization (same as prefill) so boundaries are exact.
    Replaces Wu's progressive tokenization which had 2-11 token offset.
    """
    steps = separate_steps(item["trajectory"])
    full_thoughts = "\n\n".join(steps)
    full_prompt = format_prompt(item["question"], full_thoughts, tokenizer)

    encoding = tokenizer(full_prompt, return_offsets_mapping=True)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    think_pos = full_prompt.find("<think>\n\n")
    current_char = think_pos + len("<think>\n\n")

    def char_to_token(char_pos):
        for t_idx, (cs, ce) in enumerate(offsets):
            if cs <= char_pos < ce:
                return t_idx
            if cs >= char_pos:
                return t_idx
        return len(tokens)

    limits = []
    for i, step in enumerate(steps):
        start_tok = char_to_token(current_char)
        end_tok = char_to_token(current_char + len(step))
        limits.append((start_tok, end_tok))
        current_char += len(step)
        if i < len(steps) - 1:
            current_char += 2  # "\n\n" separator

    return limits
