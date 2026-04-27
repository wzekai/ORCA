# Data preparation pipeline

Four-stage pipeline that produces the `<dataset>.pkl` files released as `wzekai99/ORCA` (dataset). Use this only to re-extract embeddings from a different LLM. To reproduce paper results from released artifacts, see the main `README.md`.

## Stages

| Script                  | Output                                  | External model        |
|-------------------------|-----------------------------------------|-----------------------|
| `1_generate.py`         | Reasoning trajectories                  | DeepSeek-R1-671B      |
| `2_embed.py`            | Per-step mean-pooled hidden states      | Target LLM            |
| `3_label.py`            | Supervised labels (teacher correctness) | Qwen3-32B teacher     |
| `3_label_consistent.py` | Consistent labels (label-free)          | Qwen3-32B teacher     |
| `4_merge.py`            | Combined `dataset.pkl`                  | (CPU only)            |

`utils.py` holds the shared dataset and model registries.

## Hardware

One GPU with at least 80 GB memory (H100, H200, or A100-80G) is required for serving 32B and 70B models with vLLM.

## Install

vLLM bundles its own PyTorch and CUDA runtime; install it first.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install vllm --torch-backend=auto       # provides torch + CUDA
uv pip install -r ../requirements-data.txt     # transformers, numpy, etc.
```

See <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/> for vLLM platform requirements.

## Run

Each stage takes a `--dataset` (one of `s1k`, `openr1_2k`, `deepmath_2k`, `math500`, `gpqa_diamond`, `aime24`, `aime25`, `aime26`). Stages 1-3 write intermediate JSON / pickle files under `output/<llm>/<dataset>/`; stage 4 writes the final `dataset.pkl`.

```bash
python 1_generate.py --dataset s1k --model deepseek_r1
python 2_embed.py    --dataset s1k --model qwen32b
python 3_label.py    --dataset s1k --teacher_model qwen3 --output_dir output/qwen32b
python 3_label_consistent.py --dataset s1k --teacher_model qwen3 --output_dir output/qwen32b

# Default merge (full intermediate format)
python 4_merge.py --dataset s1k --output_dir output/qwen32b

# Public-release format (drops original_index, max_step_tokens):
python 4_merge.py --dataset s1k --output_dir output/qwen32b \
    --release_format \
    --release_path /path/to/release/qwen2.5-32b/s1k.pkl

# GPQA-Diamond requires --strip_text (no plain text per upstream license):
python 4_merge.py --dataset gpqa_diamond --output_dir output/qwen32b \
    --release_format --strip_text \
    --release_path /path/to/release/qwen2.5-32b/gpqa_diamond.pkl
```

## Output format

A `dataset.pkl` is a Python pickle containing a top-level dict with keys
`model`, `teacher_model`, `embed_dim`, `batch_size`, `splits` (training datasets only), and `problems` (a list of per-problem dicts). See the dataset card schema for the full field list.
