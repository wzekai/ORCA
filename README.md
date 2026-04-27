# ORCA: Online Reasoning Calibration

Code, probes, and embeddings for *Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning* ([arXiv:2604.01170](https://arxiv.org/abs/2604.01170)).

## Released artifacts

- **Probes** (17 trained TTT-Probes): <https://huggingface.co/wzekai99/ORCA>
- **Datasets** (step embeddings + labels for 3 LLMs): <https://huggingface.co/datasets/wzekai99/ORCA>

## Install

```bash
git clone https://github.com/wzekai/ORCA.git
cd ORCA

# Install uv if missing:
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
```

To re-extract embeddings from a different LLM (vLLM-based), follow [`data_prepare/README.md`](data_prepare/README.md) instead.

## Reproduce paper results

```bash
bash scripts/download_artifacts.sh        # downloads probes/ and data/
bash scripts/reproduce_main.sh            # Tables 1, 2, sup_con, dhidden, ablation_arch (Qwen2.5-32B)
bash scripts/reproduce_cross_model.sh     # Table 3 (QwQ-32B, Llama-3.3-70B)
```

Each probe writes `metrics.json` under `results/<label>__<probe>/<label>/<run_name>/`.

## Train from scratch

```bash
bash scripts/train_from_scratch.sh
```

Edit the variables at the top to select LLM, label mode (supervised / consistent), and architecture variant (no_kq / qk).

## Repository layout

```
code/            utils.py, train.py, calibrate.py, test.py,
                 epoch_sweep.py, compute_token_savings.py
configs/         qwen32b.yaml, qwq32b.yaml, llama70b.yaml
data_prepare/    Trajectory generation, embedding, labeling, merging (vLLM)
scripts/         download_artifacts.sh, reproduce_*.sh, train_from_scratch.sh
```

## License

Code and probes: MIT. Datasets: CC-BY-4.0 with upstream attribution.

## Citation

```bibtex
@article{zhou2026online,
  title={Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning},
  author={Zhou, Cai and Wang, Zekai and Wu, Menghua and Zhu, Qianyu Julie and Shi, Flora C and Wang, Chenyu and Wilson, Ashia and Jaakkola, Tommi and Bates, Stephen},
  journal={arXiv preprint arXiv:2604.01170},
  year={2026}
}
```
