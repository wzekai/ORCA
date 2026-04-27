#!/usr/bin/env bash
# Download released probes and embeddings from Hugging Face into ./probes and ./data.
#
# Prereqs: huggingface_hub installed (via requirements.txt) and `hf` CLI on PATH.
# The repos are public after the paper release; for the private staging repo,
# run `hf auth login` first.

set -euo pipefail

mkdir -p probes data

echo "Downloading probes (wzekai99/ORCA, model repo)..."
hf download wzekai99/ORCA --local-dir probes

echo "Downloading datasets (wzekai99/ORCA, dataset repo)..."
hf download wzekai99/ORCA --repo-type dataset --local-dir data

echo "Done. Probes in ./probes/, data in ./data/."
