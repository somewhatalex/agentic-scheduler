#!/usr/bin/env bash
# Set up a CloudLab node: Python venv on the persistent dataset, then download
# the raw Nemotron-SFT-Agentic-v2 data. Safe to re-run; finished steps are skipped.
#
# Usage: bash scripts/setup.sh  (needs /mydata mounted read-write)
set -euo pipefail

DATA_ROOT=/mydata
VENV="$DATA_ROOT/venv"
RAW_DIR="$DATA_ROOT/raw/nemotron"
HF_REPO=nvidia/Nemotron-SFT-Agentic-v2

export HF_HOME="$DATA_ROOT/hf"

if ! mountpoint -q "$DATA_ROOT"; then
    echo "error: $DATA_ROOT is not mounted; is the dataset attached?" >&2
    exit 1
fi

# The blockstore filesystem is created owned by root.
if [ ! -w "$DATA_ROOT" ]; then
    sudo chown "$(id -u):$(id -g)" "$DATA_ROOT" || {
        echo "error: $DATA_ROOT is not writable; instantiate with readonly unchecked" >&2
        exit 1
    }
fi
mkdir -p "$HF_HOME" "$RAW_DIR" "$DATA_ROOT/processed"

if ! python3 -c 'import ensurepip, venv' 2>/dev/null; then
    sudo apt-get update -q
    sudo apt-get install -y -q python3-venv
fi

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q huggingface_hub datasets pyarrow pandas tiktoken

# hf download resumes partial downloads, so the marker only records a full success.
if [ -f "$RAW_DIR/.download_complete" ]; then
    echo "raw data already present in $RAW_DIR, skipping download"
else
    "$VENV/bin/hf" download "$HF_REPO" --repo-type dataset --local-dir "$RAW_DIR"
    touch "$RAW_DIR/.download_complete"
fi

echo
echo "done. activate the venv with: source $VENV/bin/activate"
echo "HF_HOME is only set inside this script; run 'export HF_HOME=$HF_HOME' in your shell too."
