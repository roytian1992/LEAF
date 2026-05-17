#!/usr/bin/env bash
set -euo pipefail

ROOT=/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
ENV_PREFIX=/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/tracenav_nlp

export PYTHONNOUSERSITE=1
export NLTK_DATA=/vepfs-mlp2/c20250513/241404044/users/roytian/nltk_data
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

exec "$@"
