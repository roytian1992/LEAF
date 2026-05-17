#!/usr/bin/env bash
set -euo pipefail

cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory

CONFIG="tmp/config_agentic_qwen8002_judge.yaml"
PY="./scripts/with_tracenav_nlp.sh python"
OUT_DIR="reports/agentic_memory/judge_20260518"
mkdir -p "${OUT_DIR}" logs

start_tmux() {
  local session="$1"
  shift
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}"
  fi
  tmux new-session -d -s "${session}" "$*"
}

start_tmux leaf_judge_locomo_base_8002_20260518 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   ${PY} scripts/judge_locomo_report.py \
     --config ${CONFIG} \
     --input-report reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json \
     --output-report ${OUT_DIR}/locomo10_base_gpt54mini_qwen8002_judge1_legacybinary_20260518.json \
     --judge-style legacy_binary \
     --judge-runs 1 \
     --judge-retries 3 \
     --judge-max-workers 5 \
     --resume-progress \
     2>&1 | tee logs/judge_locomo_base_qwen8002_20260518.log"

start_tmux leaf_judge_locomo_evolved_8002_20260518 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   ${PY} scripts/judge_locomo_report.py \
     --config ${CONFIG} \
     --input-report reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.json \
     --output-report ${OUT_DIR}/locomo10_evolved_gpt54mini_qwen8002_judge1_legacybinary_20260518.json \
     --judge-style legacy_binary \
     --judge-runs 1 \
     --judge-retries 3 \
     --judge-max-workers 5 \
     --resume-progress \
     2>&1 | tee logs/judge_locomo_evolved_qwen8002_20260518.log"

start_tmux leaf_judge_gvd_en_base_8002_20260518 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   ${PY} scripts/judge_gvd_gold_ref.py \
     --config ${CONFIG} \
     --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json \
     --report reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.json \
     --output ${OUT_DIR}/gvd_en_base_gpt54mini_qwen8002_goldref_judge1_20260518.json \
     --runs 1 \
     --retries 3 \
     --max-workers 5 \
     2>&1 | tee logs/judge_gvd_en_base_qwen8002_20260518.log"

start_tmux leaf_judge_gvd_en_evolved_8002_20260518 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   ${PY} scripts/judge_gvd_gold_ref.py \
     --config ${CONFIG} \
     --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json \
     --report reports/agentic_memory/gvd_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.json \
     --output ${OUT_DIR}/gvd_en_evolved_gpt54mini_qwen8002_goldref_judge1_20260518.json \
     --runs 1 \
     --retries 3 \
     --max-workers 5 \
     2>&1 | tee logs/judge_gvd_en_evolved_qwen8002_20260518.log"

start_tmux leaf_judge_gvd_zh_base_8002_20260518 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   ${PY} scripts/judge_gvd_gold_ref.py \
     --config ${CONFIG} \
     --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json \
     --report reports/agentic_memory/gvd_cn_full100_leafbase_gpt54mini_memory_structctx_20260518.json \
     --output ${OUT_DIR}/gvd_zh_base_gpt54mini_qwen8002_goldref_judge1_20260518.json \
     --runs 1 \
     --retries 3 \
     --max-workers 5 \
     2>&1 | tee logs/judge_gvd_zh_base_qwen8002_20260518.log"

start_tmux leaf_judge_gvd_zh_evolved_8002_20260518 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   ${PY} scripts/judge_gvd_gold_ref.py \
     --config ${CONFIG} \
     --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json \
     --report reports/agentic_memory/gvd_cn_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.json \
     --output ${OUT_DIR}/gvd_zh_evolved_gpt54mini_qwen8002_goldref_judge1_20260518.json \
     --runs 1 \
     --retries 3 \
     --max-workers 5 \
     2>&1 | tee logs/judge_gvd_zh_evolved_qwen8002_20260518.log"

tmux ls | grep 'leaf_judge_.*8002_20260518'
