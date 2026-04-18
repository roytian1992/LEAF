# Experiment Log - 2026-04-18

## LoCoMo dynamic-tree fresh full cold-start

- method: `dynamic-tree`
- objective: verify the current `dynamic-tree2` ingest + search path on `LoCoMo` with a true `10/10` cold-start DB instead of the historical `reuse 3 + new 7` mixed ingest run
- release / input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- future novelty cleanup applied: `N/A` for LoCoMo memory benchmark
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- branch: `dynamic-tree2`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
  - `answer_view_mode = heuristic`
  - `answer_style = short`
  - `heuristic_bypass = disabled`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshfull_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.qa_progress.jsonl`
- tmux session: `locomo_freshfull_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_freshfull_20260418_tmux.log`

Launch command:

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF
stdbuf -oL -eL /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo.py \
  --config examples/config_local.yaml \
  --db data/locomo10_dynamic_tree_freshfull_20260418.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/locomo10_dynamic_tree_freshfull_20260418.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --answer-view-mode heuristic \
  --answer-style short \
  --ingest-prepare-workers 4 \
  --disable-heuristic-bypass
```

Notes before launch:

- This run intentionally uses a fresh DB path so every sample must ingest from scratch.
- It is meant to refresh the `Ingest Baseline` row for `LEAF dynamic-tree` in `LEAF_dev/docs/leaf/locomo_results.md`.
- Current uncommitted repo changes are unrelated to LoCoMo runtime:
  - `scripts/eval_gvd.py`
  - `docs/experiment_log_20260416.md`

## LoCoMo dynamic-tree fresh full resume from conv-48

- method: `dynamic-tree`
- objective: resume the interrupted `2026-04-18` fresh cold-start run after a remote LLM disconnect during `conv-48` ingest, without redoing the already completed `conv-26/30/41/42/43/44/47`
- source partial run report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- source partial DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshfull_20260418.sqlite3`
- remaining input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_remaining_conv48_49_50_20260418.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
  - `answer_view_mode = heuristic`
  - `answer_style = short`
  - `heuristic_bypass = disabled`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshfull_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_resume48_50_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_resume48_50_20260418.qa_progress.jsonl`
- tmux session: `locomo_freshfull_resume48_50_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_freshfull_resume48_50_20260418_tmux.log`

Failure note:

- The original freshfull run stopped at `conv-48 ingest_start` with `http.client.RemoteDisconnected: Remote end closed connection without response` from the memory-side LLM call in atom extraction.

## LoCoMo dynamic-tree fresh full resume from conv-48 with memory qwen8002

- method: `dynamic-tree`
- objective: continue the remaining `conv-48/49/50` ingest + QA after repeated `gpt-4.1-mini` memory-side disconnects, by switching only `additional_llm` to local `Qwen3-235B@8002`
- source partial run report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- source partial DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshfull_20260418.sqlite3`
- remaining input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_remaining_conv48_49_50_20260418.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local_memory_qwen8002.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
  - `answer_view_mode = heuristic`
  - `answer_style = short`
  - `heuristic_bypass = disabled`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshfull_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_resume48_50_qwen8002_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_resume48_50_qwen8002_20260418.qa_progress.jsonl`
- tmux session: `locomo_freshfull_resume48_50_qwen8002_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_freshfull_resume48_50_qwen8002_20260418_tmux.log`

Check before launch:

- local probe against `additional_llm = Qwen3-235B@8002` returned `OK`

## LoCoMo dynamic-tree last3 rerun with recovered gpt-4.1-mini

- method: `dynamic-tree`
- objective: rerun the remaining `conv-48/49/50` on a clean 3-sample DB after confirming `gpt-4.1-mini` has recovered, then recombine with the earlier 7-sample partial freshfull results
- source 7-sample partial report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- remaining input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_remaining_conv48_49_50_20260418.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
  - `answer_view_mode = heuristic`
  - `answer_style = short`
  - `heuristic_bypass = disabled`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo_last3_gptmini_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_last3_gptmini_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_last3_gptmini_20260418.qa_progress.jsonl`
- tmux session: `locomo_last3_gptmini_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_last3_gptmini_20260418_tmux.log`

Recovery check before launch:

- direct `curl -m 15` probe to `gpt-4.1-mini` returned a valid chat completion with content `OK`

## LoCoMo dynamic-tree last3 rerun with answer gpt-4.1-mini + memory qwen8002

- method: `dynamic-tree`
- objective: rerun the remaining `conv-48/49/50` with memory-side ingest on local `Qwen3-235B@8002`, while keeping QA answer generation on `gpt-4.1-mini`, then recombine with the earlier 7-sample partial freshfull results
- source 7-sample partial report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- remaining input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_remaining_conv48_49_50_20260418.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local_answer_gptmini_memory_qwen8002.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
  - `answer_view_mode = heuristic`
  - `answer_style = short`
  - `heuristic_bypass = disabled`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo_last3_answergptmini_memoryqwen8002_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_last3_answergptmini_memoryqwen8002_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_last3_answergptmini_memoryqwen8002_20260418.qa_progress.jsonl`
- tmux session: `locomo_last3_answergptmini_memoryqwen8002_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_last3_answergptmini_memoryqwen8002_20260418_tmux.log`

Role split:

- `llm = gpt-4.1-mini` for QA answer generation
- `additional_llm = Qwen3-235B@8002` for memory-side atom extraction and reconciliation

## LoCoMo dynamic-tree split rerun with gpt-4.1-mini memory

- method: `dynamic-tree`
- objective: isolate whether the regression is mainly from `conv-48` by rerunning `conv-49/50` first with the original `gpt-4.1-mini` memory-side configuration, then rerunning `conv-48` alone with lower ingest parallelism
- source 7-sample partial report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`

### Phase A: conv-49/50

- input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_conv49_50_20260418.json`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo_conv49_50_gptmini_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_conv49_50_gptmini_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_conv49_50_gptmini_20260418.qa_progress.jsonl`
- tmux session: `locomo_conv49_50_gptmini_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_conv49_50_gptmini_20260418_tmux.log`
- ingest prepare workers: `4`

### Phase B: conv-48 only

- input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_conv48_only_20260418.json`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo_conv48_gptmini_w2_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_conv48_gptmini_w2_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_conv48_gptmini_w2_20260418.qa_progress.jsonl`
- tmux session: `locomo_conv48_gptmini_w2_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_conv48_gptmini_w2_20260418_tmux.log`
- ingest prepare workers: `2`

## LoCoMo dynamic-tree last3 rerun with gpt-4.1-mini retry3

- method: `dynamic-tree`
- objective: rerun `conv-48/49/50` from scratch after adding up to `3` network retries for memory/answer HTTP calls, keeping the original `gpt-4.1-mini` memory-side configuration and `4` ingest prepare workers
- source 7-sample partial report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260418.json`
- input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_remaining_conv48_49_50_20260418.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- code change: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/src/leaf/clients.py`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo_last3_gptmini_retry3_20260418.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_last3_gptmini_retry3_20260418.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo_last3_gptmini_retry3_20260418.qa_progress.jsonl`
- tmux session: `locomo_last3_gptmini_retry3_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_last3_gptmini_retry3_20260418_tmux.log`
- ingest prepare workers: `4`

## LoCoMo dynamic-tree fresh full merged Judge-5

- method: `dynamic-tree`
- objective: rerun `Judge-5` on the merged `10/10` fresh cold-start LoCoMo report using binary `legacy_binary` scoring, explicitly not using the `0-0.5-1` partial-credit path
- merged input report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_merged_retry3_20260418.json`
- judge output report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_merged_retry3_judge5_legacybinary_20260418.json`
- judge progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_merged_retry3_judge5_legacybinary_20260418.judge_progress.jsonl`
- tmux session: `locomo_judge5_20260418`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_merged_retry3_judge5_legacybinary_20260418.log`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- judge settings:
  - `judge_style = legacy_binary`
  - `judge_runs = 5`
  - `judge_max_workers = 8`
  - `judge_retries = 3`

Results:

- overall `Judge-5 mean = 0.7008`
- overall `Judge-5 std = 0.0006`
- per-run scores: `0.7013, 0.7013, 0.7000, 0.7000, 0.7013`
- `judge_count = 1540`
