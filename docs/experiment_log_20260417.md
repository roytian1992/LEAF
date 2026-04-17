## 2026-04-17

### GVD CN gold draft build

- task: build the first Chinese GVD gold-reference draft
- code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/scripts/build_gvd_gold.py`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.gpt41mini_bgem3.yaml`
- source memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- source questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl`
- source counts:
  - persona_count: `15`
  - question_count: `100`
- planned output: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.json`
- planned log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.log`
- execution mode: `tmux`
- note: this is a first draft build from the upstream Chinese MemoryBank data, not a finalized audited gold set

### GVD CN dynamic-tree provisional full100 run

- task: run dynamic-tree on Chinese GVD against the current draft gold set
- code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- answer config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- judge config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.gpt41mini_bgem3.yaml`
- benchmark memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- benchmark questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl`
- provisional gold draft: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.json`
- planned DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/data/gvd_cn_leaf_dynamic_tree_full100_20260417.sqlite3`
- planned ingest report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_ingest_20260417.json`
- planned main report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_20260417.json`
- planned gold eval: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_gold_eval_20260417.json`
- planned gold judge-1: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_goldjudge1_20260417.json`
- evaluation note: this is a provisional CN run because the current gold set is still a draft; for GVD, `gold judge-1` remains the main metric to inspect first
- execution mode: `tmux`

### GVD CN MemoryBank provisional full100 run

- task: run `MemoryBank` on Chinese GVD against the current draft gold set
- code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/scripts/eval_gvd_memorybank.py`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- benchmark memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- benchmark questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl`
- provisional gold draft: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.json`
- planned state dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/runs/gvd_memorybank_cn_state_20260417`
- planned report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_20260417.json`
- planned gold eval: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_gold_eval_20260417.json`
- planned gold judge-1: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_goldjudge1_20260417.json`
- evaluation note: provisional CN run; `gold judge-1` is the main metric to inspect first
- execution mode: `tmux`

### GVD CN MemoryOS provisional full100 run

- task: run official-default `MemoryOS` on Chinese GVD against the current draft gold set
- code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/scripts/eval_gvd_memoryos_official.py`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- answer config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- judge config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.gpt41mini_bgem3.yaml`
- benchmark memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- benchmark questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl`
- provisional gold draft: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.json`
- planned state root: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/runs/gvd_memoryos_cn_state_20260417`
- planned report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_20260417.json`
- planned gold eval: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_gold_eval_20260417.json`
- planned gold judge-1: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_goldjudge1_20260417.json`
- evaluation note: provisional CN run; `gold judge-1` is the main metric to inspect first
- execution mode: `tmux`

### GVD CN gold draft polish pass

- task: polish the current Chinese GVD gold draft into shorter benchmark-friendly canonical answers without changing evidence spans
- source gold: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.json`
- output gold: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_20260417.json`
- code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/scripts/polish_gvd_gold_cn.py`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.gpt41mini_bgem3.yaml`
- memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- rewrite mode: `flagged rows only`
- workers: `4`
- result summary:
  - row_count: `100`
  - flagged_rows_before: `52`
  - flagged_rows_after: `44`
  - rewritten_rows: `22`
  - fallback_invalid_rows: `30`
  - skipped_clean_rows: `48`
  - long_ge_20 before/after: `40 -> 27`
  - too_long_for_type before/after: `30 -> 15`
- caveat: this is a constrained cleanup pass, not a final audited benchmark release; several long recommendation/process answers still remain and may need either type-specific rule compression or a second targeted rewrite pass

### GVD CN gold draft polish pass v4

- task: refine the CN gold cleanup with deterministic rule shaping first, then LLM rewrite only for remaining noisy rows
- source gold: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_draft_20260417.json`
- output gold: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json`
- code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/scripts/polish_gvd_gold_cn.py`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.gpt41mini_bgem3.yaml`
- memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- rewrite mode: `flagged rows only`
- workers: `4`
- result summary:
  - row_count: `100`
  - flagged_rows_before: `46`
  - flagged_rows_after: `40`
  - rule_shaped_rows: `18`
  - rewritten_rows: `18`
  - fallback_invalid_rows: `28`
  - skipped_clean_rows: `36`
  - long_ge_20 before/after: `35 -> 25`
  - too_long_for_type before/after: `25 -> 14`
  - clause_like before/after: `19 -> 18`
  - weak_evidence_overlap before/after: `9 -> 10`
- current assessment: `v4` is the best CN gold candidate so far; remaining noisy rows are mainly long recommendation/process summaries and a few multi-slot factoid answers

### GVD CN full100 rescore on polished gold v4

- task: rescore the existing CN GVD full100 reports against polished gold `v4`
- eval gold: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json`
- judge config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.gpt41mini_bgem3.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- execution mode: `tmux`
- methods rescored:
  - `LEAF dynamic-tree`
    - report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_20260417.json`
    - gold eval v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_gold_eval_v4_20260417.json`
    - gold judge1 v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_goldjudge1_v4_20260417.json`
    - rescore log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_full100_rescore_v4_20260417.log`
    - result: `judge-1 0.8400`, `F1 0.2300`, `BLEU-1 0.2300`, `Strict EM 0.2300`, `Accepted match 0.2600`
  - `MemoryBank`
    - report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_20260417.json`
    - gold eval v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_gold_eval_v4_20260417.json`
    - gold judge1 v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_goldjudge1_v4_20260417.json`
    - rescore log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memorybank_cn_full100_rescore_v4_20260417.log`
    - result: `judge-1 0.8500`, `F1 0.3100`, `BLEU-1 0.3100`, `Strict EM 0.3100`, `Accepted match 0.3600`
  - `MemoryOS`
    - report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_20260417.json`
    - gold eval v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_gold_eval_v4_20260417.json`
    - gold judge1 v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_goldjudge1_v4_20260417.json`
    - rescore log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_memoryos_cn_official_full100_rescore_v4_20260417.log`
    - result: `judge-1 0.9500`, `F1 0.3117`, `BLEU-1 0.3117`, `Strict EM 0.3000`, `Accepted match 0.3200`
