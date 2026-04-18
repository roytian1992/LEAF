# Experiment Log - 2026-04-16

## LoCoMo fresh ingest + search pilot

- method: `dynamic-tree`
- objective: rerun `ingest + search` from scratch on a fresh DB after locking default ingest to `session/session_page/session_block/entity/entity_slot/root`
- release / input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_smoke_pilot_conv26_42_50_q35_20260416.json`
- base full benchmark path referenced by prior runs: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- future novelty cleanup applied: `N/A` for LoCoMo memory benchmark
- config path for this fresh rebuild pilot: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: `N/A` for retrieval-only pilot
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshpilot_ingestsearch_20260416.sqlite3`
- ingest report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshpilot_ingest_20260416.json`
- retrieval report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshpilot_retrieval_6x8_20260416.json`
- tmux session: `locomo_freshpilot_20260416`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_freshpilot_20260416_tmux.log`

Notes before launch:

- default ingest has been locked so `entity_aspect` and `entity_facet` are disabled unless explicitly requested
- this pilot is intended to answer reproducibility of the effective `ingest -> search` pipeline before any fresh full-QA run

## LoCoMo fresh full run

- method: `dynamic-tree`
- objective: run full LoCoMo from fresh ingest on the locked default ingest path after the pilot reproduced retrieval behavior
- release / input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshfull_20260416.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260416.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_freshfull_20260416.qa_progress.jsonl`
- tmux session: `locomo_freshfull_20260416`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_freshfull_20260416_tmux.log`

Practical correction:

- this launch was stopped before meaningful ingest progress because the user preferred reusing the already-built 3-sample pilot DB instead of rebuilding all 10 samples from scratch
- the replacement run reuses:
  - `locomo_conv_26`
  - `locomo_conv_42`
  - `locomo_conv_50`
  from `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshpilot_ingestsearch_20260416.sqlite3`

## LoCoMo full run reusing pilot DB

- method: `dynamic-tree`
- objective: reuse the completed 3-sample pilot DB, ingest only the remaining 7 samples, then run full LoCoMo QA
- release / input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - ingest prepare workers: `4`
  - QA workers: serial inside `eval_locomo.py`
- retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
- target DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/data/locomo10_dynamic_tree_freshpilot_ingestsearch_20260416.sqlite3`
- result report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_full_reusepilot_20260416.json`
- progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_full_reusepilot_20260416.qa_progress.jsonl`
- tmux session: `locomo_full_reusepilot_20260416`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_full_reusepilot_20260416_tmux.log`

## LoCoMo dynamic-tree judge-5 supplementation

- method: `dynamic-tree`
- objective: supplement `judge-5` on top of the completed non-`adv` full QA report without rerunning retrieval or answer generation
- release / input path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_full_reusepilot_20260416.json`
- benchmark path referenced by the source report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- worker count:
  - judge runs per row: `5`
  - judge max workers: `8`
- output report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_full_reusepilot_judge5_20260416.json`
- judge progress log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/locomo10_dynamic_tree_full_reusepilot_judge5_20260416.judge_progress.jsonl`
- tmux session: `locomo_dynjudge5_20260416`
- tmux log: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tmp/locomo_dynjudge5_20260416_tmux.log`

Notes before launch:

- this run uses the existing judged-answer supplementation script `scripts/judge_locomo_report.py`
- it does not touch ingest, retrieval, answer prompt, or predicted answers; it only appends row-level judge fields and recomputes judge summaries

## GVD dynamic-tree full100 preparation

- method: `dynamic-tree`
- objective: validate whether the current `dynamic-tree` LEAF path is broadly stronger than the canonical `LEAF(memory baseline)` on `GVD full100`
- benchmark memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json`
- benchmark questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl`
- config path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/examples/config_local.yaml`
- python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python`
- tmux required: `yes`
- planned ingest mode: `migration`
- planned retrieval settings:
  - `snapshot_limit = 6`
  - `raw_span_limit = 8`
- planned answer settings:
  - `answer_style = short`
  - `answer_revision = none`
- planned DB: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/data/gvd_leaf_dynamic_tree_full100_20260416.sqlite3`
- planned main report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_leaf_dynamic_tree_full100_20260416.json`
- planned ingest report: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_leaf_dynamic_tree_full100_ingest_20260416.json`
- planned gold eval: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_leaf_dynamic_tree_full100_gold_eval_20260416.json`
- planned gold judge-1: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_leaf_dynamic_tree_full100_goldjudge1_20260416.json`

Planned execution order:

1. ingest-only report with `scripts/report_gvd_ingest.py --ingest-mode migration`
2. main full100 QA report with `scripts/eval_gvd.py --ingest-mode migration --answer-style short --answer-revision none`
3. gold eval against canonical GVD answer sheet
4. gold-reference judge-1 on the completed report

Notes before launch:

- `eval_gvd.py` and `report_gvd_ingest.py` have been updated to accept `--ingest-mode {online,migration}`
- this run has not been launched yet; we are waiting for the LoCoMo `judge-5` supplementation to finish before starting GVD

## Ingest speed optimization note

- recorded at: `2026-04-18 13:35:37 CST`
- python env actually used for validation:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python3`
  - conda env: `base`
- code paths changed:
  - [service.py](/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/src/leaf/service.py)
  - [indexer.py](/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/src/leaf/indexer.py)
  - [store.py](/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/src/leaf/store.py)
  - [extract.py](/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/src/leaf/extract.py)

What changed:

- `migration` ingest no longer refreshes session/entity/root snapshots twice.
  - old path effectively did:
    `append_turns(refresh snapshots) -> migrate_corpus(delete + rebuild snapshots)`
  - new path does:
    `append_turns(write events/atoms/objects only) -> migrate_corpus(build snapshots once)`
- atom extraction now supports dedicated ingest-time concurrency via `LEAF_INGEST_EXTRACTION_WORKERS`.
  - default falls back to the prepare-worker count.
- derived snapshot refresh now uses batched version/event reads instead of repeated per-object point queries.
- repeated `summarize_texts(...)` calls inside entity-slot snapshot building were deduplicated.

Validation:

- syntax check passed with:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python3 -m py_compile ...`
- smoke tests passed with:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python3 -m unittest /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/tests/test_smoke.py`

Micro-benchmark on `GVD Gary` (`58` turns):

- old-equivalent path:
  - `append_turns(refresh_snapshots=True) + migrate_corpus()`
  - total: `27039.65 ms`
- new migration path:
  - `service.append_turns(..., ingest_mode="migration")`
  - total: `25771.06 ms`
- speedup from removing duplicate snapshot refresh alone:
  - `1.049x`

Atom-extraction concurrency benchmark on the same persona:

- `LEAF_INGEST_EXTRACTION_WORKERS=1`
  - total: `57803.15 ms`
- `LEAF_INGEST_EXTRACTION_WORKERS=4`
  - total: `20569.02 ms`
- measured speedup:
  - `2.81x`

Current conclusion:

- the dominant ingest bottleneck is atom extraction latency, not derived snapshot rebuilding.
- removing duplicate snapshot refresh is still correct and worth keeping, but the qualitative speed gain mainly comes from extraction concurrency.
