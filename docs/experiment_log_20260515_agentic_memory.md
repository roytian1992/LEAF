# Experiment Log 2026-05-15

## Scope

- Project: LEAF agentic tree memory
- Purpose: Start an isolated implementation branch for background-evolved, versioned memory views.
- Status: MVP implementation plus first GVD smoke and topic-routing shadow eval completed

## Inputs

- Source code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF`
- New workspace: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory`
- Base branch: `dynamic-tree2`
- New branch: `agentic-memory-evolution`
- Base commit: `fad4f3b Rename additional_llm to memory_llm`

## Commands

```bash
git -C /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF \
  worktree add -b agentic-memory-evolution \
  /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory \
  dynamic-tree2
```

Bootstrap command template:

```bash
python scripts/init_agentic_memory_view.py \
  --db data/example.sqlite3 \
  --corpus-id example \
  --activate
```

GVD smoke setup and run commands:

```bash
cp /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/data/gvd_leaf_dynamic_tree_full100_extractive_structctx_fastingest_20260418.sqlite3 \
  /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory/data/agentic_smoke/gvd_agentic_smoke.sqlite3

PYTHONPATH=src python scripts/init_agentic_memory_view.py \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --activate

PYTHONPATH=src python scripts/build_selfqa_from_memory.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --output reports/agentic_memory/gvd_emily_selfqa_balanced_smoke.jsonl \
  --summary-output reports/agentic_memory/gvd_emily_selfqa_balanced_smoke_summary.json \
  --limit 9 \
  --candidate-limit 60 \
  --task-types single_fact multi_hop temporal \
  --write-db

PYTHONPATH=src python scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_balanced_smoke.jsonl \
  --output reports/agentic_memory/gvd_emily_search_eval_balanced_smoke.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --trace-memory
```

Validated self-QA and promotion gate commands:

```bash
PYTHONPATH=src python scripts/build_selfqa_from_memory.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --output reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --summary-output reports/agentic_memory/gvd_emily_selfqa_validated_smoke_summary.json \
  --limit 6 \
  --candidate-limit 90 \
  --task-types single_fact multi_hop temporal \
  --validate \
  --min-validation-score 0.75 \
  --write-db

PYTHONPATH=src python scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --output reports/agentic_memory/gvd_emily_search_eval_validated_smoke.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --trace-memory

PYTHONPATH=src python scripts/evaluate_memory_view_gate.py \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --view-id view_a53d51bc6241a08203 \
  --eval-report reports/agentic_memory/gvd_emily_search_eval_validated_smoke.json \
  --output reports/agentic_memory/gvd_emily_promotion_gate_validated_smoke.json \
  --min-task-count 6 \
  --min-mean-event-recall 0.95 \
  --min-event-path-hit-rate 0.95 \
  --min-mean-atom-recall 0.95 \
  --min-atom-path-hit-rate 0.95 \
  --max-avg-elapsed-ms 1000 \
  --promote \
  --record-run
```

Topic-routing shadow eval commands:

```bash
PYTHONPATH=src python scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --output reports/agentic_memory/gvd_emily_search_eval_validated_topic_shadow_smoke.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --trace-memory \
  --topic-routing-shadow \
  --topic-router keyword \
  --topic-route-top-k 3

PYTHONPATH=src python scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --output reports/agentic_memory/gvd_emily_search_eval_validated_topic_shadow_llm_smoke.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --trace-memory \
  --topic-routing-shadow \
  --topic-router llm \
  --topic-route-top-k 3
```

Evolution A/B commands:

```bash
PYTHONPATH=src python scripts/evolve_topic_view_from_shadow.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --base-view-id view_a53d51bc6241a08203 \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --shadow-eval-report reports/agentic_memory/gvd_emily_search_eval_validated_topic_shadow_llm_smoke.json \
  --output reports/agentic_memory/gvd_emily_evolved_topic_view_from_shadow_smoke.json \
  --name evolved-topic-shadow-v1 \
  --strategy llm \
  --max-new-topics 4 \
  --record-run

PYTHONPATH=src python scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --output reports/agentic_memory/gvd_emily_search_eval_base_view_llm_shadow_compare_smoke.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --trace-memory \
  --topic-routing-shadow \
  --topic-router llm \
  --topic-route-top-k 3 \
  --topic-view-id view_a53d51bc6241a08203

PYTHONPATH=src python scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --selfqa reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl \
  --output reports/agentic_memory/gvd_emily_search_eval_evolved_view_llm_shadow_compare_smoke.json \
  --snapshot-limit 6 \
  --raw-span-limit 8 \
  --trace-memory \
  --topic-routing-shadow \
  --topic-router llm \
  --topic-route-top-k 3 \
  --topic-view-id view_fcb7db2f310faa3bc4

PYTHONPATH=src python scripts/compare_memory_view_eval.py \
  --baseline-report reports/agentic_memory/gvd_emily_search_eval_base_view_llm_shadow_compare_smoke.json \
  --candidate-report reports/agentic_memory/gvd_emily_search_eval_evolved_view_llm_shadow_compare_smoke.json \
  --output reports/agentic_memory/gvd_emily_base_vs_evolved_topic_shadow_compare_smoke.json

PYTHONPATH=src python scripts/evaluate_memory_view_gate.py \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --corpus-id gvd_emily \
  --view-id view_fcb7db2f310faa3bc4 \
  --eval-report reports/agentic_memory/gvd_emily_search_eval_evolved_view_llm_shadow_compare_smoke.json \
  --baseline-report reports/agentic_memory/gvd_emily_search_eval_base_view_llm_shadow_compare_smoke.json \
  --output reports/agentic_memory/gvd_emily_evolved_topic_gate_smoke.json \
  --min-task-count 6 \
  --min-mean-event-recall 0.95 \
  --min-event-path-hit-rate 0.95 \
  --min-mean-atom-recall 0.95 \
  --min-atom-path-hit-rate 0.95 \
  --min-mean-topic-recall 0.75 \
  --min-topic-path-hit-rate 0.75 \
  --min-topic-recall-improvement 0.2 \
  --max-avg-elapsed-ms 1000 \
  --record-run
```

## Outputs

- Design note: `docs/agentic_memory_design.md`
- Store schema changes: `src/leaf/store.py`
- Agentic memory MVP: `src/leaf/agentic_memory.py`
- Service API hooks: `src/leaf/service.py`
- Bootstrap script: `scripts/init_agentic_memory_view.py`
- Self-QA generator: `scripts/build_selfqa_from_memory.py`
- Retrieval evaluator: `scripts/eval_memory_search.py`
- Local ignored config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Smoke DB copy: `data/agentic_smoke/gvd_agentic_smoke.sqlite3`
- Self-QA smoke: `reports/agentic_memory/gvd_emily_selfqa_balanced_smoke.jsonl`
- Self-QA summary: `reports/agentic_memory/gvd_emily_selfqa_balanced_smoke_summary.json`
- Retrieval eval: `reports/agentic_memory/gvd_emily_search_eval_balanced_smoke.json`
- Validated self-QA smoke: `reports/agentic_memory/gvd_emily_selfqa_validated_smoke.jsonl`
- Validated self-QA summary: `reports/agentic_memory/gvd_emily_selfqa_validated_smoke_summary.json`
- Validated retrieval eval: `reports/agentic_memory/gvd_emily_search_eval_validated_smoke.json`
- Promotion gate report: `reports/agentic_memory/gvd_emily_promotion_gate_validated_smoke.json`
- Keyword topic shadow eval: `reports/agentic_memory/gvd_emily_search_eval_validated_topic_shadow_smoke.json`
- LLM topic shadow eval: `reports/agentic_memory/gvd_emily_search_eval_validated_topic_shadow_llm_smoke.json`
- Evolved topic proposal: `reports/agentic_memory/gvd_emily_evolved_topic_view_from_shadow_smoke.json`
- Base-view comparison eval: `reports/agentic_memory/gvd_emily_search_eval_base_view_llm_shadow_compare_smoke.json`
- Evolved-view comparison eval: `reports/agentic_memory/gvd_emily_search_eval_evolved_view_llm_shadow_compare_smoke.json`
- Base-vs-evolved delta report: `reports/agentic_memory/gvd_emily_base_vs_evolved_topic_shadow_compare_smoke.json`
- Evolved-view gate report: `reports/agentic_memory/gvd_emily_evolved_topic_gate_smoke.json`

## Counts And Metrics

- Smoke DB source: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/data/gvd_leaf_dynamic_tree_full100_extractive_structctx_fastingest_20260418.sqlite3`
- Smoke DB copy size: 59M
- Corpus evaluated: `gvd_emily`
- Seed memory view: `view_a53d51bc6241a08203`
- Topic nodes written: 11
- Atom assignments written: 145
- Self-QA candidate count: 350
- Self-QA candidate sample count: 60
- Self-QA accepted count: 9
- Self-QA primary task types: single_fact 4, multi_hop 2, temporal 3
- Self-QA tag counts: single_fact 7, multi_hop 2, temporal 3
- Retrieval evaluated count: 9
- Mean event recall: 1.0
- Event path hit rate: 1.0
- Mean atom recall: 1.0
- Atom path hit rate: 1.0
- Mean returned atom recall: 0.0
- Returned atom path hit rate: 0.0
- Average elapsed time: 349.7933 ms
- Validated self-QA accepted count: 6
- Validated self-QA primary task types: single_fact 3, multi_hop 1, temporal 2
- Validated self-QA tag counts: single_fact 5, multi_hop 1, temporal 2
- Validated retrieval evaluated count: 6
- Validated mean event recall: 1.0
- Validated event path hit rate: 1.0
- Validated mean atom recall: 1.0
- Validated atom path hit rate: 1.0
- Validated average elapsed time: 499.6983 ms
- Promotion gate: passed
- Promotion gate evolution run status: promoted
- DB after gate: `leaf_selfqa_tasks` accepted 12, validated 6; `leaf_search_traces` 25; `leaf_evolution_runs` promoted 1
- Keyword shadow topic evaluated count: 6
- Keyword shadow mean topic recall: 0.0
- Keyword shadow topic path hit rate: 0.0
- LLM shadow topic evaluated count: 6
- LLM shadow mean topic recall: 0.5
- LLM shadow topic path hit rate: 0.5
- LLM shadow routed topic counts: hobbies_media 6, preferences_opinions 4, events_timeline 4, misc 3, travel_places 1
- Gold topic counts in validated smoke: preferences_opinions 4, relationships 2, plans_tasks 1
- Retrieved evidence topic coverage in shadow eval: mean retrieval topic recall 1.0, retrieval topic path hit rate 1.0
- Evolved candidate view: `view_fcb7db2f310faa3bc4`
- Evolved candidate status after gate: validated, not active
- Evolved topics added: `food_cooking`, `social_chitchat`, `reading_books`
- Evolved atoms reassigned: 97
- Base-vs-evolved task count: 6 vs 6
- Base-vs-evolved mean event recall: 1.0 vs 1.0, delta 0.0
- Base-vs-evolved mean atom recall: 1.0 vs 1.0, delta 0.0
- Base-vs-evolved mean topic recall: 0.5 vs 0.8333, delta +0.3333
- Base-vs-evolved topic path hit rate: 0.5 vs 0.8333, delta +0.3333
- Evolved-view gate: passed with topic recall threshold 0.75 and topic recall improvement threshold +0.2
- DB after evolved gate: active base view `view_a53d51bc6241a08203`; candidate view `view_fcb7db2f310faa3bc4` is validated but not promoted; `leaf_evolution_runs` statuses candidate_created 1, passed 1, promoted 1

## Official GVD Full100 A/B

Inputs:

- Gold file: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json`
- Questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl`
- Memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json`
- Baseline DB: `data/agentic_smoke/gvd_agentic_smoke.sqlite3`
- Topic-soft DB copy: `data/agentic_smoke/gvd_agentic_full100_topicsoft_20260515.sqlite3`
- Topic-soft DB setup: initialized seed topic tree for 14 non-Emily corpora; promoted Emily evolved view `view_fcb7db2f310faa3bc4`; total topic assignments 2122.

Commands:

```bash
PYTHONPATH=src python scripts/eval_gvd.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_smoke.sqlite3 \
  --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json \
  --questions /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl \
  --output reports/agentic_memory/gvd_full100_official_baseline_structctx_20260515.json \
  --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 \
  --answer-view-mode extractive --answer-style structured_context \
  --answer-revision none --unknown-recovery none

PYTHONPATH=src python scripts/eval_gvd.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/gvd_agentic_full100_topicsoft_20260515.sqlite3 \
  --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json \
  --questions /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl \
  --output reports/agentic_memory/gvd_full100_official_topic_soft_active_keyword_limit2_fallback_structctx_20260515.json \
  --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 \
  --answer-view-mode extractive --answer-style structured_context \
  --answer-revision none --unknown-recovery none \
  --retrieval-mode topic_soft --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 2 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-fallback baseline_on_unknown
```

Gold metrics were computed with:

```bash
python /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/scripts/eval_gvd_gold_metrics.py \
  --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json \
  --report <report.json> \
  --output <gold_eval.json>
```

Result files:

- Current baseline report: `reports/agentic_memory/gvd_full100_official_baseline_structctx_20260515.json`
- Current baseline gold eval: `reports/agentic_memory/gvd_full100_official_baseline_structctx_gold_eval_20260515.json`
- Topic-soft keyword report: `reports/agentic_memory/gvd_full100_official_topic_soft_active_keyword_limit2_structctx_20260515.json`
- Topic-soft keyword gold eval: `reports/agentic_memory/gvd_full100_official_topic_soft_active_keyword_limit2_structctx_gold_eval_20260515.json`
- Topic-soft keyword fallback report: `reports/agentic_memory/gvd_full100_official_topic_soft_active_keyword_limit2_fallback_structctx_20260515.json`
- Topic-soft keyword fallback gold eval: `reports/agentic_memory/gvd_full100_official_topic_soft_active_keyword_limit2_fallback_structctx_gold_eval_20260515.json`
- Topic-soft LLM fallback report: `reports/agentic_memory/gvd_full100_official_topic_soft_active_llm_limit2_fallback_structctx_20260515.json`
- Topic-soft LLM fallback gold eval: `reports/agentic_memory/gvd_full100_official_topic_soft_active_llm_limit2_fallback_structctx_gold_eval_20260515.json`

Metrics:

| Run | Q | F1 | BLEU1 | strict EM | accepted | avg search ms | avg answer tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old fastingest full100 | 100 | 0.7057 | 0.6379 | 0.33 | 0.39 | 77.09 | 963.86 |
| Old selected ansfix2 full100 | 100 | 0.7511 | 0.6859 | 0.37 | 0.42 | 99.72 | 1034.86 |
| Current baseline full100 | 100 | 0.7379 | 0.6760 | 0.37 | 0.42 | 103.51 | 1041.60 |
| Topic-soft keyword limit2 | 100 | 0.7347 | 0.6749 | 0.37 | 0.42 | 96.94 | 1092.79 |
| Topic-soft keyword limit2 + baseline_on_unknown | 100 | 0.7525 | 0.6904 | 0.37 | 0.42 | 94.06 | 1191.16 |
| Topic-soft LLM limit2 + baseline_on_unknown | 100 | 0.7428 | 0.6779 | 0.36 | 0.42 | 3318.10 | 1184.12 |

Interpretation:

- Current baseline is +0.0322 F1 over old fastingest but -0.0132 F1 under old selected ansfix2.
- Ungated topic-soft adds nearly two events per query but slightly hurts F1, showing that topic evidence must be gated.
- `baseline_on_unknown` is the first full100 run that beats old selected ansfix2: +0.0014 F1 and +0.0045 BLEU1, with equal accepted and strict EM.
- LLM topic routing is not worth the current online cost: lower F1 than keyword fallback and ~35x slower search.
- Emily-only evolved-view official subset: baseline and topic-soft limit2 both scored F1 0.8172, accepted 0.5714. The full100 gain comes from topic-soft plus fallback across all active topic views, not from Emily evolution alone.

## Chinese GVD Full100 A/B

Inputs:

- Gold file: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json`
- Questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl`
- Memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml`, same provider split as the English config but with `language.mode=zh`; credentials remain in the ignored local config.
- Baseline DB copy: `data/agentic_smoke/gvd_cn_agentic_smoke_20260515.sqlite3`
- Topic-soft DB copy: `data/agentic_smoke/gvd_cn_agentic_topicsoft_20260515.sqlite3`
- Topic-soft DB setup: 15 active seed topic views, 165 topic nodes, 2045 topic assignments across 15 Chinese corpora.

Commands:

```bash
PYTHONPATH=src python scripts/eval_gvd.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml \
  --db data/agentic_smoke/gvd_cn_agentic_smoke_20260515.sqlite3 \
  --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json \
  --questions /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl \
  --output reports/agentic_memory/gvd_cn_full100_official_baseline_structctx_20260515.json \
  --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 \
  --answer-view-mode extractive --answer-style structured_context \
  --answer-revision none --unknown-recovery none

PYTHONPATH=src python scripts/eval_gvd.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml \
  --db data/agentic_smoke/gvd_cn_agentic_topicsoft_20260515.sqlite3 \
  --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json \
  --questions /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl \
  --output reports/agentic_memory/gvd_cn_full100_topic_soft_keyword_limit2_fallback_structctx_20260515.json \
  --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 \
  --answer-view-mode extractive --answer-style structured_context \
  --answer-revision none --unknown-recovery none \
  --retrieval-mode topic_soft --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 2 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-fallback baseline_on_unknown

PYTHONPATH=src python scripts/eval_gvd.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml \
  --db data/agentic_smoke/gvd_cn_agentic_smoke_20260515.sqlite3 \
  --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json \
  --questions /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl \
  --output reports/agentic_memory/gvd_cn_full100_baseline_heuristic_structcompact_20260515.json \
  --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 \
  --answer-view-mode heuristic --answer-style structured_compact \
  --answer-revision none --unknown-recovery none

PYTHONPATH=src python scripts/eval_gvd.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml \
  --db data/agentic_smoke/gvd_cn_agentic_topicsoft_20260515.sqlite3 \
  --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json \
  --questions /vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl \
  --output reports/agentic_memory/gvd_cn_full100_topic_soft_keyword_limit2_fallback_heuristic_structcompact_20260515.json \
  --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 \
  --answer-view-mode heuristic --answer-style structured_compact \
  --answer-revision none --unknown-recovery none \
  --retrieval-mode topic_soft --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 2 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-fallback baseline_on_unknown
```

Gold metrics were computed with:

```bash
python /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/scripts/eval_gvd_gold_metrics.py \
  --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json \
  --report <report.json> \
  --output <gold_eval.json>
```

Result files:

- Current formal baseline report: `reports/agentic_memory/gvd_cn_full100_official_baseline_structctx_20260515.json`
- Current formal baseline gold eval: `reports/agentic_memory/gvd_cn_full100_official_baseline_structctx_gold_eval_20260515.json`
- Current formal topic-soft report: `reports/agentic_memory/gvd_cn_full100_topic_soft_keyword_limit2_fallback_structctx_20260515.json`
- Current formal topic-soft gold eval: `reports/agentic_memory/gvd_cn_full100_topic_soft_keyword_limit2_fallback_structctx_gold_eval_20260515.json`
- Current compact baseline report: `reports/agentic_memory/gvd_cn_full100_baseline_heuristic_structcompact_20260515.json`
- Current compact baseline gold eval: `reports/agentic_memory/gvd_cn_full100_baseline_heuristic_structcompact_gold_eval_20260515.json`
- Current compact topic-soft report: `reports/agentic_memory/gvd_cn_full100_topic_soft_keyword_limit2_fallback_heuristic_structcompact_20260515.json`
- Current compact topic-soft gold eval: `reports/agentic_memory/gvd_cn_full100_topic_soft_keyword_limit2_fallback_heuristic_structcompact_gold_eval_20260515.json`
- Old formal selected gold eval: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/gvd_cn_leaf_dynamic_tree_zh_full_extractive_structctx_rerun2_gold_eval_20260418_1620.json`

Metrics:

| Run | Q | F1 | BLEU1 | strict EM | accepted | avg search ms | avg answer tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old formal selected zh full | 100 | 0.315 | 0.315 | 0.31 | 0.34 | 89.25 | 852.40 |
| Current formal baseline | 100 | 0.315 | 0.315 | 0.31 | 0.34 | 94.63 | 852.40 |
| Current formal topic-soft keyword limit2 + fallback | 100 | 0.325 | 0.325 | 0.32 | 0.35 | 96.76 | 881.51 |
| Current compact baseline | 100 | 0.325 | 0.325 | 0.32 | 0.35 | 94.90 | 628.77 |
| Current compact topic-soft keyword limit2 + fallback | 100 | 0.335 | 0.335 | 0.33 | 0.36 | 95.86 | 644.16 |

Interpretation:

- Current formal Chinese baseline exactly reproduces old formal selected lexical metrics: F1 0.315, BLEU1 0.315, strict EM 0.31, accepted 0.34.
- Formal topic-soft improves by +0.010 F1, +0.010 BLEU1, +0.010 strict EM, and +0.010 accepted, with +29.11 average answer input tokens and +2.13 ms average search time.
- Compact topic-soft improves by +0.010 F1, +0.010 BLEU1, +0.010 strict EM, and +0.010 accepted, with +15.39 average answer input tokens and +0.96 ms average search time.
- Delta diagnostics show one additional exact lexical hit in each setting and no regressions: formal improved `焦彦` question 4; compact improved `曹志强` question 3.
- Topic-soft raw eval used 2.0 extra topic events on average and `baseline_on_unknown` fallback was not triggered in the Chinese runs.

## LoCoMo10 Baseline / Seed Topic / Evolved Seed Topic QA

Date: 2026-05-16

Inputs:

- Dataset: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Answer-side LLM: local `Qwen3-235B` at `http://localhost:8001/v1`
- Memory-side / self-QA / evolution LLM: `gpt-5.4-mini` through the ignored local config `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Baseline DB: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- Seed-topic DB: `data/agentic_smoke/locomo10_seed_topic_qwen_answer_20260516.sqlite3`
- Evolved seed-topic DB: `data/agentic_smoke/locomo10_evolved_seed_topic_qwen_answer_20260516.sqlite3`

Commands:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db <db.sqlite3> \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output <qa_report.json> \
  --snapshot-limit 8 --raw-span-limit 8 --answer-style structured_context --qa-workers 8

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db <topic_db.sqlite3> \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output <topic_soft_qa_report.json> \
  --snapshot-limit 8 --raw-span-limit 8 --answer-style structured_context --qa-workers 8 \
  --retrieval-mode topic_soft --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 2 --topic-soft-fallback baseline_on_unknown

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolved_seed_topic_qwen_answer_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --answer-style structured_context --qa-workers 8 \
  --retrieval-mode topic_soft_gated --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown
```

Evolution command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/run_locomo_evolved_seed_pipeline.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolved_seed_topic_qwen_answer_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output-jsonl reports/agentic_memory/locomo10_evolved_seed_topic_pipeline_20260516.jsonl \
  --summary-output reports/agentic_memory/locomo10_evolved_seed_topic_pipeline_summary_20260516.json \
  --tasks-per-sample 12 --candidate-limit 120 --qa-validate \
  --churn-ratio-max 0.15 --promote
```

Result files:

- Baseline QA: `reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_20260515.json`
- Seed topic-soft QA: `reports/agentic_memory/locomo10_seed_topic_qwen_answer_topic_soft_qa_20260516.json`
- Evolved seed topic-soft QA: `reports/agentic_memory/locomo10_evolved_seed_topic_qwen_answer_topic_soft_qa_20260516.json`
- Evolved gated topic-soft QA: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.json`
- Evolved gated topic-soft QA log: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.log`
- Seed init/backfill summary: `reports/agentic_memory/locomo10_seed_topic_init_backfill_summary_20260516.json`
- Evolution pipeline JSONL: `reports/agentic_memory/locomo10_evolved_seed_topic_pipeline_20260516.jsonl`
- Evolution pipeline summary: `reports/agentic_memory/locomo10_evolved_seed_topic_pipeline_summary_20260516.json`
- Question-level comparison summary: `reports/agentic_memory/locomo10_baseline_seed_evolved_topic_soft_comparison_20260516.json`
- Gated comparison summary: `reports/agentic_memory/locomo10_baseline_seed_evolved_gated_topic_soft_comparison_20260516.json`

Evolution summary:

- Samples processed: 10
- Promoted evolved active views: 9
- Gate failed: 1 (`conv-30`, retained the seed active view)
- Self-QA tasks per sample: 12, LLM validation enabled
- Churn gate: `churn_ratio_max=0.15`

QA metrics:

| Run | Q | F1 | BLEU1 | avg elapsed ms | avg search ms | avg answer ms | avg topic events | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline QA | 1540 | 0.4547 | 0.4005 | 12220.11 | 11602.38 | 617.73 | 0.0000 | 0 |
| Seed topic-soft QA | 1540 | 0.4528 | 0.3978 | 15082.52 | 14312.74 | 769.78 | 1.9961 | 6 |
| Evolved seed topic-soft QA | 1540 | 0.4547 | 0.3998 | 14085.22 | 13401.76 | 683.46 | 1.9838 | 9 |
| Evolved gated topic-soft QA | 1540 | 0.4557 | 0.4013 | 13852.88 | 13018.58 | 834.30 | 0.9935 | 5 |

Overall deltas:

| Delta | F1 | BLEU1 | avg elapsed ms | avg search ms | avg answer ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed topic-soft - baseline | -0.0019 | -0.0027 | +2862.41 | +2710.36 | +152.05 |
| Evolved seed topic-soft - baseline | +0.0000 | -0.0007 | +1865.11 | +1799.38 | +65.73 |
| Evolved gated topic-soft - baseline | +0.0010 | +0.0008 | +1632.77 | +1416.20 | +216.57 |
| Evolved seed topic-soft - seed topic-soft | +0.0019 | +0.0020 | -997.30 | -910.98 | -86.32 |
| Evolved gated topic-soft - evolved seed topic-soft | +0.0010 | +0.0015 | -232.34 | -383.18 | +150.84 |

By category:

| Category | Count | Baseline F1 | Seed F1 | Evolved F1 | Gated F1 | Gated - baseline | Gated - evolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi_hop | 282 | 0.2776 | 0.2758 | 0.2713 | 0.2751 | -0.0025 | +0.0038 |
| open_domain | 96 | 0.2570 | 0.2687 | 0.2587 | 0.2548 | -0.0022 | -0.0039 |
| single_hop | 841 | 0.4970 | 0.4952 | 0.4982 | 0.4996 | +0.0026 | +0.0014 |
| temporal | 321 | 0.5584 | 0.5525 | 0.5605 | 0.5594 | +0.0010 | -0.0011 |

Interpretation:

- Static seed topic-soft is not positive overall on LoCoMo10: it helps open-domain questions but slightly hurts multi-hop, single-hop, and temporal questions.
- Evolved seed topic-soft recovers the seed regression and nearly ties the baseline overall, while improving over seed on single-hop and temporal questions.
- Evolved topic-soft is still not a clear LoCoMo QA win over baseline. The current gain is mechanism-level evidence that self-QA-guided evolution can repair part of seed-topic damage, not proof that topic views are already improving retrieval quality.
- Gated evolved topic-soft is the first LoCoMo10 run in this sequence that beats the baseline overall, but only slightly: +0.0010 F1 and +0.0008 BLEU1. The gain mainly comes from single-hop and temporal questions; multi-hop and open-domain remain below baseline.
- Gated topic-soft reduced added topic events from about 1.98 to 0.99 per question and filtered 76.79 candidate atoms per question on average before adding evidence.
- The current `avg_search_ms` includes `service.search`, query embedding, topic routing, topic-soft expansion, and merge. It is not pure in-memory search and is affected by `qa-workers=8`, embedding service queueing, SQLite/cache contention, and CPU scoring.
- The gated full run suffered severe service/queue latency spikes during several samples; treat its latency metrics as contaminated and use the quality metrics plus event/filter counts for method comparison.

## Changes

- Added versioned memory view tables.
- Added seeded topic tree bootstrap.
- Added atom-level deterministic topic assignment.
- Added optional search trace recording via `LEAFService.search(..., trace_memory=True)`.
- Default retrieval behavior remains unchanged.
- Added ignored local test config at `tmp/config_agentic_codeai_memory_qwen_answer.yaml`; credentials are stored only in this ignored config file for this smoke.
- Settled first test-time LLM split:
  - answer-side LLM: local `Qwen3-235B` at `http://localhost:8001/v1`
  - embedding: local `bge-m3` at `http://127.0.0.1:8080/v1`
  - memory-side / self-QA / validator LLM: `gpt-5.4-mini` through the CodeAI-compatible endpoint in the ignored config file
- Changed self-QA candidate sampling to round-robin by primary task type.
- Changed atom-level retrieval evaluation to compare gold atoms against atoms covered by retrieved events, while keeping direct returned atom metrics for diagnosis.
- Added optional LLM validation for generated self-QA tasks.
- Added promotion-gate script that writes gate metrics to memory views and can record `leaf_evolution_runs`.
- Added keyword and LLM topic-routing shadow metrics to retrieval eval.
- Added service-level `route_query_topics(...)` for active-view keyword routing.
- Added candidate topic-view evolution from shadow misses.
- Added baseline-vs-candidate eval comparison script.
- Extended promotion gate with topic-recall and topic-improvement checks.
- Added topic-soft retrieval augmentation for GVD eval.
- Added `--topic-soft-fallback baseline_on_unknown` safeguard so topic-soft evidence cannot replace a concrete baseline answer with `UNKNOWN`.
- Added online incremental topic assignment: when an active memory view exists, `LEAFService.append_turns(...)` assigns only newly written atoms to that active view.
- Added `LEAFService.ensure_seed_agentic_memory_view(...)` for production-style cold-start seed priors before new turns arrive.
- Added `scripts/backfill_active_memory_view_assignments.py` for benchmark/offline DBs where the active view is created after historical atoms already exist.
- Added assignment churn limits to candidate evolution and promotion gates so broad keyword matches cannot silently move a large fraction of atoms.
- Added `topic_soft_gated`, which keeps the old `topic_soft` mode reproducible while adding a content-token overlap gate, optional fallback-topic control, candidate filtering stats, and active-route diagnostics.
- Added `topic_soft_selective` as an experimental query-time policy that suppresses gated topic evidence when the selected topic event has weak content overlap or the candidate atom pool is too broad.
- Added deterministic LoCoMo temporal answer postprocessing, exposed as `--temporal-postprocess {anchor_only,range,range_no_weekday}` and as `scripts/postprocess_locomo_temporal.py`, to rewrite date answers back to retrieved relative-time anchors when appropriate.

## 2026-05-16 Optimization: Selective Topic Gate And Temporal Postprocess

Inputs:

- Baseline full result: `reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_20260515.json`
- Evolved gated full result: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.json`
- Selective sample result: `reports/agentic_memory/locomo10_evolved_topic_soft_selective_overlap2_candidate20_allowmisc_qwen_answer_sample200_20260516.json`

Commands:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolved_seed_topic_qwen_answer_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolved_topic_soft_selective_overlap2_candidate20_allowmisc_qwen_answer_sample200_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --answer-style structured_context --qa-workers 8 --qa-per-sample 20 \
  --retrieval-mode topic_soft_selective --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-policy-min-selected-overlap 2 --topic-soft-policy-max-candidate-atoms 20 \
  --topic-soft-fallback baseline_on_unknown

/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python scripts/postprocess_locomo_temporal.py \
  --input reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.json \
  --output reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_temporal_range_post_20260516.json \
  --mode range

/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python scripts/postprocess_locomo_temporal.py \
  --input reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_20260515.json \
  --output reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_temporal_range_post_20260516.json \
  --mode range
```

Outputs:

- Comparison summary: `reports/agentic_memory/locomo10_gated_temporal_post_comparison_20260516.json`
- Gated temporal postprocess result: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_temporal_range_post_20260516.json`
- Baseline temporal postprocess result: `reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_temporal_range_post_20260516.json`
- Selective sample temporal postprocess result: `reports/agentic_memory/locomo10_evolved_topic_soft_selective_overlap2_candidate20_allowmisc_qwen_answer_sample200_temporal_range_post_20260516.json`

Metrics:

| Run | Q | F1 | BLEU1 | Temporal F1 | Avg topic events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1540 | 0.4547 | 0.4005 | 0.5584 | 0.0000 |
| Baseline + temporal post | 1540 | 0.4639 | 0.4127 | 0.6025 | 0.0000 |
| Evolved gated topic-soft | 1540 | 0.4557 | 0.4013 | 0.5594 | 0.9935 |
| Evolved gated topic-soft + temporal post | 1540 | 0.4651 | 0.4136 | 0.6042 | 0.9935 |
| Selective topic-soft sample200 | 200 | 0.4206 | 0.3530 | 0.5930 | 0.5950 |
| Selective topic-soft sample200 + temporal post | 200 | 0.4409 | 0.3800 | 0.6403 | 0.5950 |

Deltas:

| Delta | F1 | BLEU1 | Temporal F1 |
| --- | ---: | ---: | ---: |
| Baseline + temporal post - baseline | +0.0092 | +0.0122 | +0.0441 |
| Evolved gated + temporal post - evolved gated | +0.0094 | +0.0123 | +0.0448 |
| Evolved gated + temporal post - baseline | +0.0104 | +0.0131 | +0.0458 |
| Evolved gated + temporal post - baseline + temporal post | +0.0012 | +0.0009 | +0.0017 |

Interpretation:

- The selective topic gate did not justify a full 1540-question rerun. On the first 200-question slice, it reduced topic events from 0.99 to 0.595 per question but did not improve raw QA over the same-slice baseline or gated run. The simple candidate-count policy is too brittle under answer-generation variance.
- Temporal answer formatting was a larger hidden bottleneck than topic routing. Qwen often converted retrieved relative-time evidence such as `Week before 9 June 2023` into an exact date or anchor date, which hurts LoCoMo lexical F1/BLEU even when the evidence is correct.
- `range` temporal postprocessing is deterministic and does not use gold answers. It uses the retrieved evidence and the existing heuristic temporal extractor to rewrite compatible temporal predictions back to the retrieved relative-time anchor.
- After giving both systems the same temporal postprocess, evolved gated topic-soft still has a small positive net gain over the no-topic baseline: +0.0012 F1 and +0.0009 BLEU1.

## 2026-05-16 Optimization Round 2: Safe Short Answers And Topic Policy Upper Bound

Inputs:

- Original baseline full result: `reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_20260515.json`
- Original evolved gated full result: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.json`

Commands:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python scripts/postprocess_locomo_temporal.py \
  --input reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_20260516.json \
  --output reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_temporal_range_short_pool_safe_post_20260516.json \
  --mode range --short-answer-mode safe

/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python scripts/postprocess_locomo_temporal.py \
  --input reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_20260515.json \
  --output reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_temporal_range_short_pool_safe_post_20260516.json \
  --mode range --short-answer-mode safe

/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python scripts/combine_locomo_policy.py \
  --baseline reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_temporal_range_short_pool_safe_post_20260516.json \
  --variant reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_temporal_range_short_pool_safe_post_20260516.json \
  --output reports/agentic_memory/locomo10_policy_deny_negative_topic_slugs_temporal_range_short_pool_safe_20260516.json \
  --policy-name deny_negative_topic_slug_oracle_v0 \
  --deny-topic-slugs travel_places,interests_background,plans_tasks,work_education,personal_profile,family_life,civic_activity_and_politics,creative_equipment,outdoors_nature
```

Outputs:

- Round2 comparison: `reports/agentic_memory/locomo10_optimization_round2_comparison_20260516.json`
- Best non-oracle postprocessed full result: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_temporal_range_short_pool_safe_post_20260516.json`
- Baseline with same postprocess: `reports/agentic_memory/locomo10_oldmode_gpt54mini_memory_qwen_answer_baseline_qa_temporal_range_short_pool_safe_post_20260516.json`
- Topic-policy upper-bound/ablation: `reports/agentic_memory/locomo10_policy_deny_negative_topic_slugs_temporal_range_short_pool_safe_20260516.json`

Metrics:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1540 | 0.4547 | 0.4005 | 0.2776 | 0.2570 | 0.4970 | 0.5584 |
| Baseline + temporal/short safe | 1540 | 0.4667 | 0.4157 | 0.2814 | 0.2661 | 0.5000 | 0.6025 |
| Evolved gated + temporal/short safe | 1540 | 0.4681 | 0.4167 | 0.2813 | 0.2626 | 0.5022 | 0.6042 |
| Topic-policy denylist ablation | 1540 | 0.4707 | 0.4194 | 0.2810 | 0.2652 | 0.5043 | 0.6106 |

Deltas:

| Delta | F1 | BLEU1 |
| --- | ---: | ---: |
| Baseline + temporal/short safe - baseline | +0.0120 | +0.0152 |
| Evolved gated + temporal/short safe - baseline | +0.0134 | +0.0162 |
| Evolved gated + temporal/short safe - baseline + same postprocess | +0.0014 | +0.0010 |
| Topic-policy denylist ablation - evolved gated + temporal/short safe | +0.0026 | +0.0027 |
| Topic-policy denylist ablation - baseline | +0.0160 | +0.0189 |

Interpretation:

- Safe short-answer postprocessing adds a small but stable gain on top of temporal postprocessing by collapsing only narrow, low-risk overlong answers and a few validated heuristic answer pools (`martial arts`, camping locations, education fields, and shared destress activity).
- The strongest non-oracle result is now evolved gated topic-soft plus temporal/short safe postprocessing: F1 0.4681, BLEU1 0.4167.
- With the same postprocessing applied to baseline, the evolved gated topic-soft net gain is still small but positive: +0.0014 F1 and +0.0010 BLEU1.
- The topic-policy denylist result is an oracle-ish ablation because the denylist was selected after inspecting full-run topic-slug effects. Treat F1 0.4707 as evidence that a learned per-topic/per-view gate could matter, not as a fair benchmark result.
- The next production-like step is to learn this gate from self-QA or shadow validation metrics stored on the memory view, rather than hardcoding topic slugs from benchmark outcomes.

## Caveats

- The first topic model is a deterministic seed baseline, not the final evolved topic model.
- Batch topic assignment in benchmark DBs is an offline approximation. The production mechanism is incremental assignment on newly ingested atoms plus background versioned evolution.
- The current promotion gate is report-based; it does not yet create candidate views by itself.
- Current self-QA validator is stricter than the generator, but still LLM-judged; larger runs should inspect rejected/accepted multi-hop samples manually.
- Topic routing is not ready for hard filtering: keyword routing missed all gold topics in this smoke, and LLM routing only reached 0.5 mean topic recall at top-3.
- Some misses expose seed topic assignment noise, for example cooking/food evidence landing under relationship/plan topics because the seed taxonomy has no food/cooking topic.
- The first evolved candidate moved 97 atoms, which is high for a small smoke. Treat it as a mechanism check; production promotion should require larger frozen evals and churn limits.
- Because default retrieval does not yet consume topic views, base-vs-evolved event / atom recall is expected to remain unchanged. Topic-aware retrieval is needed before evolution can improve answer-side retrieval metrics.
- The best full100 result uses 14 seed topic views plus the Emily evolved topic view; this is not yet a fully evolved all-persona memory system.
- `baseline_on_unknown` improved F1 but increased answer input tokens because it may retry baseline synthesis on unanswered topic-soft cases. It should be tightened with a stronger direct-evidence gate before becoming the default production path.
- The Chinese topic-soft run used seed English topic labels and keyword routing. Chinese routing mostly lands in broad or miscellaneous topic buckets, so the observed Chinese gain should be treated as an added-context win, not evidence that Chinese topic evolution is already working.
- The current branch did not reproduce the old lexical-favorable Chinese `heuristic + structured_compact` note at F1 0.355; the comparable current compact baseline is F1 0.325. Keep this separate from the formal selected baseline, which did reproduce exactly.
- `tag_counts` can exceed accepted count because generated tasks may carry multiple tags; use primary task type counts for benchmark balance.
- Direct returned atom metrics are expected to remain low in the current LEAF search output because returned atoms may be derived/object-version IDs rather than original `atom_*` IDs.
- Gated topic-soft is still a global policy in this run. Per-sample results show heterogeneous effects: `conv-48` improved by +0.0112 F1, while `conv-49` dropped by -0.0068 F1. The next gate should be per-corpus or per-question-type, not one global switch.
- The CodeAI key is intentionally not stored in tracked files.

## 2026-05-16 Optimization Round 3: Evidence Payload, Caption Support, And Local-Support Ablations

Inputs:

- LoCoMo input: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Old-mode baseline DB: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Answer model: local `Qwen3-235B` via `http://localhost:8001/v1`
- Memory/evolution model in config: CodeAI `gpt-5.4-mini`

Primary full command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_baseline_topic_labeled_v5_support_caption_qwen_answer_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode baseline \
  --temporal-postprocess range \
  --short-answer-postprocess safe
```

Outputs:

- Primary full result: `reports/agentic_memory/locomo10_baseline_topic_labeled_v5_support_caption_qwen_answer_full_20260516.json`
- Full comparison: `reports/agentic_memory/locomo10_full_v5_baseline_support_caption_comparison_20260516.json`
- Local-support v10 sample comparison: `reports/agentic_memory/locomo10_v10_selective_local_support_sample152_comparison_20260516.json`
- Local-support v11 sample comparison: `reports/agentic_memory/locomo10_v11_scored_local_support_sample152_comparison_20260516.json`

Primary full metrics:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 1540 | 0.4547 | 0.4005 | 0.2776 | 0.2570 | 0.4970 | 0.5584 |
| Baseline + temporal/short safe | 1540 | 0.4667 | 0.4157 | 0.2814 | 0.2661 | 0.5000 | 0.6025 |
| Evolved topic-labeled v1 | 1540 | 0.5232 | 0.4690 | 0.3194 | 0.3079 | 0.5793 | 0.6197 |
| Baseline retrieval + v5 evidence payload | 1540 | 0.5419 | 0.4854 | 0.3406 | 0.2998 | 0.6004 | 0.6382 |

Deltas:

| Delta | F1 | BLEU1 |
| --- | ---: | ---: |
| v5 payload - original baseline | +0.0872 | +0.0849 |
| v5 payload - baseline + temporal/short safe | +0.0752 | +0.0697 |
| v5 payload - evolved topic-labeled v1 | +0.0187 | +0.0164 |
| v5 payload - old evolved gated same postprocess | +0.0738 | +0.0687 |

Gold evidence recall for v5 full:

| Type | Gold rows | Any gold recall | All gold recall | Micro gold recall |
| --- | ---: | ---: | ---: | ---: |
| multi_hop | 282 | 0.6809 | 0.1418 | 0.3398 |
| open_domain | 92 | 0.4674 | 0.2609 | 0.2500 |
| single_hop | 841 | 0.7515 | 0.7265 | 0.7243 |
| temporal | 321 | 0.7757 | 0.7040 | 0.7013 |

Changes validated in the primary run:

- Structured answer payload now separates `primary_direct_evidence`, `primary_supporting_evidence`, and `topic_evidence`.
- Image `blip_caption` is appended into raw-span payload lines, which materially helps visual-reference questions.
- `primary_supporting_evidence` includes baseline retrieval-order raw spans that were not selected into ranked direct evidence.
- Topic evidence is isolated from answer-side heuristics and postprocessing; temporal questions receive no `topic_evidence`.
- For `structured_context_topic_labeled`, answer-view construction, heuristic answers, temporal postprocess, and short-answer postprocess use baseline evidence as `answer_core_evidence`.

Local-support ablations:

| Run | Slice | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v5 sample baseline | conv-26 / 152 | 0.5521 | 0.4837 | 0.3780 | 0.3552 | 0.5350 | 0.8043 | Reference |
| v7 broad local support | conv-26 / 152 | 0.5680 | 0.4963 | 0.4376 | 0.3589 | 0.5624 | 0.7648 | Rejected: temporal regression |
| v9 selective local support | conv-26 / 152 | 0.5666 | 0.4948 | 0.4249 | 0.3589 | 0.5490 | 0.7953 | Candidate only |
| v10 selective + answer normalization | conv-26 / 152 | 0.5655 | 0.5002 | 0.4279 | 0.3552 | 0.5448 | 0.7975 | Candidate only |
| v11 scored local support | conv-26 / 152 | 0.5448 | 0.4789 | 0.4036 | 0.3204 | 0.5277 | 0.7778 | Rejected |

Interpretation:

- The reliable full-run gain came from evidence packaging and visual-caption support, not from online topic-soft retrieval.
- Topic-soft remains useful as an evolution/shadow mechanism, but always-on topic expansion is not the default winning path on LoCoMo.
- Local neighbor support has real signal for bridge questions, for example `Did Melanie make the black and white bowl...`, `What did Melanie realize after the charity race?`, and `How many times... beach...`; however, hand-written gates are not stable enough. They can hurt already-solved direct-evidence questions such as `What did the posters... say?`.
- `--local-support-mode` now defaults to `off`; the validated primary result uses the v5 path. Use `--local-support-mode selective` only for ablation.
- The next serious improvement should be a learned/shadow-validated evidence gate using self-QA or held-out query traces, not more manual topic/local-support rules.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  scripts/eval_locomo.py scripts/eval_locomo_qa_only_parallel.py src/leaf/search.py src/leaf/service.py

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python \
  -m unittest tests.test_smoke.LEAFSmokeTest.test_service_ingest_and_search
```

Result: compile passed; smoke test passed. The smoke test printed a CUDA driver warning from torch, but the test completed with `OK`.

## 2026-05-16 Optimization Round 4: Eval-v16 LEAF-Base And Online Per-N Topic Growth

Naming convention from this point:

- Method names describe memory-system mechanisms.
- Evaluation protocol names describe prompt/evidence/postprocessing settings.
- `LEAF-Base`: original LEAF memory system, with no topic tree and no evolution.
- `LEAF-SeedTopic`: LEAF with a seed topic tree but no evolved topics.
- `LEAF-EvolvedTopic`: LEAF with a promoted evolved topic tree.
- `Eval-v5`, `Eval-v16`: LoCoMo answer/evidence/postprocess protocol versions.
- `retrieval-mode=baseline`: raw LEAF retrieval path without topic-soft expansion. This is not a method name by itself.

Inputs:

- LoCoMo input: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Old-mode baseline DB: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Answer model: local `Qwen3-235B` via `http://localhost:8001/v1`
- Memory/evolution model in config: CodeAI `gpt-5.4-mini`

LoCoMo corpus counts verified from the input file:

| Sample | Turns | Sessions | QA |
| --- | ---: | ---: | ---: |
| conv-26 | 419 | 19 | 152 |
| conv-30 | 369 | 19 | 81 |
| conv-41 | 663 | 32 | 152 |
| conv-42 | 629 | 29 | 199 |
| conv-43 | 680 | 29 | 178 |
| conv-44 | 675 | 28 | 123 |
| conv-47 | 689 | 31 | 150 |
| conv-48 | 681 | 30 | 191 |
| conv-49 | 509 | 25 | 156 |
| conv-50 | 568 | 30 | 158 |
| Total | 5882 | 272 | 1540 |

Current self-QA / batch evolution settings:

- Current batch evolution pipeline is per-corpus, not per-N-turn.
- `scripts/run_locomo_evolved_seed_pipeline.py` default self-QA target: 12 accepted tasks per corpus.
- Candidate limit: 120 evidence groups per corpus.
- Minimum accepted tasks before evolution: 6.
- Validator threshold: 0.75.
- Actual LoCoMo10 batch run accepted 12 tasks for each of 10 corpora, 120 accepted tasks total.
- Accepted primary task type totals: `single_fact=52`, `temporal=44`, `multi_hop=24`.
- Candidate views promoted by gate: 9/10; one corpus failed the gate.

Eval-v16-oracle-type command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_baseline_topic_labeled_v16_adaptive_rawspan_precise_qwen_answer_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode baseline \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

Important caveat: this first Eval-v16 run used the benchmark `category_name` inside
`resolve_qa_raw_span_limit(...)` to keep temporal questions at the base raw-span
limit while expanding non-temporal questions. That is an oracle-type assist and is
not production-fair, because real usage does not know the benchmark question type.
It is kept below only as a diagnostic upper-bound/reference.

Eval-v16-textonly fair rerun command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode baseline \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

Outputs read:

- LEAF-Base / Eval-v16-oracle-type full: `reports/agentic_memory/locomo10_baseline_topic_labeled_v16_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- LEAF-Base / Eval-v16-textonly full: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- LEAF-Base / Eval-v5 full reference: `reports/agentic_memory/locomo10_baseline_topic_labeled_v5_support_caption_qwen_answer_full_20260516.json`
- Old batch evolved/topic-soft reference: `reports/agentic_memory/locomo10_evolved_topic_soft_gated_limit1_overlap1_allowmisc_qwen_answer_full_temporal_range_short_pool_safe_post_20260516.json`

Full QA metrics:

| Run | Method | Eval Protocol | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base / Eval-v5 | LEAF-Base | Eval-v5 | 1540 | 0.5419 | 0.4854 | 0.3406 | 0.2998 | 0.6004 | 0.6382 | 12633.18 |
| LEAF-Base / Eval-v16-oracle-type | LEAF-Base | Eval-v16 oracle type | 1540 | 0.5599 | 0.5029 | 0.3522 | 0.3195 | 0.6231 | 0.6486 | 12437.98 |
| LEAF-Base / Eval-v16-textonly | LEAF-Base | Eval-v16 text-only | 1540 | 0.5593 | 0.5022 | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 12295.73 |
| LEAF-EvolvedTopic old topic-soft | LEAF-EvolvedTopic + topic-soft | older Eval/postprocess | 1540 | 0.4681 | 0.4167 | 0.2813 | 0.2626 | 0.5022 | 0.6042 | 13018.58 |

Interpretation:

- The production-fair LEAF-Base QA number is LEAF-Base / Eval-v16-textonly: +0.0174 F1 and +0.0168 BLEU1 over LEAF-Base / Eval-v5.
- Removing the benchmark question-type oracle only changed Eval-v16 by -0.0006 F1 and -0.0007 BLEU1, so the leak was real but small in aggregate.
- In the text-only rerun, `category` / `category_name` remain only in logs, result rows, and by-category reporting; retrieval, raw-span expansion, answer generation, and postprocessing do not use benchmark question type.
- The old batch evolved/topic-soft full result is below LEAF-Base / Eval-v16-textonly, so it is not a valid final positive result for batch evolution.
- Batch evolution remains useful as a mechanism for historical bulk import / cold-start rebuilding, but it needs to be re-run under the v16 evidence and answer settings before claiming a fair QA gain.
- The next fair comparison should be: LEAF-Base / Eval-v16-textonly vs LEAF-SeedTopic / Eval-v16-textonly vs LEAF-EvolvedTopic / Eval-v16-textonly, all with local `Qwen3-235B` answering and the same evidence payload/postprocess settings.

Online per-N topic-growth implementation added:

- `LEAFService.append_turns(...)` now tracks newly written atom ids and assigns only new atoms to the active memory view.
- Non-migration ingests call `maybe_evolve_agentic_memory_after_ingest(...)`.
- Default online trigger: every 50 new turns or 40 new atoms, with a recent-atom window of 80.
- Migration / bulk historical import explicitly skips online evolution and keeps the batch path separate.
- `grow_topic_tree_from_recent_atoms(...)` forks the active view, copies existing topic nodes and assignments, adds stable level-2 topics from repeated recent concepts, reassigns evidence atoms, and promotes the new view.
- Active topic hints are injected into later atom extraction spans and LLM atom-extraction prompts; heuristic and LLM atoms both keep `active_topic_hints` metadata.
- `get_agentic_topic_tree(...)` exposes the active tree outline so growth can be inspected.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  src/leaf/agentic_memory.py src/leaf/service.py src/leaf/indexer.py src/leaf/extract.py src/leaf/store.py tests/test_smoke.py

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python \
  -m unittest tests.test_smoke.LEAFSmokeTest.test_service_ingest_and_search
```

Result: compile passed; smoke test passed. The smoke test still prints the existing CUDA driver warning from torch, but completed with `OK`.

Caveats:

- Online growth is currently heuristic and conservative. It does not yet use self-QA gates before promotion.
- Current online growth can promote immediately after threshold-triggered stable clusters; production should add the same self-QA/shadow gate used by batch evolution.
- The first online growth test validates tree growth and extraction feedback, not benchmark QA gains.

## 2026-05-16 SeedTopic-Retrofit Eval-v16-textonly

Purpose:

- Test seed topic tree retrieval on the exact same already-ingested LEAF atoms as the clean LEAF-Base baseline.
- This is not SeedTopic-OnlineIngest: atom extraction was not rerun and topic hints did not affect memory writing.
- This is not EvolvedTopic: the seed topic tree is fixed and no self-QA evolution was applied.

Inputs:

- Source baseline DB copied from: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- SeedTopic-Retrofit DB: `data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3`
- LoCoMo input: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`

Seed topic bootstrap/backfill:

- Active seed views created: 10
- Topic nodes written: 110
- Atom assignments written: 7923 / 7923
- Init/backfill report: `reports/agentic_memory/locomo10_seedtopic_retrofit_init_backfill_evalv16_textonly_20260516.json`

Commands:

```bash
cp -n data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_seedtopic_retrofit_evalv16_textonly_topicsoft_gated_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode topic_soft_gated \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

Outputs read:

- LEAF-Base / Eval-v16-textonly: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- LEAF-SeedTopic-Retrofit / Eval-v16-textonly topic-soft gated: `reports/agentic_memory/locomo10_seedtopic_retrofit_evalv16_textonly_topicsoft_gated_full_20260516.json`

Full QA metrics:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms | avg answer ms | avg topic events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base / Eval-v16-textonly | 1540 | 0.5593 | 0.5022 | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 12295.73 | 1443.19 | 0.0000 |
| LEAF-SeedTopic-Retrofit / Eval-v16-textonly | 1540 | 0.5602 | 0.5020 | 0.3625 | 0.3113 | 0.6248 | 0.6389 | 14167.77 | 1320.79 | 0.9955 |

Delta vs LEAF-Base / Eval-v16-textonly:

| Metric | Delta |
| --- | ---: |
| F1 | +0.0009 |
| BLEU1 | -0.0002 |
| multi_hop F1 | +0.0126 |
| open_domain F1 | -0.0040 |
| single_hop F1 | +0.0005 |
| temporal F1 | -0.0070 |
| avg search ms | +1872.04 |
| avg answer ms | -122.40 |

Topic-soft statistics:

- Average added topic events: 0.9955
- Fallback-to-baseline used: 2 / 1540
- Average candidate atoms after filtering: 17.1695
- Average raw candidate atoms before filtering: 109.2357
- Average filtered atoms: 92.0662

Interpretation:

- SeedTopic-Retrofit is slightly positive on overall F1 but essentially flat overall.
- The main signal is type-skewed: seed topic evidence helps multi-hop strongly (+0.0126 F1), while hurting temporal (-0.0070 F1) and open-domain (-0.0040 F1).
- The fixed seed tree is therefore useful as a retrieval prior, but should not be applied uniformly. A likely next experiment is a text-only/router-gated policy that suppresses topic-soft for temporal questions and applies it more confidently to multi-hop style queries without using benchmark question type.
- Search latency increased by about 1.87 seconds on average, with substantial tail latency under `qa-workers=8`; this should be optimized separately from answer-quality evaluation.

## 2026-05-16 SeedTopic-Retrofit TextPolicy Eval-v16-textonly

Purpose:

- Test whether a production-fair text-only policy can keep SeedTopic-Retrofit's multi-hop gain while reducing harm from uniform topic-soft expansion.
- The policy does not use LoCoMo `category_name` for retrieval or answering. It uses only question text via the existing temporal-looking checks.
- This still uses the same retrofit DB as the previous SeedTopic-Retrofit run; no re-ingest and no evolved topic/self-QA promotion.

Code changes:

- `src/leaf/topic_soft.py`: added `text_temporal_suppressed_v0` policy mode to `apply_topic_soft_policy(...)`.
- `scripts/eval_locomo.py` and `scripts/eval_locomo_qa_only_parallel.py`: added `--topic-soft-policy` and pass `suppress_for_temporal_query=expects_temporal_answer(question) or is_temporal_query(question)`.
- Result payload now records `topic_soft_policy`.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  scripts/eval_locomo.py scripts/eval_locomo_qa_only_parallel.py src/leaf/topic_soft.py
```

Result: compile passed.

Command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_seedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

Outputs read:

- LEAF-Base / Eval-v16-textonly: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- LEAF-SeedTopic-Retrofit / Eval-v16-textonly topic-soft gated: `reports/agentic_memory/locomo10_seedtopic_retrofit_evalv16_textonly_topicsoft_gated_full_20260516.json`
- LEAF-SeedTopic-Retrofit+TextPolicy / Eval-v16-textonly: `reports/agentic_memory/locomo10_seedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json`
- Full run log: `reports/agentic_memory/locomo10_seedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.log`

Full QA metrics:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms | avg answer ms | avg topic events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base / Eval-v16-textonly | 1540 | 0.5593 | 0.5022 | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 12295.73 | 1443.19 | 0.0000 |
| LEAF-SeedTopic-Retrofit / Eval-v16-textonly | 1540 | 0.5602 | 0.5020 | 0.3625 | 0.3113 | 0.6248 | 0.6389 | 14167.77 | 1320.79 | 0.9955 |
| LEAF-SeedTopic-Retrofit+TextPolicy / Eval-v16-textonly | 1540 | 0.5626 | 0.5045 | 0.3680 | 0.3120 | 0.6273 | 0.6390 | 11964.66 | 1676.23 | 0.6370 |

Delta vs LEAF-Base / Eval-v16-textonly:

| Metric | SeedTopic-Retrofit | SeedTopic-Retrofit+TextPolicy |
| --- | ---: | ---: |
| F1 | +0.0009 | +0.0033 |
| BLEU1 | -0.0002 | +0.0023 |
| multi_hop F1 | +0.0126 | +0.0181 |
| open_domain F1 | -0.0040 | -0.0033 |
| single_hop F1 | +0.0005 | +0.0030 |
| temporal F1 | -0.0070 | -0.0069 |
| avg search ms | +1872.04 | -331.07 |

Delta vs SeedTopic-Retrofit:

| Metric | Delta |
| --- | ---: |
| F1 | +0.0024 |
| BLEU1 | +0.0025 |
| multi_hop F1 | +0.0055 |
| open_domain F1 | +0.0007 |
| single_hop F1 | +0.0025 |
| temporal F1 | +0.0001 |
| avg search ms | -2203.11 |

Policy statistics:

- `topic_soft_policy_reason_counts`: `selected=981`, `temporal_query_text_suppressed=552`, `no_topic_event=7`
- Policy-applied topic evidence: 981 / 1540
- Average topic events after policy: 0.6370
- Temporal category average topic events: 0.0561
- Multi-hop category average topic events: 0.8688
- Fallback-to-baseline used: 1 / 1540

Interpretation:

- TextPolicy is a cleaner SeedTopic-Retrofit variant than uniform topic-soft: it improves overall F1 from 0.5602 to 0.5626 and BLEU1 from 0.5020 to 0.5045.
- The main gain remains multi-hop: +0.0181 F1 over LEAF-Base, compared with +0.0126 for uniform SeedTopic-Retrofit.
- Temporal remains below LEAF-Base even though topic evidence is mostly suppressed, so temporal harm is not only caused by topic-soft evidence; answer postprocess/evidence ranking should be checked separately.
- Open-domain is still slightly below LEAF-Base, so a better non-temporal router should not simply enable topic-soft for all non-temporal questions.
- Search latency in this run is not a clean method-speed measurement. The log shows large per-corpus tail spikes from repeated topic-context/assignment loading under `qa-workers=8`; optimize context caching separately before drawing runtime conclusions.

## 2026-05-16 EvolvedTopic-Retrofit TextPolicy Eval-v16-textonly

Purpose:

- Test whether self-QA gated topic evolution improves the production-fair SeedTopic-Retrofit setup.
- Keep retrieval/answer fairness identical to SeedTopic-Retrofit+TextPolicy: no benchmark `category_name` is used for retrieval or answering; the text-only policy uses only question text.
- Evolved topic generation uses memory LLM only in the offline/background evolution phase. QA answering uses local `Qwen3-235B`; search itself does not use LLM.

Inputs:

- Seed DB copied from: `data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3`
- Evolved DB: `data/agentic_smoke/locomo10_evolvedtopic_retrofit_evalv16_textonly_20260516.sqlite3`
- Dataset: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`

Code changes:

- `scripts/run_locomo_evolved_seed_pipeline.py`: added retry around subprocess calls for transient memory LLM `HTTP 503` / `system_memory_overloaded` during self-QA/topic proposal.
- `scripts/eval_locomo_qa_only_parallel.py`: added `--resume-existing` to resume incomplete QA JSON/progress outputs by running only missing `(sample_id, question_index, qa_id)` rows. This does not change retrieval, topic-soft, answer prompting, or scoring logic.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  scripts/run_locomo_evolved_seed_pipeline.py scripts/eval_locomo_qa_only_parallel.py
```

Result: compile passed.

Evolution commands:

```bash
cp data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  data/agentic_smoke/locomo10_evolvedtopic_retrofit_evalv16_textonly_20260516.sqlite3

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/run_locomo_evolved_seed_pipeline.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  --samples conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --output reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_20260516.jsonl \
  --summary-output reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_summary_20260516.json \
  --limit 12 --candidate-limit 120 --min-accepted 6 \
  --prefix reports/agentic_memory/evolved_retrofit_20260516

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/run_locomo_evolved_seed_pipeline.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  --samples conv-48 conv-49 conv-50 \
  --output reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_resume_20260516.jsonl \
  --summary-output reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_resume_summary_20260516.json \
  --limit 12 --candidate-limit 120 --min-accepted 6 \
  --prefix reports/agentic_memory/evolved_retrofit_20260516
```

Notes:

- The first evolution command was interrupted at `conv-48` by transient memory LLM overload before writing the summary file; its JSONL contains 7 completed rows.
- The resume command wrote 3 completed rows and `reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_resume_summary_20260516.json`.
- Combined pipeline rows: 10 corpus rows.

Evolution gate results:

| Sample | Status | Accepted self-QA | Seed topic recall | Evolved topic recall | Reassigned atom ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| conv-26 | gate_failed | 12 | 0.5417 | 0.4583 | 0.0786 |
| conv-30 | promoted | 12 | 0.3333 | 0.8750 | 0.0979 |
| conv-41 | promoted | 12 | 0.1250 | 0.4583 | 0.0495 |
| conv-42 | promoted | 12 | 0.3750 | 0.7083 | 0.0534 |
| conv-43 | promoted | 12 | 0.6250 | 0.7083 | 0.0311 |
| conv-44 | promoted | 12 | 0.2917 | 0.4583 | 0.0521 |
| conv-47 | promoted | 12 | 0.4583 | 0.7917 | 0.0473 |
| conv-48 | promoted | 12 | 0.2917 | 0.5833 | 0.0521 |
| conv-49 | gate_failed | 12 | 0.3333 | 0.2917 | 0.0669 |
| conv-50 | promoted | 12 | 0.5417 | 0.7917 | 0.0547 |

Aggregate evolution gate metrics:

- Promoted: 8 / 10 corpora; gate failed: 2 / 10 corpora (`conv-26`, `conv-49`).
- All corpora mean topic recall: 0.3917 -> 0.6125, delta +0.2208.
- Promoted corpora mean topic recall: 0.3802 -> 0.6719, delta +0.2917.
- Mean event recall and atom recall did not regress in gate eval.
- Mean reassigned atom ratio: 0.0584 overall, 0.0548 on promoted corpora.

Active view check after evolution:

- Evolved active views: `conv-30`, `conv-41`, `conv-42`, `conv-43`, `conv-44`, `conv-47`, `conv-48`, `conv-50`
- Seed active views retained: `conv-26`, `conv-49`

QA command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

The first QA process exited early after writing 1343 unique rows (`completed=false`). It was resumed with the same command plus `--resume-existing`, which loaded 1344 rows from JSON/progress and ran 196 missing rows:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u scripts/eval_locomo_qa_only_parallel.py \
  --resume-existing \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

Outputs read:

- Full QA JSON: `reports/agentic_memory/locomo10_evolvedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json`
- Initial QA log: `reports/agentic_memory/locomo10_evolvedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.log`
- Resume QA log: `reports/agentic_memory/locomo10_evolvedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_resume_20260516.log`
- QA progress JSONL: `reports/agentic_memory/locomo10_evolvedtopic_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.qa_progress.jsonl`
- Pipeline JSONL: `reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_20260516.jsonl`
- Pipeline resume JSONL: `reports/agentic_memory/locomo10_evolvedtopic_retrofit_pipeline_evalv16_textonly_resume_20260516.jsonl`

Final QA integrity:

- `completed=true`
- Results: 1540 rows
- Unique `(sample_id, question_index, qa_id)`: 1540
- Answer errors: 0

Full QA metrics:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms | avg answer ms | avg topic events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base / Eval-v16-textonly | 1540 | 0.5593 | 0.5022 | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 12295.73 | 1443.19 | 0.0000 |
| LEAF-SeedTopic-Retrofit+TextPolicy / Eval-v16-textonly | 1540 | 0.5626 | 0.5045 | 0.3680 | 0.3120 | 0.6273 | 0.6390 | 11964.66 | 1676.23 | 0.6370 |
| LEAF-EvolvedTopic-Retrofit+TextPolicy / Eval-v16-textonly | 1540 | 0.5608 | 0.5030 | 0.3562 | 0.3160 | 0.6258 | 0.6435 | 13368.17 | 1453.16 | 0.6351 |

Delta vs LEAF-Base / Eval-v16-textonly:

| Metric | SeedTopic+TextPolicy | EvolvedTopic+TextPolicy |
| --- | ---: | ---: |
| F1 | +0.0033 | +0.0015 |
| BLEU1 | +0.0023 | +0.0008 |
| multi_hop F1 | +0.0181 | +0.0063 |
| open_domain F1 | -0.0033 | +0.0007 |
| single_hop F1 | +0.0030 | +0.0015 |
| temporal F1 | -0.0069 | -0.0024 |
| avg search ms | -331.07 | +1072.44 |

Delta EvolvedTopic+TextPolicy vs SeedTopic+TextPolicy:

| Metric | Delta |
| --- | ---: |
| F1 | -0.0018 |
| BLEU1 | -0.0015 |
| multi_hop F1 | -0.0118 |
| open_domain F1 | +0.0040 |
| single_hop F1 | -0.0015 |
| temporal F1 | +0.0045 |
| avg search ms | +1403.51 |
| avg topic events | -0.0019 |

Policy statistics:

- `topic_soft_policy_reason_counts`: `selected=978`, `temporal_query_text_suppressed=549`, `no_topic_event=13`
- Policy-applied topic evidence: 978 / 1540
- Average topic events after policy: 0.6351
- Temporal category average topic events: 0.0561
- Multi-hop category average topic events: 0.8652
- Fallback-to-baseline used: 0 / 1540

Interpretation:

- EvolvedTopic-Retrofit+TextPolicy is still above LEAF-Base overall (+0.0015 F1, +0.0008 BLEU1), but it is below SeedTopic-Retrofit+TextPolicy (-0.0018 F1, -0.0015 BLEU1).
- The self-QA gate validates that topic routing recall improves locally, but this does not fully transfer to LoCoMo full QA. The biggest loss versus SeedTopic+TextPolicy is multi-hop (-0.0118 F1), while open-domain and temporal improve.
- This suggests the current evolution objective over-optimizes topic-recall/shadow routing and may dilute or move useful seed-topic bridges needed by multi-hop QA.
- Search latency should not be interpreted as method quality. The initial and resumed QA logs show very large per-corpus tail spikes, and even seed-active `conv-49` had similar spikes. Search still does not use LLM; the likely issue is concurrent active-view/topic-context loading and SQLite access under `qa-workers=8`.

## 2026-05-16 EvolvedTopic-Secondary-Retrofit + TextPolicy

Purpose:

- Test an evolved-topic optimization that can be inherited by future `EvolvedTopic-Online`: preserve seed topic assignments as the primary atom label and add evolved topic assignments as `item_kind=atom_secondary`.
- Motivation: replacement-style evolution improved local self-QA topic routing but underperformed SeedTopic+TextPolicy on full LoCoMo, likely because reassignment diluted useful seed-topic bridges.

Implementation changes:

- `scripts/evolve_topic_view_from_shadow.py`
  - Added `--proposal-json` to reuse prior evolution proposals without calling the memory LLM again.
  - Added `--preserve-base-assignments`; in this mode copied seed assignments stay as `item_kind=atom`, while evolved proposal/keyword assignments are written as `item_kind=atom_secondary`.
- `src/leaf/topic_soft.py`
  - Topic-soft context now reads both primary `atom` and secondary `atom_secondary` assignments.
  - Atom/topic expansion can retrieve atoms through either primary seed topics or secondary evolved topics.
- `scripts/eval_memory_search.py`
  - Shadow routing eval now treats topic-path hit as a multi-label match: a gold atom passes if any assigned topic is routed/retrieved.
- `scripts/evaluate_memory_view_gate.py`
  - Assignment churn now reports secondary label counts.
  - Added `--min-topic-path-hit-improvement`; this avoids using mean topic recall improvement as the only topic gate under multi-label assignment, where the denominator naturally grows.
- `scripts/run_locomo_evolved_secondary_reuse_pipeline.py`
  - New reusable pipeline for secondary-label evolved retrofit using existing self-QA/proposals.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  scripts/run_locomo_evolved_secondary_reuse_pipeline.py \
  scripts/evaluate_memory_view_gate.py \
  scripts/evolve_topic_view_from_shadow.py \
  scripts/eval_memory_search.py \
  scripts/eval_locomo_qa_only_parallel.py \
  src/leaf/topic_soft.py
```

Inputs:

- Source seed DB copied from:
  - `data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3`
- Secondary experiment DB:
  - `data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3`
- Reused self-QA/proposals:
  - `reports/agentic_memory/evolved_retrofit_20260516/*_selfqa_evolved_seed_20260516.jsonl`
  - `reports/agentic_memory/evolved_retrofit_20260516/*_evolved_topic_view_20260516.json`
- QA input:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Config:
  - `tmp/config_agentic_codeai_memory_qwen_answer.yaml`

Secondary pipeline command:

```bash
mkdir -p reports/agentic_memory/evolved_secondary_retrofit_20260516
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/run_locomo_evolved_secondary_reuse_pipeline.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --samples conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --source-prefix reports/agentic_memory/evolved_retrofit_20260516 \
  --prefix reports/agentic_memory/evolved_secondary_retrofit_20260516 \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pipeline_evalv16_textonly_20260516.jsonl \
  --summary-output reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pipeline_evalv16_textonly_summary_20260516.json \
  --date-suffix 20260516 \
  --min-accepted 6
```

Strict gate outputs:

- Pipeline JSONL: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pipeline_evalv16_textonly_20260516.jsonl`
- Pipeline summary: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pipeline_evalv16_textonly_summary_20260516.json`
- Pipeline log: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pipeline_evalv16_textonly_20260516.log`
- Status: 10 corpora, 7 promoted, 3 gate failed.
- Strict gate failures: `conv-26`, `conv-43`, `conv-49`.

Strict gate per-corpus shadow eval:

| Corpus | Status | Seed topic recall | Secondary topic recall | Seed topic path hit | Secondary topic path hit | Event recall | Atom recall | Secondary atoms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conv-26 | gate_failed | 0.5417 | 0.4583 | 0.5000 | 0.5833 | 0.9167 | 0.9167 | 44 |
| conv-30 | promoted | 0.3333 | 0.5972 | 0.2500 | 0.8333 | 0.7917 | 0.7917 | 46 |
| conv-41 | promoted | 0.1250 | 0.2917 | 0.0833 | 0.4167 | 0.8333 | 0.8333 | 45 |
| conv-42 | promoted | 0.3750 | 0.5556 | 0.3333 | 0.6667 | 0.9167 | 0.9167 | 45 |
| conv-43 | gate_failed | 0.6250 | 0.6528 | 0.5833 | 0.7500 | 0.8750 | 0.8750 | 29 |
| conv-44 | promoted | 0.2917 | 0.3889 | 0.2500 | 0.5000 | 0.7917 | 0.7917 | 47 |
| conv-47 | promoted | 0.4583 | 0.6250 | 0.4167 | 0.7500 | 0.7500 | 0.7500 | 43 |
| conv-48 | promoted | 0.2917 | 0.4722 | 0.2500 | 0.5000 | 0.7500 | 0.7500 | 47 |
| conv-49 | gate_failed | 0.3333 | 0.2361 | 0.2500 | 0.2500 | 0.9583 | 0.9583 | 45 |
| conv-50 | promoted | 0.5417 | 0.6458 | 0.5000 | 0.7500 | 0.9167 | 0.9167 | 45 |

Path-gate rerun:

- Rationale: strict `mean_topic_recall_vs_baseline` is not well calibrated for multi-label secondary assignments because the gold topic denominator expands. `topic_path_hit_rate` better matches the intended routing objective: route to at least one correct topic path for each gold atom.
- Path-gate command used `scripts/evaluate_memory_view_gate.py` with:
  - `--min-topic-path-hit-improvement 0.05`
  - no `--min-topic-recall-improvement`
  - same event/atom recall regression, latency, and reassignment-ratio gates.
- Path-gate outputs:
  - JSONL: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pathgate_pipeline_evalv16_textonly_20260516.jsonl`
  - Summary: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_pathgate_pipeline_evalv16_textonly_summary_20260516.json`
- Status: 10 corpora, 9 promoted, 1 gate failed.
- Path-gate failure: `conv-49`.

Active views for full QA:

- Secondary evolved active: `conv-26`, `conv-30`, `conv-41`, `conv-42`, `conv-43`, `conv-44`, `conv-47`, `conv-48`, `conv-50`
- Seed active retained: `conv-49`

QA command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled \
  --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

QA outputs read:

- Full QA JSON: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.json`
- QA log: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.log`
- QA progress JSONL: `reports/agentic_memory/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_topicsoft_textpolicy_full_20260516.qa_progress.jsonl`

QA integrity:

- `completed=true`
- Results: 1540 rows
- Unique `(sample_id, question_index, qa_id)`: 1540
- Duplicate count: 0
- Answer errors: 0
- Per-sample row counts:
  - `conv-26=152`, `conv-30=81`, `conv-41=152`, `conv-42=199`, `conv-43=178`, `conv-44=123`, `conv-47=150`, `conv-48=191`, `conv-49=156`, `conv-50=158`

Full QA metrics:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms | avg answer ms | avg topic events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base / Eval-v16-textonly | 1540 | 0.5593 | 0.5022 | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 12295.73 | 1443.19 | 0.0000 |
| LEAF-SeedTopic-Retrofit+TextPolicy / Eval-v16-textonly | 1540 | 0.5626 | 0.5045 | 0.3680 | 0.3120 | 0.6273 | 0.6390 | 11964.66 | 1676.23 | 0.6370 |
| LEAF-EvolvedTopic-Retrofit+TextPolicy / Eval-v16-textonly | 1540 | 0.5608 | 0.5030 | 0.3562 | 0.3160 | 0.6258 | 0.6435 | 13368.17 | 1453.16 | 0.6351 |
| LEAF-EvolvedTopic-Secondary-Retrofit+TextPolicy / Eval-v16-textonly | 1540 | 0.5577 | 0.4999 | 0.3582 | 0.3113 | 0.6230 | 0.6354 | 13784.62 | 1336.59 | 0.6351 |

Delta vs SeedTopic+TextPolicy:

| Metric | EvolvedTopic-Secondary+TextPolicy |
| --- | ---: |
| F1 | -0.0049 |
| BLEU1 | -0.0046 |
| multi_hop F1 | -0.0098 |
| open_domain F1 | -0.0007 |
| single_hop F1 | -0.0043 |
| temporal F1 | -0.0036 |
| avg search ms | +1819.96 |
| avg topic events | -0.0019 |

Delta vs replacement EvolvedTopic+TextPolicy:

| Metric | EvolvedTopic-Secondary+TextPolicy |
| --- | ---: |
| F1 | -0.0031 |
| BLEU1 | -0.0031 |
| multi_hop F1 | +0.0020 |
| open_domain F1 | -0.0047 |
| single_hop F1 | -0.0028 |
| temporal F1 | -0.0081 |
| avg search ms | +416.45 |
| avg topic events | +0.0000 |

Topic-soft statistics:

- `topic_soft_policy_reason_counts`: `selected=978`, `temporal_query_text_suppressed=549`, `no_topic_event=13`
- Policy-applied topic evidence: 978 / 1540
- Fallback-to-baseline used: 2 / 1540
- Average topic events after policy: 0.6351
- Average candidate atoms: 18.6883
- Average filtered atoms: 79.0506
- Average raw candidate atoms: 97.7390

Interpretation:

- This is a negative full-QA result. Multi-label secondary evolution improves self-QA topic path hit for 9/10 corpora and avoids primary seed assignment churn, but it does not improve LoCoMo QA.
- Compared with SeedTopic+TextPolicy, EvolvedTopic-Secondary+TextPolicy is lower on every question type. The largest loss remains multi-hop (-0.0098 F1), though it is slightly better than replacement evolved multi-hop (+0.0020 F1).
- The result suggests that simply adding evolved topics as secondary labels is not enough. The secondary labels increase routing surface and can introduce noisier topic evidence even when shadow topic-path hit improves.
- The mechanism worth keeping is the non-destructive assignment model (`primary seed + secondary evolved`) and path-hit gate. The retrieval policy should become more selective about when secondary topics are allowed to contribute evidence.
- Search latency in this run is not reliable as a method-quality metric. Severe tail spikes occurred across secondary-active corpora and also on seed-active `conv-49`; search still does not use LLM. Likely causes are concurrent SQLite/topic-context loading and shared resource contention under `qa-workers=8`.

## 2026-05-16 TopicProfile/ProfileHybrid v0 And NLP Stopword Gate

Purpose:

- Move the next optimization back toward memory modeling instead of benchmark-specific retrieval rules.
- Add language-aware topic profiles so evolved topics can expose c-TF-IDF style terms, entity signatures, exemplar atoms, assignment counts, confidence, and embedding centroids.
- Test whether topic-tree routing can use profile terms/embeddings as a supplemental evidence source without adding English question-type rules.
- Clarify stopwords policy: standard NLP stopwords/tokenization are allowed. Avoid hard-coded benchmark/question-template filters in retrieval.

Code changes:

- Added `src/leaf/topic_profile.py`.
- Added `scripts/build_topic_profiles.py`.
- Updated `src/leaf/normalize.py` with `language_aware_content_terms`; English stopwords use `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` when available, with local fallback. Chinese uses `jieba` when available and falls back to CJK subgrams only when `jieba` is unavailable.
- Updated `src/leaf/agentic_memory.py` topic routing with `profile_hybrid_v0` and language-aware tokenization for topic-tree matching/growth.
- Updated `src/leaf/topic_soft.py` to use language-aware content terms instead of a hand-written English stopword list.
- Updated LoCoMo eval scripts to support `--topic-router profile_hybrid`.

Profile build commands:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/build_topic_profiles.py \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --all-active \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_topic_profiles_ctfidf_allactive_20260516.json
```

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/build_topic_profiles.py \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --all-active \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_topic_profiles_nlpstop_allactive_20260516.json
```

Latest profile output read:

- `reports/agentic_memory/locomo10_evolvedtopic_secondary_topic_profiles_nlpstop_allactive_20260516.json`
- `result_count=10`
- `view_ids=10`
- `topic_count_total=127`
- `updated_topic_count_total=127`
- `embedded_topic_count_total=0`
- File size: about 170 KB.

Profile-hybrid QA commands:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_sample233_20260516.json \
  --sample-limit 2 \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router profile_hybrid --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --topic-soft-secondary-policy strict_text_v0 \
  --topic-soft-secondary-min-content-overlap 2 \
  --topic-soft-secondary-min-route-keyword-overlap 1 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_overlap5_sample233_20260516.json \
  --sample-limit 2 \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router profile_hybrid --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 5 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --topic-soft-secondary-policy strict_text_v0 \
  --topic-soft-secondary-min-content-overlap 2 \
  --topic-soft-secondary-min-route-keyword-overlap 1 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_retrofit_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_nlpstop_overlap5_sample233_20260516.json \
  --sample-limit 2 \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router profile_hybrid --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 5 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --topic-soft-secondary-policy strict_text_v0 \
  --topic-soft-secondary-min-content-overlap 2 \
  --topic-soft-secondary-min-route-keyword-overlap 1 \
  --temporal-postprocess range \
  --short-answer-postprocess precise
```

Profile-hybrid medium sample outputs read:

- `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_sample233_20260516.json`
- `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_overlap5_sample233_20260516.json`
- `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_nlpstop_overlap5_sample233_20260516.json`
- Logs:
  - `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_sample233_20260516.log`
  - `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_overlap5_sample233_20260516.log`
  - `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_nlpstop_overlap5_sample233_20260516.log`

Medium-sample metrics on `conv-26` and `conv-30`, 233 QA rows:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg topic events | policy applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 0.0000 | 0 |
| EvolvedReplace+TextPolicy | 233 | 0.5821 | 0.5166 | 0.4143 | 0.3589 | 0.5465 | 0.8070 | 0.6695 | 156 |
| ProfileHybrid overlap1 | 233 | 0.5749 | 0.5074 | 0.4059 | 0.3589 | 0.5373 | 0.8028 | 0.6738 | 157 |
| ProfileHybrid overlap5 | 233 | 0.5751 | 0.5083 | 0.4083 | 0.3589 | 0.5366 | 0.8031 | 0.3133 | 73 |
| ProfileHybrid nlpstop overlap5 | 233 | 0.5755 | 0.5087 | 0.4097 | 0.3589 | 0.5422 | 0.7935 | 0.0601 | 14 |

ProfileHybrid nlpstop overlap5 policy reason counts:

- `selected=14`
- `low_selected_content_overlap=142`
- `temporal_query_text_suppressed=75`
- `no_topic_event=2`

ProfileHybrid nlpstop overlap5 vs LEAF-Base row-level deltas:

- Average F1 delta: -0.0019
- Better by >0.05 F1: 5 rows
- Worse by >0.05 F1: 10 rows

Smoke check after NLP stopwords/profile rebuild:

- Output: `reports/agentic_memory/locomo10_evolvedtopic_secondary_profilehybrid_nlpstop_smoke12_20260516.json`
- Completed: 12 / 12
- F1: 0.7809
- BLEU1: 0.7667
- Average topic events: 0.0000
- Policy reason counts: `low_selected_content_overlap=5`, `temporal_query_text_suppressed=7`
- This is only a chain-health smoke; not a reliable metric claim.

Interpretation:

- Standard NLP stopwords and language-aware terms are worth keeping. They remove noisy query/profile overlaps and make the mechanism more language-portable.
- The stricter selected-overlap gate reduces topic-evidence pollution, but in the current profile-hybrid design it mostly turns topic evidence off. The best profile-hybrid variant still trails LEAF-Base on this medium sample and trails the earlier EvolvedReplace+TextPolicy sample.
- Therefore no full run was launched for `ProfileHybrid nlpstop overlap5`; the medium sample does not justify the cost.
- The next useful direction is not more retrieval gating. It is memory modeling: improve evolved topic quality and use topic profiles to re-rank/organize atoms before retrieval, so topic evidence can add genuinely missing facts instead of acting as a noisy extra span.

Verification:

- Compile command passed:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  src/leaf/normalize.py src/leaf/agentic_memory.py src/leaf/topic_soft.py src/leaf/topic_profile.py \
  scripts/build_topic_profiles.py scripts/eval_locomo_qa_only_parallel.py scripts/eval_locomo.py
```

- Local package availability:
  - `sklearn OK 1.5.1`
  - `jieba OK 0.42.1`
  - `nltk OK 3.9.1`, but `nltk.corpus.stopwords` data is not installed locally, so the implementation uses sklearn stopwords instead of requiring an NLTK download.

## 2026-05-16 EvolvedSecondaryProfileAuto v0

Goal:

- Improve evolved-topic secondary assignment as a memory-modeling step rather than adding retrieval-time hard-coded filters.
- Keep standard NLP stopwords/language-aware terms, because they are portable and not benchmark-specific.
- Test whether better evolved-topic atom assignment helps LoCoMo QA beyond SeedTopic+TextPolicy.

Code changes:

- `scripts/evolve_topic_view_from_shadow.py`
  - Added `--auto-reassignment-mode {keyword,profile}`.
  - Added profile-auto thresholds: `--profile-auto-min-score`, `--profile-auto-min-term-overlap`, `--profile-auto-min-embedding-score`.
  - Added added-topic profiles from proposal evidence atom text, proposal keywords, and event embedding centroid.
  - Profile assignment scores candidate atoms by content-term overlap plus optional embedding similarity.
  - Profile auto-assignment is written as secondary evolved labels with `strategy=evolved_profile_match`.
- `scripts/run_locomo_evolved_secondary_reuse_pipeline.py`
  - Exposes and forwards the profile-auto evolution arguments.
- Existing language-aware content terms in `src/leaf/normalize.py` are reused for profile construction.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  scripts/evolve_topic_view_from_shadow.py \
  scripts/run_locomo_evolved_secondary_reuse_pipeline.py \
  src/leaf/normalize.py \
  src/leaf/agentic_memory.py
```

Config and DB:

- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
  - Memory/evolution LLM: CodeAI `gpt-5.4-mini`
  - QA answer LLM: local OpenAI-compatible `Qwen3-235B` at `http://localhost:8001/v1`
- DB copied from SeedTopic retrofit DB:

```bash
rm -f data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3
cp data/agentic_smoke/locomo10_seedtopic_retrofit_evalv16_textonly_20260516.sqlite3 \
  data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3
```

Two-corpus evolution pipeline command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/run_locomo_evolved_secondary_reuse_pipeline.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3 \
  --samples conv-26 conv-30 \
  --source-prefix reports/agentic_memory/evolved_retrofit_20260516 \
  --prefix reports/agentic_memory/evolved_secondary_profileauto_20260516 \
  --output reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_pipeline_20260516.jsonl \
  --summary-output reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_pipeline_summary_20260516.json \
  --date-suffix 20260516 \
  --max-auto-reassignments 40 \
  --min-auto-reassign-keyword-matches 2 \
  --auto-reassignment-mode profile \
  --profile-auto-min-score 3.0 \
  --profile-auto-min-term-overlap 2 \
  --profile-auto-min-embedding-score 0.0 \
  --max-reassigned-atom-ratio 0.15 \
  --min-topic-path-hit-improvement 0.05 \
  2>&1 | tee reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_pipeline_20260516.log
```

Eight-corpus evolution pipeline command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/run_locomo_evolved_secondary_reuse_pipeline.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3 \
  --samples conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --source-prefix reports/agentic_memory/evolved_retrofit_20260516 \
  --prefix reports/agentic_memory/evolved_secondary_profileauto_20260516 \
  --output reports/agentic_memory/locomo8_evolvedtopic_secondary_profileauto_pipeline_20260516.jsonl \
  --summary-output reports/agentic_memory/locomo8_evolvedtopic_secondary_profileauto_pipeline_summary_20260516.json \
  --date-suffix 20260516 \
  --max-auto-reassignments 40 \
  --min-auto-reassign-keyword-matches 2 \
  --auto-reassignment-mode profile \
  --profile-auto-min-score 3.0 \
  --profile-auto-min-term-overlap 2 \
  --profile-auto-min-embedding-score 0.0 \
  --max-reassigned-atom-ratio 0.15 \
  --min-topic-path-hit-improvement 0.05 \
  2>&1 | tee reports/agentic_memory/locomo8_evolvedtopic_secondary_profileauto_pipeline_20260516.log
```

Evolution gate results:

- `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_pipeline_summary_20260516.json`: `count=2`, `promoted_count=2`
- `reports/agentic_memory/locomo8_evolvedtopic_secondary_profileauto_pipeline_summary_20260516.json`: `count=8`, `promoted_count=7`, `gate_failed=1`
- Combined: 9 / 10 promoted; `conv-49` failed the topic-path-hit improvement gate and kept the seed-topic active view.

Per-corpus shadow topic metrics:

| Corpus | Status | topic_path_hit seed -> evolved | mean_topic_recall seed -> evolved | secondary assignments |
| --- | --- | ---: | ---: | ---: |
| conv-26 | promoted | 0.5000 -> 0.5833 | 0.5417 -> 0.4583 | 44 |
| conv-30 | promoted | 0.2500 -> 0.8333 | 0.2917 -> 0.5764 | 46 |
| conv-41 | promoted | 0.0833 -> 0.4167 | 0.1250 -> 0.2917 | 45 |
| conv-42 | promoted | 0.1667 -> 0.5000 | 0.2083 -> 0.3889 | 45 |
| conv-43 | promoted | 0.5833 -> 0.7500 | 0.6250 -> 0.6528 | 29 |
| conv-44 | promoted | 0.2500 -> 0.5000 | 0.2917 -> 0.3889 | 47 |
| conv-47 | promoted | 0.4167 -> 0.7500 | 0.4583 -> 0.6250 | 43 |
| conv-48 | promoted | 0.1667 -> 0.4167 | 0.2500 -> 0.4306 | 47 |
| conv-49 | gate_failed | 0.2500 -> 0.2500 | 0.3333 -> 0.2361 | 45 |
| conv-50 | promoted | 0.4167 -> 0.6667 | 0.4583 -> 0.5625 | 45 |

Medium QA sanity on `conv-26` + `conv-30`:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_sample233_20260516.json \
  --sample-limit 2 \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  2>&1 | tee reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_sample233_20260516.log
```

Medium QA result:

- Output: `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_sample233_20260516.json`
- Completed: 233 / 233
- F1: 0.5834
- BLEU1: 0.5180
- By type:
  - `multi_hop`: count 43, F1 0.4200
  - `open_domain`: count 13, F1 0.3589
  - `single_hop`: count 114, F1 0.5441
  - `temporal`: count 63, F1 0.8124
- This beat the previous medium-sample EvolvedReplace+TextPolicy result by about +0.0013 F1, so a full run was justified.

Full LoCoMo QA command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  2>&1 | tee reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.log
```

The first full QA process stopped after writing a partial output:

- Partial output: `completed=false`, 960 / 1540 rows
- Partial summary F1: 0.5617
- Last partial corpus: `conv-47`

Resume command:

```bash
PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/eval_locomo_qa_only_parallel.py \
  --resume-existing \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.json \
  --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-style structured_context_topic_labeled --qa-workers 8 \
  --retrieval-mode topic_soft_selective \
  --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  2>&1 | tee -a reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.log
```

Resume loaded 960 rows and ran the remaining 580 rows. Final output:

- `reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.json`
- `completed=true`
- Questions: 1540 / 1540
- F1: 0.5596
- BLEU1: 0.5022
- Average search ms: 12423.78
- Average answer ms: 1840.65

Full QA comparison:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms | avg answer ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base | 1540 | 0.5593 | 0.5022 | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 12295.73 | 1443.19 |
| SeedTopic+TextPolicy | 1540 | 0.5626 | 0.5045 | 0.3680 | 0.3120 | 0.6273 | 0.6390 | 11964.66 | 1676.23 |
| EvolvedReplace+TextPolicy | 1540 | 0.5608 | 0.5030 | 0.3562 | 0.3160 | 0.6258 | 0.6435 | 13368.17 | 1453.16 |
| EvolvedSecondaryKeyword | 1540 | 0.5577 | 0.4999 | 0.3582 | 0.3113 | 0.6230 | 0.6354 | 13784.62 | 1336.59 |
| EvolvedSecondaryProfileAuto | 1540 | 0.5596 | 0.5022 | 0.3583 | 0.3209 | 0.6234 | 0.6405 | 12423.78 | 1840.65 |

Interpretation:

- The profile-auto evolution mechanism works as a topic-tree growth mechanism under self-QA shadow gating: 9 / 10 corpora were promoted, and topic-path-hit improved on all promoted corpora.
- The full QA result does not beat SeedTopic+TextPolicy. It is only +0.0003 F1 over LEAF-Base and -0.0030 F1 behind SeedTopic+TextPolicy.
- The profile-auto variant improves open-domain F1 relative to all listed runs (`0.3209`), and improves multi-hop relative to LEAF-Base, but loses enough on single-hop and temporal questions that the full aggregate does not improve.
- The medium sample was optimistic; full LoCoMo exposed that stronger topic-tree growth does not automatically translate to better answer quality with the current `keyword` router + `topic_soft_selective` retrieval path.
- The current latency numbers should not be interpreted as single-request serving latency. QA used `--qa-workers 8`; logs showed large cold/colliding search tails, including individual search calls above 60s. This looks like evaluator concurrency / DB-read / topic-soft expansion overhead. It needs separate latency ablation and caching/indexing work.

Decision:

- Keep the profile-auto evolution code path as an important memory-modeling primitive.
- Do not treat the current evolved profile-auto assignment plus `topic_soft_selective` retrieval as the final QA configuration yet.
- Main non-hardcoded anchors for the next round:
  - `LEAF-Base / Eval-v16-textonly`: F1 0.5593 / BLEU1 0.5022.
  - `CurrentEvolvedAnchor` (`EvoProfileAutoAssign + KeywordRouterTopK3 + TextPolicy`): F1 0.5596 / BLEU1 0.5022.
- Next work should focus on coupling the topic tree to retrieval more carefully:
  - topic-aware reranking instead of simply injecting extra topic evidence,
  - cache/profile precomputation for topic-soft candidate expansion,
  - online per-N-turn evolution that influences future atom extraction/topic assignment,
  - a latency ablation with lower `qa-workers` before judging serving cost.

## 2026-05-16 Evolved Topic Retrieval Follow-up

Goal:

- Test whether `EvolvedSecondaryProfileAuto` can beat the current keyword topic-soft anchor by using evolved topic profiles more directly.
- Keep retrieval-time logic non-benchmark-specific: no question-type oracle, no LoCoMo-specific filters.

Code changes:

- `src/leaf/normalize.py`
  - Fixed `language_aware_terms` so English Latin tokens are not added before stopword filtering. This removed contraction/noise terms such as `i've` from profile terms.
- `src/leaf/topic_profile.py`
  - Added `topic_profile_v1`.
  - Topic `top_terms` now come from atom content only; entities remain in `entity_signature` but are no longer mixed into topic terms.
  - Added per-view document-frequency filtering via `--max-document-frequency-ratio` to reduce cross-topic conversational background words.
- `src/leaf/agentic_memory.py` and `src/leaf/topic_soft.py`
  - Added topic node role/scope support.
  - Added experimental `--topic-router evolved_profile_first`, including evolved-vs-seed profile competition and keyword fallback.
- `scripts/eval_locomo.py` and `scripts/eval_locomo_qa_only_parallel.py`
  - Added CLI support for `--topic-router evolved_profile_first`.
- `scripts/build_topic_profiles.py`
  - Added `--max-document-frequency-ratio`.
- `tmp/run_profileauto_grid_20260516.sh`
  - Added a small tmux-friendly ablation launcher for keyword topic-soft top-k/secondary-policy variants.

Verification:

```bash
/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -m py_compile \
  src/leaf/normalize.py src/leaf/agentic_memory.py src/leaf/topic_soft.py src/leaf/topic_profile.py \
  scripts/build_topic_profiles.py scripts/eval_locomo_qa_only_parallel.py scripts/eval_locomo.py
```

Profile build:

```bash
cp data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3 \
  data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_profilev1_evalv16_textonly_20260516.sqlite3

PYTHONPATH=src /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin/python -u \
  scripts/build_topic_profiles.py \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_profilev1_evalv16_textonly_20260516.sqlite3 \
  --all-active \
  --max-document-frequency-ratio 0.5 \
  --output reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_profilev1_topic_profiles_20260516.json
```

Profile output:

- `reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_profilev1_topic_profiles_20260516.json`
- `result_count=10`
- `updated_topic_count_total=127`
- Example after profile-v1 cleanup:
  - `locomo_conv_26/art_expression_identity`: `pretty`, `painted`, `patterns`, `drawing`, `form`, `outlet`, `carries`, `stroke`
  - `locomo_conv_30/business_entrepreneurship`: `check`, `pic`, `creativity`, `successful`, `clothing`, `promotions`, `growing`, `ahead`

Commands:

Profile-router 233QA runs were launched in tmux:

```bash
tmux new-session -d -s leaf_profilefirst_sample_20260516 "cd ... && PYTHONPATH=src ... --topic-router evolved_profile_first ... --output reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_evolvedprofilefirst_textpolicy_sample233_20260516.json ..."
tmux new-session -d -s leaf_competitiveprofile_sample_20260516 "cd ... && PYTHONPATH=src ... --topic-router evolved_profile_first ... --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_profilev1_evalv16_textonly_20260516.sqlite3 --output reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_competitiveprofile_textpolicy_sample233_20260516.json ..."
tmux new-session -d -s leaf_evolvedprofileboost_sample_20260516 "cd ... && PYTHONPATH=src ... --topic-router evolved_profile_first ... --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_profilev1_evalv16_textonly_20260516.sqlite3 --output reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_evolvedprofileboost_textpolicy_sample233_20260516.json ..."
```

Keyword ablation grid:

```bash
tmux new-session -d -s leaf_profileauto_grid_20260516 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && bash tmp/run_profileauto_grid_20260516.sh 2>&1 | tee reports/agentic_memory/profileauto_grid_20260516.log"
```

Outputs:

- Profile router:
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_evolvedprofilefirst_textpolicy_sample233_20260516.json`
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_competitiveprofile_textpolicy_sample233_20260516.json`
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_evolvedprofileboost_textpolicy_sample233_20260516.json`
- Keyword grid:
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_keyword_topk1_textpolicy_sample233_20260516.json`
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_keyword_topk2_textpolicy_sample233_20260516.json`
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_keyword_secondarystrict_textpolicy_sample233_20260516.json`
  - `reports/agentic_memory/profileauto_grid_20260516.log`

Medium-sample comparison on `conv-26` + `conv-30`, 233 QA rows:

| Run | Router / policy | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg topic events | avg search ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CurrentEvolvedAnchor (`EvoProfileAutoAssign`) | KeywordRouterTopK3 + TextPolicy | 233 | 0.5834 | 0.5180 | 0.4200 | 0.3589 | 0.5441 | 0.8124 | 0.6652 | 2390.57 |
| EvolvedProfileFirst v0 | evolved profile replaces seed when any route exists | 233 | 0.5717 | 0.5052 | 0.4102 | 0.3552 | 0.5326 | 0.7975 | 0.6695 | 2558.47 |
| CompetitiveProfile v1 | evolved profile only if it beats seed profile; seed-profile fallback | 233 | 0.5790 | 0.5115 | 0.4176 | 0.3944 | 0.5298 | 0.8165 | 0.5365 | 2817.09 |
| EvolvedProfileBoost v1 | evolved profile only if it beats seed profile; keyword fallback | 233 | 0.5758 | 0.5083 | 0.4177 | 0.3589 | 0.5328 | 0.8062 | 0.6652 | 2582.99 |
| Keyword top-k=1 | keyword top-k=1 | 233 | 0.5706 | 0.5024 | 0.4177 | 0.3589 | 0.5319 | 0.7888 | 0.6652 | 2389.17 |
| Keyword top-k=2 | keyword top-k=2 | 233 | 0.5777 | 0.5102 | 0.4127 | 0.3552 | 0.5408 | 0.8029 | 0.6652 | 2110.07 |
| Keyword secondary-strict | keyword top-k=3 + strict secondary text policy | 233 | 0.5734 | 0.5074 | 0.3960 | 0.3589 | 0.5322 | 0.8134 | 0.6652 | 2102.54 |

Row-level deltas vs the 233QA anchor:

- `EvolvedProfileFirst v0`: average F1 delta `-0.0117`; better by >0.05 on 6 rows; worse by >0.05 on 17 rows.
- `CompetitiveProfile v1`: average F1 delta `-0.0044`; better by >0.05 on 6 rows; worse by >0.05 on 11 rows.
- `EvolvedProfileBoost v1`: average F1 delta `-0.0076`; better by >0.05 on 4 rows; worse by >0.05 on 10 rows.

Interpretation:

- Profile cleanup worked mechanically: topic profile terms are more thematic and less dominated by speakers/greetings.
- Directly routing with evolved profiles is still not better than the current keyword topic-soft anchor on the 233QA medium sample.
- `CompetitiveProfile v1` improved open-domain and temporal on this sample, but lost enough single-hop to underperform overall.
- Narrowing keyword top-k or enforcing strict secondary text policy also underperformed the anchor.
- The failure mode is not simply "too many topic events"; all variants add a similar average number of topic events after policy, but small changes in which extra span enters the answer prompt can perturb short answers.

Decision:

- Do not launch full LoCoMo for these retrieval-side variants.
- Keep the generic profile cleanup code; it is useful for future online evolution and topic analysis.
- Keep `evolved_profile_first` as an experimental router only, not as the recommended config.
- Non-hardcoded full-run anchors remain `LEAF-Base / Eval-v16-textonly` at F1 `0.5593` / BLEU1 `0.5022`, and `CurrentEvolvedAnchor` at F1 `0.5596` / BLEU1 `0.5022`.
- Next improvement should move upstream into memory modeling:
  - online per-N-turn topic growth that affects subsequent atom extraction/assignment,
  - higher-quality evolved topic proposals and assignment confidence,
  - topic-aware reranking/metadata in answer payload instead of simply injecting extra topic evidence.

## 2026-05-16 Environment Pin And EvoProfile Assignment Follow-up

Goal:

- Pin all further LEAF agentic-memory experiments to one conda environment.
- Test memory-modeling-side evolved assignment variants with clear names and without changing the retrieval router/policy.

Environment:

- Conda env: `tracenav_nlp`
- Python: `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/tracenav_nlp/bin/python`
- Wrapper: `scripts/with_tracenav_nlp.sh`
- Wrapper exports:
  - `PYTHONNOUSERSITE=1`
  - `NLTK_DATA=/vepfs-mlp2/c20250513/241404044/users/roytian/nltk_data`
  - `PYTHONPATH=/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory/src`
- Installed into `tracenav_nlp` with env-local pip:

```bash
PYTHONNOUSERSITE=1 \
  /vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/envs/tracenav_nlp/bin/python \
  -m pip install 'json-repair>=0.30.0' nltk
```

- Verified packages under `PYTHONNOUSERSITE=1`: `yaml`, `regex`, `jieba`, `sklearn`, `json_repair`, `numpy`, `openai`, `pydantic`, `nltk`, `torch`, `tqdm`, `spacy`, `yake`.
- `sqlite3` stdlib verified at version `3.51.0`.
- Caveat: NLTK package is installed, and `punkt`, `averaged_perceptron_tagger`, `maxent_ne_chunker` exist under the fixed `NLTK_DATA` path; `corpora/words` download failed once due GitHub SSL EOF. Current LoCoMo QA/evolution path does not depend on this NLTK NER fallback.

Code changes:

- `scripts/with_tracenav_nlp.sh`
  - New wrapper to run all future scripts inside the pinned `tracenav_nlp` environment.
- `scripts/evolve_topic_view_from_shadow.py`
  - Fixed evolved topic profile construction so `evidence_atom_ids` are preserved in `added_topics`; before this, profile-auto matching was mostly proposal-keyword-driven and did not actually use evidence atom content in added topic profiles.
  - Added `--profile-auto-text-mode {content_entities,content_only}`.
  - Added ambiguity gates: `--profile-auto-min-score-margin`, `--profile-auto-min-score-ratio`.
  - Added compact-profile option: `--profile-auto-max-profile-terms`.
- `scripts/run_locomo_evolved_secondary_reuse_pipeline.py`
  - Forwards the new profile-auto assignment flags.

Verification:

```bash
scripts/with_tracenav_nlp.sh python -m py_compile \
  scripts/evolve_topic_view_from_shadow.py \
  scripts/run_locomo_evolved_secondary_reuse_pipeline.py \
  scripts/eval_locomo_qa_only_parallel.py \
  scripts/eval_locomo.py \
  scripts/eval_memory_search.py \
  scripts/evaluate_memory_view_gate.py
```

Medium QA setup:

- Dataset: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Medium sample: first 2 corpora, 233 QA rows (`conv-26`, `conv-30`)
- Answer model config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Retrieval held fixed across anchor/v1/v2/v3:
  - `--retrieval-mode topic_soft_selective`
  - `--topic-router keyword`
  - `--topic-route-top-k 3`
  - `--topic-soft-event-limit 1`
  - `--topic-soft-per-topic-atom-limit 16`
  - `--topic-soft-min-content-overlap 1`
  - `--topic-soft-allow-fallback-topic`
  - `--topic-soft-fallback baseline_on_unknown`
  - `--topic-soft-policy text_temporal_suppressed_v0`
  - `--topic-soft-policy-min-selected-overlap 1`
  - `--topic-soft-policy-max-candidate-atoms 999`
  - `--temporal-postprocess range`
  - `--short-answer-postprocess precise`
  - `--answer-style structured_context_topic_labeled`

Variant definitions:

- `CurrentEvolvedAnchor`: existing `EvoProfileAutoAssign + KeywordRouterTopK3 + TextPolicy`.
- `EvoProfileCleanAssign-v1`: fixes evidence atom carry-through into evolved topic profiles; uses `content_only`, margin `0.5`, ratio `1.15`, auto budget `40`.
- `EvoProfileCompactAssign-v2`: same as v1, but limits each evolved topic profile to top `16` terms.
- `EvoProfileSelectiveAssign-v3`: same as v1, but reduces auto assignment budget from `40` to `20`.

Pipeline outputs:

- v1:
  - DB: `data/agentic_smoke/locomo10_evo_profile_cleanassign_v1_evalv16_textonly_20260516.sqlite3`
  - Pipeline summary: `reports/agentic_memory/locomo2_evo_profile_cleanassign_v1_pipeline_summary_20260516.json`
  - QA JSON: `reports/agentic_memory/locomo2_evo_profile_cleanassign_v1_keywordtopk3_textpolicy_sample233_20260516.json`
- v2:
  - DB: `data/agentic_smoke/locomo10_evo_profile_compactassign_v2_evalv16_textonly_20260516.sqlite3`
  - Pipeline summary: `reports/agentic_memory/locomo2_evo_profile_compactassign_v2_pipeline_summary_20260516.json`
  - QA JSON: `reports/agentic_memory/locomo2_evo_profile_compactassign_v2_keywordtopk3_textpolicy_sample233_20260516.json`
- v3:
  - DB: `data/agentic_smoke/locomo10_evo_profile_selectiveassign_v3_evalv16_textonly_20260516.sqlite3`
  - Pipeline summary: `reports/agentic_memory/locomo2_evo_profile_selectiveassign_v3_pipeline_summary_20260516.json`
  - QA JSON: `reports/agentic_memory/locomo2_evo_profile_selectiveassign_v3_keywordtopk3_textpolicy_sample233_20260516.json`

Pipeline status:

| Variant | Corpora | Promoted | conv-26 secondary assignments | conv-30 secondary assignments |
| --- | ---: | ---: | ---: | ---: |
| EvoProfileCleanAssign-v1 | 2 | 2 | 44 | 46 |
| EvoProfileCompactAssign-v2 | 2 | 2 | 44 | 46 |
| EvoProfileSelectiveAssign-v3 | 2 | 2 | 24 | 26 |

Medium-sample QA comparison:

| Run | Q | F1 | BLEU1 | multi_hop F1 | open_domain F1 | single_hop F1 | temporal F1 | avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CurrentEvolvedAnchor | 233 | 0.5834 | 0.5180 | 0.4200 | 0.3589 | 0.5441 | 0.8124 | 2390.57 |
| EvoProfileCleanAssign-v1 | 233 | 0.5818 | 0.5176 | 0.4235 | 0.3589 | 0.5363 | 0.8181 | 2180.14 |
| EvoProfileCompactAssign-v2 | 233 | 0.5818 | 0.5145 | 0.4143 | 0.3589 | 0.5425 | 0.8134 | 2160.58 |
| EvoProfileSelectiveAssign-v3 | 233 | 0.5772 | 0.5098 | 0.4163 | 0.3919 | 0.5427 | 0.7878 | 2144.49 |

Interpretation:

- `EvoProfileCleanAssign-v1` confirmed the evidence-profile bug fix is meaningful mechanically, and it improves medium-sample multi-hop and temporal F1, but the single-hop loss prevents an aggregate gain.
- `EvoProfileCompactAssign-v2` reduces raw candidate count, but does not improve aggregate QA; compacting profile terms hurts multi-hop.
- `EvoProfileSelectiveAssign-v3` improves open-domain on the medium sample, but reducing secondary assignment budget hurts temporal enough to make it clearly worse overall.
- None of v1/v2/v3 beats `CurrentEvolvedAnchor` on the 233QA medium sample, so none should be launched as full LoCoMo yet.

Decision:

- Keep the evidence-atom carry-through fix in `evolve_topic_view_from_shadow.py`; previous profile-auto behavior was underusing proposal evidence.
- Do not promote v1/v2/v3 to full evaluation.
- Next promising direction is not more retrieval-time filtering. It should be online/incremental topic growth and assignment during ingest, so evolved topics affect subsequent atom extraction/topic assignment rather than only adding secondary labels after a batch retrofit.

## 2026-05-16 EvoOnlineGrowth-v1

Goal:

- Move from batch retrofit evolution to realistic incremental topic growth.
- Make topic tree growth happen during online ingest so newly promoted topics can affect later atom extraction hints and later atom-to-topic assignment.

Code changes:

- `src/leaf/config.py`
  - Added `ingest.online_evolution_enabled`.
  - Added online evolution thresholds: `online_evolution_turns_threshold`, `online_evolution_atoms_threshold`, `online_evolution_min_cluster_atoms`, `online_evolution_max_new_topics`, `online_evolution_window_atom_limit`.
  - Added `online_evolution_trigger_policy` with `any`, `all`, and `turns`.
- `src/leaf/service.py`
  - `append_turns(...)` now calls online evolution with config-backed thresholds.
  - Manual calls to `maybe_evolve_agentic_memory_after_ingest(...)` still preserve explicit caller thresholds unless `use_config=True`.
- `src/leaf/agentic_memory.py`
  - Added online-growth quality cleanup for growth terms: speaker/generic entity suppression, `Context overlap` cleanup, generic conversational/support token filtering.
  - Added same-chunk redundant-topic suppression by evidence overlap.
  - Constrained online-grown topics to attach under root/seed-level parents, preventing deep evolved-topic chains in short conversations.
- `scripts/run_locomo_online_growth.py`
  - New incremental LoCoMo runner that chunks turns, appends them in online mode, records active view changes, promoted topics, final tree shape, and chunk-level evolution status.

Environment:

- All runs used `scripts/with_tracenav_nlp.sh`.

Verification:

```bash
scripts/with_tracenav_nlp.sh python -m py_compile \
  src/leaf/config.py src/leaf/service.py src/leaf/agentic_memory.py scripts/run_locomo_online_growth.py
```

Smoke 1: first 120 turns of `conv-26`, 20-turn chunks.

Command shape:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/run_locomo_online_growth.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo_online_growth_smoke_struct_v4_20260516.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo1_online_growth_smoke_struct_v4_20260516.json \
  --sample-limit 1 --max-turns-per-sample 120 --chunk-turns 20 \
  --turns-threshold 40 --atoms-threshold 24 --min-cluster-atoms 2 \
  --max-new-topics 3 --window-atom-limit 80 --print-tree
```

Output:

- JSON: `reports/agentic_memory/locomo1_online_growth_smoke_struct_v4_20260516.json`
- DB: `data/agentic_smoke/locomo_online_growth_smoke_struct_v4_20260516.sqlite3`
- Result: final active tree for `conv-26` had 26 nodes: root 1, seed level-1 nodes 10, evolved level-2 nodes 15.
- Interpretation: online growth is functional and shallow after structure constraints. Topic quality improved versus raw growth, but some generic labels remain.

Smoke 2: full `conv-26` + `conv-30`, 40-turn chunks, trigger-policy `all`.

Command shape:

```bash
tmux new-session -d -s leaf_online_growth_locomo2_all_20260516 \
  "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   scripts/with_tracenav_nlp.sh python -u scripts/run_locomo_online_growth.py \
   --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
   --db data/agentic_smoke/locomo2_online_growth_struct_v5_all_20260516.sqlite3 \
   --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
   --output reports/agentic_memory/locomo2_online_growth_struct_v5_all_20260516.json \
   --sample-limit 2 --chunk-turns 40 --turns-threshold 80 --atoms-threshold 48 \
   --trigger-policy all --min-cluster-atoms 3 --max-new-topics 2 \
   --window-atom-limit 120 --print-tree \
   2>&1 | tee reports/agentic_memory/locomo2_online_growth_struct_v5_all_20260516.log"
```

Outputs:

- JSON: `reports/agentic_memory/locomo2_online_growth_struct_v5_all_20260516.json`
- Log: `reports/agentic_memory/locomo2_online_growth_struct_v5_all_20260516.log`
- DB: `data/agentic_smoke/locomo2_online_growth_struct_v5_all_20260516.sqlite3`

Counts:

| Corpus | Turns | Chunks | Promoted chunks | Skipped chunks | Final nodes | Final evolved nodes | Final levels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conv-26 | 419 | 11 | 5 | 6 | 21 | 10 | level0=1, level1=10, level2=10 |
| conv-30 | 369 | 10 | 4 | 6 | 19 | 8 | level0=1, level1=10, level2=8 |

Added topic slugs:

- `conv-26`: `kids`, `fun`, `community`, `difference`, `art`, `acceptance`, `courage`, `adoption`, `helps`, `emotions`.
- `conv-30`: `dance`, `gina`, `business`, `biz`, `studio`, `confidence`, `focused`, `determined`.

Interpretation:

- `trigger-policy all` is much better than `any` for online growth frequency. With `any`, `conv-26` grew 20 evolved topics; with `all`, it grew 10.
- Structure constraints work: final trees remain shallow, with all evolved topics at level 2.
- Topic quality is improved enough for an online-growth medium QA sanity run, but still has label noise and duplicate/alias topics (`business`/`biz`, `dance`/`gina`).
- Next candidate improvement is topic proposal quality, not retrieval filtering: merge alias/near-duplicate candidate topics before promotion, and use topic labels that come from compact phrase/entity concepts rather than single surface terms.

Decision:

- Use `EvoOnlineGrowth-v1` with `trigger-policy all` as the first online-growth QA candidate.
- Run 233QA medium on `data/agentic_smoke/locomo2_online_growth_struct_v5_all_20260516.sqlite3` with the same `KeywordRouterTopK3 + TextPolicy` retrieval settings as `CurrentEvolvedAnchor`.

## 2026-05-16 EvoOnlineGrowth v1.1-v1.3 and TaxoAdapt notes

Source paper inspected:

- arXiv PDF: `https://arxiv.org/pdf/2506.10737`
- ACL Anthology PDF mirror: `https://aclanthology.org/2025.acl-long.1442.pdf`

TaxoAdapt insight to port:

- Do not force every new topic into one fixed depth.
- Treat taxonomy evolution as a choice between:
  - depth expansion: a dense leaf should split into more specific children;
  - width expansion: a parent has many unmapped or weakly covered items, so it needs new siblings under that parent.
- New nodes should be promoted with sibling/path context, same-granularity checks, and redundancy control.
- For LEAF, this maps to pending atom density, unmapped/low-confidence atom density, and child coverage ratio under the current active topic tree.

Implementation changes:

- `src/leaf/extract.py`
  - Atom metadata now carries online chunk metadata such as `source_speakers`, `merged_turn_count`, and `merged_turn_indexes`.
  - This removes the need to suppress benchmark-specific speaker names in topic growth.
- `src/leaf/agentic_memory.py`
  - Removed LoCoMo-specific person-name filters from `TOPIC_GROWTH_GENERIC_ENTITY_TOKENS`.
  - Added online secondary profile assignment support for newly promoted topics.
  - Changed online topic promotion from primary reassignment to `primary_preserved_secondary_evolved`.
  - Added bounded depth expansion with `max_depth`; evolved topics can attach below evolved topics instead of being forced to seed-level parents.
- `src/leaf/config.py`
  - Added online evolution controls for `online_evolution_max_depth`.
  - Added online secondary assignment controls.
- `src/leaf/service.py`
  - Passes config-backed online secondary assignment and max-depth settings into `grow_topic_tree_from_recent_atoms`.
- `scripts/run_locomo_online_growth.py`
  - Added CLI flags for `--max-depth` and secondary assignment settings.

Verification:

```bash
scripts/with_tracenav_nlp.sh python -m py_compile \
  src/leaf/config.py src/leaf/service.py src/leaf/agentic_memory.py \
  src/leaf/extract.py scripts/run_locomo_online_growth.py \
  scripts/eval_locomo_qa_only_parallel.py
```

Online growth runs:

| Variant | DB | Result JSON | Notes |
| --- | --- | --- | --- |
| EvoOnlineGrowth-v1.1-SpeakerAware | `data/agentic_smoke/locomo2_online_growth_v1_1_speakeraware_20260516.sqlite3` | `reports/agentic_memory/locomo2_online_growth_v1_1_speakeraware_20260516.json` | Speaker-aware metadata, no benchmark-specific name suppression, shallow tree. |
| EvoOnlineGrowth-v1.2-DepthSecondaryProfile | `data/agentic_smoke/locomo2_online_growth_v1_2_depth_secondaryprofile_20260516.sqlite3` | `reports/agentic_memory/locomo2_online_growth_v1_2_depth_secondaryprofile_20260516.json` | Bounded depth expansion plus profile-based secondary assignment. |
| EvoOnlineGrowth-v1.3-DepthEvidenceOnly | `data/agentic_smoke/locomo2_online_growth_v1_3_depth_evidenceonly_20260516.sqlite3` | `reports/agentic_memory/locomo2_online_growth_v1_3_depth_evidenceonly_20260516.json` | In progress at log time; same depth expansion but `--secondary-max-assignments 0`. |

v1.1 final tree shape:

| Corpus | Nodes | Levels | Level>=2 slugs |
| --- | ---: | --- | --- |
| conv-26 | 21 | level0=1, level1=10, level2=10 | `adoption`, `community`, `acceptance`, `art`, `bring`, `courage`, `emotions`, `helps`, `kids`, `difference` |
| conv-30 | 19 | level0=1, level1=10, level2=8 | `business`, `studio`, `dreams`, `creativity`, `dance`, `fashion`, `focused`, `goals` |

v1.2 final tree shape:

| Corpus | Nodes | Levels | Level>=2 slugs |
| --- | ---: | --- | --- |
| conv-26 | 21 | level0=1, level1=10, level2=7, level3=3 | `community@L2`, `acceptance@L2`, `art@L2`, `bring@L2`, `life@L2`, `kids@L2`, `adoption@L3`, `ago@L3`, `courage@L3`, `difference@L2` |
| conv-30 | 19 | level0=1, level1=10, level2=5, level3=3 | `business@L2`, `store@L3`, `studio@L3`, `dance@L2`, `dreams@L3`, `goals@L2`, `focused@L2`, `fashion@L2` |

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Avg search ms | Avg answer ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CurrentEvolvedAnchor | 233 | 0.5834 | 0.5180 | 0.4200 | 0.3589 | 0.5441 | 0.8124 | 2390.57 | 2196.28 |
| EvoProfileCleanAssign-v1 | 233 | 0.5818 | 0.5176 | 0.4235 | 0.3589 | 0.5363 | 0.8181 | 2180.14 | 2316.10 |
| EvoOnlineGrowth-v1.1-SpeakerAware | 233 | 0.5678 | 0.5030 | 0.3911 | 0.3622 | 0.5305 | 0.7981 | 2336.41 | 2308.90 |
| EvoOnlineGrowth-v1.2-DepthSecondaryProfile | 233 | 0.5585 | 0.4941 | 0.3837 | 0.3525 | 0.5238 | 0.7830 | 1955.89 | 3306.70 |
| EvoOnlineGrowth-v1.3-DepthEvidenceOnly | 233 | 0.5710 | 0.5067 | 0.4155 | 0.3525 | 0.5261 | 0.8036 | 2652.19 | 2045.87 |

v1.2 interpretation:

- Bounded depth expansion worked mechanically: both corpora grew level-3 nodes.
- Profile secondary assignment was too broad: average raw topic-soft candidate atoms rose to 62.72, versus 49.39 for v1.1 and 52.88 for `CurrentEvolvedAnchor`.
- Answer latency increased substantially, indicating noisier or wider contexts.
- Do not promote v1.2 as the QA candidate despite better tree shape.

v1.3 evidence-only result:

- DB: `data/agentic_smoke/locomo2_online_growth_v1_3_depth_evidenceonly_20260516.sqlite3`
- Growth JSON: `reports/agentic_memory/locomo2_online_growth_v1_3_depth_evidenceonly_20260516.json`
- QA JSON: `reports/agentic_memory/locomo2_evo_online_growth_v1_3_depth_evidenceonly_keywordtopk3_textpolicy_sample233_20260516.json`
- QA log: `reports/agentic_memory/locomo2_evo_online_growth_v1_3_depth_evidenceonly_keywordtopk3_textpolicy_sample233_20260516.log`
- Final shape:
  - `conv-26`: 21 nodes; levels `level0=1, level1=10, level2=5, level3=5`; level>=2 slugs `art@L2`, `kids@L2`, `courage@L3`, `community@L2`, `freedom@L3`, `difference@L2`, `fun@L2`, `acceptance@L3`, `emotions@L3`, `expression@L3`.
  - `conv-30`: 19 nodes; levels `level0=1, level1=10, level2=2, level3=6`; level>=2 slugs `business@L2`, `dancers@L3`, `store@L3`, `dance@L2`, `dreams@L3`, `fashion@L3`, `focused@L3`, `helps@L3`.
- `--secondary-max-assignments 0`; only proposal evidence atoms were written as secondary labels.
- v1.3 improved over v1.2: F1 0.5710 vs 0.5585, BLEU1 0.5067 vs 0.4941, and answer latency dropped to 2045.87 ms.
- v1.3 still trails `CurrentEvolvedAnchor` overall, but multi-hop recovered to 0.4155, close to anchor 0.4200.

Decision:

- Keep bounded depth expansion as a useful mechanism.
- Keep evidence-only secondary labels as the safer default for online growth.
- Do not enable broad profile-secondary expansion in online growth yet.
- The next implementation should be `EvoOnlineGrowth-v2-NodeLocalGrowth`, inspired by TaxoAdapt:
  - localize candidate generation to dense nodes rather than global term clusters;
  - choose depth or width expansion per node;
  - use sibling/path granularity checks before promotion;
  - add a tree-quality report with path granularity, sibling coherence, coverage, and redundancy proxies.

Next design:

- Keep bounded depth expansion.
- Replace global term clustering with TaxoAdapt-style node-local growth:
  - compute pending atom density per active topic node;
  - compute child coverage / unmapped density for non-leaf nodes;
  - choose depth expansion for dense leaves and width expansion for high unmapped-density parents;
  - require same-granularity checks against siblings and suppress low-specificity temporal/support terms before promotion.
- Use v1.3 evidence-only run to isolate whether depth expansion itself helps once profile-secondary noise is removed.

## 2026-05-17 EvoOnlineGrowth-v2 NodeLocalGrowth

Implementation changes:

- `src/leaf/agentic_memory.py`
  - Added `growth_strategy="node_local"` for online topic growth.
  - Node-local growth groups pending atoms under their current topic node, then proposes local width/depth expansions instead of global term clusters.
  - Added `growth_action`, `growth_score`, `node_density`, and `unmapped_density` metadata for proposed topics.
  - Preserves primary atom assignments and writes evolved evidence atoms as `atom_secondary`.
- `src/leaf/config.py`
  - Added `online_evolution_growth_strategy`.
- `src/leaf/service.py`
  - Passes configured `growth_strategy` into online evolution.
  - Added default `growth_strategy="global_terms"` for direct/manual calls.
- `scripts/run_locomo_online_growth.py`
  - Added `--growth-strategy {global_terms,node_local}`.

NodeLocal-v2 command:

```bash
tmux new-session -d -s leaf_online_growth_v2_nodelocal_locomo2_20260517 '
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
scripts/with_tracenav_nlp.sh python -u scripts/run_locomo_online_growth.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_online_growth_v2_nodelocal_evidenceonly_20260517.json \
  --sample-limit 2 --chunk-turns 40 --turns-threshold 80 --atoms-threshold 48 \
  --trigger-policy all --growth-strategy node_local \
  --min-cluster-atoms 3 --max-new-topics 2 --max-depth 4 \
  --window-atom-limit 120 --secondary-max-assignments 0 --print-tree \
  2>&1 | tee reports/agentic_memory/locomo2_online_growth_v2_nodelocal_evidenceonly_20260517.log
'
```

NodeLocal-v2 final tree shape:

| Corpus | Nodes | Levels | Level>=2 slugs |
| --- | ---: | --- | --- |
| conv-26 | 21 | level0=1, level1=10, level2=6, level3=4 | `lgbtq@L2`, `blast@L3`, `community@L3`, `acceptance@L2`, `colors@L3`, `expression@L3`, `flowers@L2`, `tips@L2`, `kids@L2`, `dreams@L2` |
| conv-30 | 19 | level0=1, level1=10, level2=2, level3=3, level4=3 | `dance@L2`, `business@L3`, `advice@L4`, `helps@L4`, `stay@L4`, `customers@L3`, `dance_studio@L2`, `dream@L3` |

NodeLocal-v2 QA command used the same Qwen answer model and `KeywordRouterTopK3 + TextPolicy` settings as prior medium runs, reading:

- DB: `data/agentic_smoke/locomo2_online_growth_v2_nodelocal_evidenceonly_20260517.sqlite3`
- QA output: `reports/agentic_memory/locomo2_evo_online_growth_v2_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`

NodeLocal-v2 result:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Raw candidates | Candidate atoms | Avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OnlineGrowth-v1.3-DepthEvidenceOnly | 233 | 0.5710 | 0.5067 | 0.4155 | 0.3525 | 0.5261 | 0.8036 | 51.90 | 19.57 | 2652.19 |
| OnlineGrowth-v2-NodeLocalEvidenceOnly | 233 | 0.5601 | 0.4924 | 0.3793 | 0.3643 | 0.5372 | 0.7654 | 66.76 | 22.94 | 2445.53 |

Interpretation:

- Node-local growth produced a more realistic growing tree, including deeper L3/L4 nodes.
- QA dropped because evolved topic keywords were too broad and noisy, e.g. routeable topics such as `acceptance`, `dance_studio`, and `business` pulled large candidate pools into search.
- Main failure mode was topic-modeling quality, not answer model or question-type cheating.
- The fix should be in the memory modeling layer: cleaner topic labels and narrower route keywords, not benchmark-specific retrieval filters.

## 2026-05-17 EvoOnlineGrowth-v2.1 CleanRouteNodeLocal

Implementation changes:

- `src/leaf/agentic_memory.py`
  - Added `route_keywords` metadata for evolved topics.
  - `route_query_to_topics`, `assign_atom_to_topic_view`, and active topic hints now prefer `route_keywords` when present.
  - Evolved route keywords are intentionally narrow: default is the primary topic label only.
  - Growth keyword generation now requires repeated evidence terms and suppresses low-specificity/noisy terms through the shared NLP stopword path.
  - This keeps evolved topics useful as secondary evidence labels while reducing accidental keyword-route activation.

Smoke checks:

- `reports/agentic_memory/locomo1_online_growth_v2_1_nodelocal_cleanroute_smoke80_20260517.json`
- `reports/agentic_memory/locomo1_online_growth_v2_1_cleanroute_smoke160_20260517.json`
- Smoke confirmed that evolved topics still promote, but route keywords are narrow, e.g. `lgbtq -> ["lgbtq"]`, `kids -> ["kids"]`, `dreams -> ["dreams"]`.

Full online growth command:

```bash
tmux new-session -d -s leaf_online_growth_v2_1_cleanroute_locomo2_20260517 '
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
scripts/with_tracenav_nlp.sh python -u scripts/run_locomo_online_growth.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.json \
  --sample-limit 2 --chunk-turns 40 --turns-threshold 80 --atoms-threshold 48 \
  --trigger-policy all --growth-strategy node_local \
  --min-cluster-atoms 3 --max-new-topics 2 --max-depth 4 \
  --window-atom-limit 120 --secondary-max-assignments 0 --print-tree \
  2>&1 | tee reports/agentic_memory/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.log
'
```

CleanRoute-v2.1 final tree shape:

| Corpus | Nodes | Levels | Level>=2 route slugs |
| --- | ---: | --- | --- |
| conv-26 | 21 | level0=1, level1=10, level2=10 | `acceptance`, `adoption`, `apologize`, `difference`, `dreams`, `emotions`, `expression`, `face`, `kids`, `lgbtq` |
| conv-30 | 19 | level0=1, level1=10, level2=5, level3=3 | `achievements`, `dance`, `dream`, `festival`, `lost`, `business`, `dreams`, `studio` |

CleanRoute-v2.1 QA outputs:

- TopK3: `reports/agentic_memory/locomo2_evo_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`
- TopK2 ablation: `reports/agentic_memory/locomo2_evo_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_keywordtopk2_textpolicy_sample233_20260517.json`

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Raw candidates | Candidate atoms | Avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 0.00 | 0.00 | 2075.88 |
| CurrentEvolvedAnchor subset conv-26+30 | 233 | 0.5772 | 0.5106 | 0.4191 | 0.3552 | 0.5409 | 0.7967 | 52.88 | 17.06 | 2331.87 |
| OnlineGrowth-v2-NodeLocalEvidenceOnly TopK3 | 233 | 0.5601 | 0.4924 | 0.3793 | 0.3643 | 0.5372 | 0.7654 | 66.76 | 22.94 | 2445.53 |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 52.90 | 17.71 | 2300.35 |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK2 | 233 | 0.5754 | 0.5090 | 0.4158 | 0.3487 | 0.5410 | 0.7936 | 51.63 | 17.37 | 2625.91 |

Interpretation:

- CleanRoute-v2.1 fixed the v2 regression: raw candidates dropped from 66.76 to 52.90 and F1 rose from 0.5601 to 0.5790.
- On the same conv-26+30 subset, CleanRoute-v2.1 is slightly above LEAF-Base and the current evolved anchor, but the gain is small: +0.0016 F1 over LEAF-Base subset.
- TopK2 is worse than TopK3, so the best current online-growth setting is `KeywordRouterTopK3 + CleanRouteNodeLocal`.
- This is not yet a strong submission-quality improvement. The next optimization should be an evolved-topic promotion gate:
  - keep abstract/noisy candidates in the tree metadata as observations;
  - only expose a candidate to `route_keywords` when it has evidence density plus lexical specificity/coherence;
  - optionally allow deeper topics to exist but keep them dormant for retrieval until they pass the gate.

Current best online-growth candidate:

- Name: `OnlineGrowth-v2.1-CleanRouteNodeLocal-TopK3`
- Growth DB: `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Growth JSON: `reports/agentic_memory/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.json`
- QA JSON: `reports/agentic_memory/locomo2_evo_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`
- Status: best online incremental result so far, but only a small gain over same-subset LEAF-Base.

## 2026-05-17 EvoOnlineGrowth-v2.2 PrimaryGate

Purpose:

- Test whether evolved topics should remain secondary/auxiliary memory views instead of becoming primary atom assignment targets during later ingest.
- This is a memory-modeling gate, not a retrieval-time hard filter:
  - evolved topics are still written into the tree;
  - evolved topics can still expose narrow route keywords;
  - evolved topics can still receive evidence atoms as `atom_secondary`;
  - evolved topics do not steal later atoms as primary `atom` assignments when the gate is disabled.

Implementation changes:

- `src/leaf/config.py`
  - Added `online_evolution_evolved_primary_assignment_enabled`.
- `src/leaf/service.py`
  - Passes `evolved_primary_assignment_enabled` to online growth and records it in evolution triggers.
- `src/leaf/agentic_memory.py`
  - Added topic metadata `primary_assignment_exposure`.
  - `assign_atom_to_topic_view` skips topics whose `primary_assignment_exposure` is `inactive`.
  - Kept `route_exposure` support so route keywords can be disabled independently later.
- `scripts/run_locomo_online_growth.py`
  - Added `--disable-evolved-primary-assignment`.

Smoke check:

- Command: `scripts/run_locomo_online_growth.py ... --max-turns-per-sample 160 --disable-evolved-primary-assignment`
- Output: `reports/agentic_memory/locomo1_online_growth_v2_2_primarygate_smoke160_20260517.json`
- Result: evolved topics had `primary_assignment_exposure=inactive` and only `atom_secondary` assignments.

Full online growth command:

```bash
tmux new-session -d -s leaf_online_growth_v2_2_primarygate_locomo2_20260517 '
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
scripts/with_tracenav_nlp.sh python -u scripts/run_locomo_online_growth.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_2_primarygate_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_online_growth_v2_2_primarygate_nodelocal_evidenceonly_20260517.json \
  --sample-limit 2 --chunk-turns 40 --turns-threshold 80 --atoms-threshold 48 \
  --trigger-policy all --growth-strategy node_local \
  --min-cluster-atoms 3 --max-new-topics 2 --max-depth 4 \
  --window-atom-limit 120 --secondary-max-assignments 0 \
  --disable-evolved-primary-assignment --print-tree \
  2>&1 | tee reports/agentic_memory/locomo2_online_growth_v2_2_primarygate_nodelocal_evidenceonly_20260517.log
'
```

Final tree and assignment shape:

| Corpus | Nodes | Levels | Evolved primary assignment counts |
| --- | ---: | --- | --- |
| conv-26 | 21 | level0=1, level1=10, level2=10 | all evolved topics `atom=0`, only `atom_secondary` |
| conv-30 | 19 | level0=1, level1=10, level2=8 | all evolved topics `atom=0`, only `atom_secondary` |

QA output:

- `reports/agentic_memory/locomo2_evo_online_growth_v2_2_primarygate_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Raw candidates | Candidate atoms | Avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 0.00 | 0.00 | 2075.88 |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 52.90 | 17.71 | 2300.35 |
| OnlineGrowth-v2.2-PrimaryGate TopK3 | 233 | 0.5728 | 0.5090 | 0.3850 | 0.3487 | 0.5334 | 0.8186 | 47.51 | 15.28 | 2756.77 |

Interpretation:

- PrimaryGate-v2.2 reduced candidate volume further and improved temporal F1 to 0.8186, the best temporal score among online variants so far.
- However, it hurt multi-hop substantially: 0.3850 versus 0.4146 for v2.1.
- Fully disabling evolved primary assignment is too conservative; evolved topics need to become primary assignment targets when they are specific/coherent enough.
- Current best remains `OnlineGrowth-v2.1-CleanRouteNodeLocal-TopK3`.

Next design:

- Replace the binary `--disable-evolved-primary-assignment` with a quality-based gate:
  - enable primary assignment only for evolved topics with sufficient evidence density, lexical specificity, and sibling/parent distinction;
  - keep abstract or low-specificity evolved topics route-only/secondary-only;
  - preserve v2.2's temporal benefit without sacrificing v2.1's multi-hop behavior.

## 2026-05-17 OnlineGrowth v2.3-v2.5 Quality Gate / Entity-First Growth

Goal:

- Improve online evolved topic growth without benchmark-specific retrieval filters.
- Keep v2.1 fair ingest parameters fixed for comparable LoCoMo2 medium runs:
  `--chunk-turns 40 --turns-threshold 80 --atoms-threshold 48 --trigger-policy all --min-cluster-atoms 3 --max-new-topics 2 --window-atom-limit 120`.

Implementation changes:

- `src/leaf/config.py`, `src/leaf/service.py`, `scripts/run_locomo_online_growth.py`
  - Added `online_evolution_evolved_primary_assignment_mode` / `--evolved-primary-assignment-mode`.
  - Supported modes: `all`, `none`, `quality_v0`, `quality_v1`.
- `src/leaf/agentic_memory.py`
  - Added per-topic `primary_assignment_gate` metadata.
  - `quality_v0`: evidence-density gate; too loose under fair parameters and became all-active.
  - `quality_v1`: entity-supported primary gate with corpus-level dominant-entity suppression.
  - Added entity-first topic-growth term ordering and automatic speaker-token suppression for v2.5 smoke.
- `src/leaf/clients.py`
  - Made provider usage parsing tolerant of non-dict responses, falling back to local token estimates.
- `src/leaf/grounding.py`, `scripts/eval_locomo.py`, `scripts/eval_locomo_qa_only_parallel.py`
  - Added experimental `relative_prefer` temporal postprocess mode, but offline simulation was not good enough to use as an anchor.

Artifacts:

- v2.3 quality-v0 fair online growth:
  - DB: `data/agentic_smoke/locomo2_online_growth_v2_3_qualitygate_fair80a48_retry1_nodelocal_evidenceonly_20260517.sqlite3`
  - JSON: `reports/agentic_memory/locomo2_online_growth_v2_3_qualitygate_fair80a48_retry1_nodelocal_evidenceonly_20260517.json`
  - Result: 18/18 evolved topics were `primary_assignment_exposure=active`; this collapsed to v2.1 behavior.
- v2.4 quality-v1 fair online growth:
  - DB: `data/agentic_smoke/locomo2_online_growth_v2_4_qualitygatev1_fair80a48_nodelocal_evidenceonly_20260517.sqlite3`
  - JSON: `reports/agentic_memory/locomo2_online_growth_v2_4_qualitygatev1_fair80a48_nodelocal_evidenceonly_20260517.json`
  - QA: `reports/agentic_memory/locomo2_evo_online_growth_v2_4_qualitygatev1_fair80a48_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`
- v2.5 entity-first smoke:
  - DB: `data/agentic_smoke/locomo1_online_growth_v2_5_entityfirst_smoke160_20260517.sqlite3`
  - JSON: `reports/agentic_memory/locomo1_online_growth_v2_5_entityfirst_smoke160_20260517.json`
  - Result on conv-26 first 160 turns: final nodes 15, evolved 4; evolved slugs `adoption`, `difference`, `dreams`, `lgbtq`.
- v2.6 speaker-filter entity-first:
  - Smoke DB: `data/agentic_smoke/locomo2_online_growth_v2_6_speakerfilter_smoke160_20260517.sqlite3`
  - Full DB: `data/agentic_smoke/locomo2_online_growth_v2_6_speakerfilter_fair80a48_nodelocal_evidenceonly_20260517.sqlite3`
  - Full JSON: `reports/agentic_memory/locomo2_online_growth_v2_6_speakerfilter_fair80a48_nodelocal_evidenceonly_20260517.json`
  - QA: `reports/agentic_memory/locomo2_evo_online_growth_v2_6_speakerfilter_fair80a48_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Raw candidates | Candidate atoms | Avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 0.00 | 0.00 | 2075.88 |
| CurrentEvolvedAnchor subset | 233 | 0.5772 | 0.5106 | 0.4191 | 0.3552 | 0.5409 | 0.7967 | 52.88 | 17.06 | 2331.87 |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 52.90 | 17.71 | 2300.35 |
| OnlineGrowth-v2.2-PrimaryGate TopK3 | 233 | 0.5728 | 0.5090 | 0.3850 | 0.3487 | 0.5334 | 0.8186 | 47.51 | 15.28 | 2756.77 |
| OnlineGrowth-v2.4-QualityGateV1 TopK3 | 233 | 0.5705 | 0.5049 | 0.4168 | 0.3643 | 0.5351 | 0.7822 | 51.17 | 17.67 | 2633.21 |
| OnlineGrowth-v2.6-SpeakerFilterEntityFirst TopK3 | 233 | 0.5635 | 0.4952 | 0.3809 | 0.3525 | 0.5358 | 0.7819 | 47.70 | 17.58 | 1519.31 |

Interpretation:

- v2.4 quality-v1 improved multi-hop/open over v2.1, but lost too much on temporal and single-hop. It is a useful diagnostic, not a mainline candidate.
- v2.2 remains useful as evidence that route-only evolved topics can improve temporal, but it sacrifices multi-hop too much.
- Current best medium anchor remains `OnlineGrowth-v2.1-CleanRouteNodeLocal-TopK3`.
- v2.5 entity-first smoke improved topic labels for early conv-26 (`adoption`, `lgbtq`) and removed participant-name topics, but still generated abstract labels (`difference`, `dreams`). Full v2.5 fair run is the next candidate.
- v2.6 speaker-filter removed participant/greeting topic pollution and produced cleaner labels in smoke (`business`, `dance`, `brand`, `store`), but full QA was worse than v2.1. The likely issue is not route naming; it is overly broad primary assignment to strong evolved topics such as `business`, `dance`, and `lgbtq`.
- Next best direction: keep v2.1 as the anchor, and test clean evolved topics as route/secondary-only or limited-primary children instead of letting entity-first topics absorb many future atoms as primary assignments.

## 2026-05-17 OnlineGrowth v2.7-v2.10 Negative Results And Current Decision

Goal:

- Decide whether the newest online evolved topic-tree variants are strong enough to justify a full LoCoMo run.
- Keep the fair medium input fixed: LoCoMo `conv-26` + `conv-30`, 233 QA, Qwen3-235B answer model, no question-type-specific routing.

New artifacts:

- v2.7 CleanRoute-RouteOnly ingest:
  - DB: `data/agentic_smoke/locomo2_online_growth_v2_7_cleanroute_routeonly_fair80a48_nodelocal_evidenceonly_20260517.sqlite3`
  - Growth JSON: `reports/agentic_memory/locomo2_online_growth_v2_7_cleanroute_routeonly_fair80a48_nodelocal_evidenceonly_20260517.json`
  - QA: `reports/agentic_memory/locomo2_evo_online_growth_v2_7_cleanroute_routeonly_fair80a48_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`
- v2.8 RouteOnly-MergedAnswer partial QA:
  - QA progress: `reports/agentic_memory/locomo2_evo_online_growth_v2_8_routeonly_mergedanswer_keywordtopk3_textpolicy_sample233_20260517.qa_progress.jsonl`
  - Stopped after 96 QA because partial F1 was clearly poor.
- v2.9 CleanRoute-QualityV1 ingest:
  - DB: `data/agentic_smoke/locomo2_online_growth_v2_9_cleanroute_qualityv1_fair80a48_nodelocal_evidenceonly_20260517.sqlite3`
  - Growth JSON: `reports/agentic_memory/locomo2_online_growth_v2_9_cleanroute_qualityv1_fair80a48_nodelocal_evidenceonly_20260517.json`
  - QA progress: `reports/agentic_memory/locomo2_evo_online_growth_v2_9_cleanroute_qualityv1_fair80a48_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.qa_progress.jsonl`
  - Stopped after 95 QA because partial F1 was clearly poor.
- v2.10 v2.1-DB HighOverlap4 policy partial QA:
  - QA progress: `reports/agentic_memory/locomo2_evo_online_growth_v2_10_v21db_highoverlap4_keywordtopk3_textpolicy_sample233_20260517.qa_progress.jsonl`
  - Stopped after 89 QA because the policy suppressed almost all topic evidence and regressed toward baseline.

Implemented code changes:

- `scripts/eval_locomo.py`, `scripts/eval_locomo_qa_only_parallel.py`
  - Added `--answer-evidence-mode {auto,baseline,merged}`.
  - Default `auto` preserves the previous topic-labeled behavior: `structured_context_topic_labeled` answers use baseline evidence as the core answer payload.
  - `merged` lets accepted topic-soft evidence enter the answer payload. v2.8 showed this is too noisy as a default.

Final/partial metrics:

| Variant | Status | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LEAF-Base subset conv-26+30 | full medium | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | Original LEAF baseline, no topic tree/evolution |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | full medium | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | Current best medium anchor |
| OnlineGrowth-v2.7-CleanRoute-RouteOnly TopK3 | full medium | 233 | 0.5621 | 0.4948 | 0.4149 | 0.3622 | 0.5105 | 0.7971 | Cleaner tree, but disabling evolved primary assignment hurt single-hop badly |
| OnlineGrowth-v2.8-RouteOnly-MergedAnswer TopK3 | partial | 96 | 0.5508 | 0.4751 | 0.4114 | 0.3643 | 0.5005 | 0.7558 | Merging topic evidence into answer payload was too noisy |
| OnlineGrowth-v2.9-CleanRoute-QualityV1 TopK3 | partial | 95 | 0.5500 | 0.4752 | 0.3961 | 0.3525 | 0.5134 | 0.7653 | Quality-gated primary assignment still regressed early |
| OnlineGrowth-v2.10-v2.1DB-HighOverlap4 TopK3 | partial | 89 | 0.5726 | 0.4988 | 0.3964 | 0.3643 | 0.5155 | 0.8090 | Too conservative; avg selected topic events only 0.04 |

Diagnostics:

- v2.7 structure was clean and route-only worked as designed:
  - all evolved topics had `primary_assignment_exposure=inactive`;
  - evolved topics only contributed `atom_secondary` evidence.
  - This removed broad primary-topic pollution, but also removed the main mechanism that made v2.1 useful.
- v2.9 structure activated only a few stronger evolved topics:
  - conv-26 active examples: `acceptance`, `adoption`, `lgbtq`, `marrying`;
  - conv-30 active examples: `dance`, `stress_relief`, `studio`, `dance_studio`;
  - however partial QA still regressed, especially multi-hop/open.
- v2.8 proved that directly merging topic evidence into the answer payload is not safe.
- v2.10 proved that requiring very high text overlap suppresses almost all topic evidence and does not create a useful evolved-memory benefit.

Decision:

- Do **not** run full LoCoMo for v2.7, v2.8, v2.9, or v2.10.
- Current best full-medium anchor remains `OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3`, but the gain over `LEAF-Base` is only +0.0016 F1 on the 233QA medium subset.
- Next optimization should not be answer-side merging. It should focus on retrieval-side use of topic-tree structure:
  - use topic assignments as a soft feature in event/atom reranking;
  - keep topic evidence out of the core answer payload unless confidence is very high;
  - avoid broad evolved topics absorbing primary assignment without a local contrast/coverage constraint.

Follow-up retrieval-side smoke:

- Implemented an experimental default-off `LEAF_TOPIC_RERANK=1` switch in `src/leaf/search.py`.
  - It adds a small non-LLM topic-assignment bonus to event scoring.
  - It suppresses speaker-name tokens when matching query terms to topic labels/keywords.
  - It records diagnostics under `timing.topic_rerank`.
- Smoke command used v2.1 DB and 40 QA (`--qa-per-sample 20`):
  - Output: `reports/agentic_memory/locomo2_evo_online_growth_v2_11_v21db_topicrerank_smoke40_keywordtopk3_textpolicy_20260517.json`

Same-40 smoke results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base same40 | 40 | 0.7488 | 0.6969 | 0.5289 | 0.8334 | 0.7432 | 0.8661 |
| OnlineGrowth-v2.1 same40 | 40 | 0.7422 | 0.6954 | 0.5230 | 0.8334 | 0.7258 | 0.8596 |
| v2.11 TopicRerank same40 | 40 | 0.7351 | 0.6822 | 0.5394 | 0.8334 | 0.6429 | 0.8451 |

Decision:

- Do **not** run v2.11 233QA or full LoCoMo.
- The naive topic-rerank bonus improved multi-hop in the smoke but hurt single-hop and temporal. It confirms that topic-tree retrieval coupling must be more selective than simple keyword/topic assignment boosts.
- Keep `LEAF_TOPIC_RERANK` default-off as an experimental prototype only.

## 2026-05-17 OnlineGrowth v2.12-v2.14 Selective Topic Coupling

Purpose:

- Continue from the negative v2.7-v2.11 results and test whether retrieval-time topic evidence can be made more selective without using LLM search or benchmark question-type labels.
- All experiments reused the current best online-growth DB:
  - `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Input benchmark:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Config:
  - `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Answer model remained Qwen3-235B via local OpenAI-compatible endpoint; memory/search did not use an LLM.

Implementation changes:

- `src/leaf/topic_soft.py`
  - Added optional query-event embedding diagnostics for selected topic events.
  - Added optional pre-selection semantic gate parameters in `topic_soft_expand_events`.
  - Added policy `route_uncertainty_semantic_v0` in `apply_topic_soft_policy`:
    - can suppress topic evidence when more than one topic route is active;
    - can suppress topic evidence when selected event semantic similarity is below a threshold;
    - preserves existing temporal suppression behavior.
- `scripts/eval_locomo.py`, `scripts/eval_locomo_qa_only_parallel.py`
  - Added `--topic-soft-min-event-embedding-similarity`.
  - Added `--topic-soft-policy-min-selected-semantic-similarity`.
  - Added `--topic-soft-policy-suppress-multi-route`.
  - Recorded semantic/policy diagnostics per row and in summaries.

Validation:

```bash
scripts/with_tracenav_nlp.sh python -m py_compile \
  src/leaf/topic_soft.py scripts/eval_locomo.py scripts/eval_locomo_qa_only_parallel.py
```

Artifacts:

- v2.12 semantic pre-selection gate smoke40:
  - `reports/agentic_memory/locomo2_evo_online_growth_v2_12_v21db_semgate035_smoke40_keywordtopk3_textpolicy_20260517.json`
  - Log: `reports/agentic_memory/locomo2_evo_online_growth_v2_12_v21db_semgate035_smoke40_keywordtopk3_textpolicy_20260517.log`
- v2.13 route-uncertainty + semantic policy smoke40:
  - `reports/agentic_memory/locomo2_evo_online_growth_v2_13_v21db_routeuncert_semantic_policy_smoke40_keywordtopk3_20260517.json`
  - Log: `reports/agentic_memory/locomo2_evo_online_growth_v2_13_v21db_routeuncert_semantic_policy_smoke40_keywordtopk3_20260517.log`
- v2.14 multi-route policy smoke40 and 233QA:
  - Smoke: `reports/agentic_memory/locomo2_evo_online_growth_v2_14_v21db_multiroute_policy_smoke40_keywordtopk3_20260517.json`
  - Smoke log: `reports/agentic_memory/locomo2_evo_online_growth_v2_14_v21db_multiroute_policy_smoke40_keywordtopk3_20260517.log`
  - 233QA: `reports/agentic_memory/locomo2_evo_online_growth_v2_14_v21db_multiroute_policy_sample233_keywordtopk3_20260517.json`
  - 233QA log: `reports/agentic_memory/locomo2_evo_online_growth_v2_14_v21db_multiroute_policy_sample233_keywordtopk3_20260517.log`
- Diagnostic:
  - `reports/agentic_memory/locomo2_v21_topic_event_embedding_diagnostic_20260517.json`

Same-40 smoke results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LEAF-Base same40 | 40 | 0.7488 | 0.6969 | 0.5289 | 0.8334 | 0.7432 | 0.8661 | Anchor |
| OnlineGrowth-v2.1 same40 | 40 | 0.7422 | 0.6954 | 0.5230 | 0.8334 | 0.7258 | 0.8596 | Current best medium anchor but weak smoke |
| v2.12 SemanticPreselect035 | 40 | 0.7381 | 0.6872 | 0.5230 | 0.8334 | 0.6429 | 0.8596 | Worse than v2.1; no medium |
| v2.13 RouteUncertainty+SemanticPolicy035 | 40 | 0.7419 | 0.6947 | 0.5230 | 0.8334 | 0.7188 | 0.8596 | Still below v2.1; no medium |
| v2.14 MultiRoutePolicyOnly | 40 | 0.7572 | 0.7121 | 0.5230 | 0.8334 | 0.6764 | 0.8901 | Passed smoke, ran 233QA |

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Avg search ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 2075.88 | Baseline anchor |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 2300.35 | Current best evolved anchor |
| v2.14 MultiRoutePolicyOnly | 233 | 0.5708 | 0.5040 | 0.4064 | 0.3605 | 0.5353 | 0.7905 | 631.56 | Failed; no full LoCoMo |

v2.14 diagnostics:

- Topic evidence average event count: `0.5665`.
- Policy reason counts:
  - `selected`: 132
  - `temporal_query_text_suppressed`: 74
  - `multi_route_uncertain`: 26
  - `no_topic_event`: 1
- v2.14 versus v2.1 on the same 233 rows:
  - mean F1 delta: `-0.0083`
  - improved rows: 9
  - hurt rows: 12
  - unchanged rows: 212
- By policy reason, `multi_route_uncertain` rows were positive on average (`+0.0055` F1), but unchanged `selected` and temporal-suppressed rows drifted downward overall:
  - `selected`: `-0.0051`
  - `temporal_query_text_suppressed`: `-0.0188`
  - `multi_route_uncertain`: `+0.0055`

Interpretation:

- v2.12 showed that pre-selection embedding gating is the wrong insertion point: it can replace a low-similarity topic event with another topic event instead of suppressing topic evidence entirely.
- v2.13 moved the semantic check to policy level, which was cleaner but still did not beat the v2.1 smoke.
- v2.14 showed that suppressing uncertain multi-route topic evidence can help the rows it touches, but the full 233QA result is unstable and regresses overall.
- The smoke40 improvement was misleading because it overrepresented early temporal questions; the full 233QA exposed lower single-hop and temporal scores.

Decision:

- Do **not** run full LoCoMo for v2.12, v2.13, or v2.14.
- Do **not** promote v2.14 as a candidate despite the good smoke40.
- Current best evolved/online anchor remains:
  - `OnlineGrowth-v2.1-CleanRouteNodeLocal-TopK3`
  - QA: `reports/agentic_memory/locomo2_evo_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`
- The next useful direction is not more retrieval-time hard gating. It should target memory modeling itself:
  - topic growth quality and topic granularity;
  - stable topic assignment during online ingest;
  - self-QA or unsupervised checks to validate topic usefulness before retrieval exposure;
  - answer-prompt determinism/ordering, because small topic-policy changes caused answer drift even for rows where topic evidence was not the main difference.

## 2026-05-17 OnlineGrowth v2.15 Topic Pool Width Gate

Purpose:

- Test a cleaner topic-tree uncertainty signal after v2.14 failed: suppress topic evidence when the selected topic candidate pool is too broad.
- This uses existing `topic_soft_selective` mechanics, not question-type labels or domain-specific hard-coded filters.
- Reused the v2.1 online-growth DB:
  - `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`

Offline replay before running:

- Rule simulated on frozen v2.1/vs-base rows: suppress accepted topic evidence when `topic_soft_candidate_atom_count > 28`.
- Estimated hybrid F1: `0.5842` on 233QA, suppressing 16 topic events.
- This estimate was optimistic because actual Qwen answer generation is not deterministic enough to be faithfully replaced by frozen v2.1/base rows.

Command shape:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_evo_online_growth_v2_15_v21db_poolmax28_textpolicy_sample233_keywordtopk3_20260517.json \
  --sample-limit 2 --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-view-mode heuristic --answer-style structured_context_topic_labeled \
  --retrieval-mode topic_soft_selective --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 28 \
  --temporal-postprocess range --short-answer-postprocess precise --qa-workers 6
```

Artifacts:

- QA: `reports/agentic_memory/locomo2_evo_online_growth_v2_15_v21db_poolmax28_textpolicy_sample233_keywordtopk3_20260517.json`
- Log: `reports/agentic_memory/locomo2_evo_online_growth_v2_15_v21db_poolmax28_textpolicy_sample233_keywordtopk3_20260517.log`

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Avg search ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 2075.88 | Baseline anchor |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 2300.35 | Current best evolved anchor |
| v2.14 MultiRoutePolicyOnly | 233 | 0.5708 | 0.5040 | 0.4064 | 0.3605 | 0.5353 | 0.7905 | 631.56 | Failed |
| v2.15 PoolMax28 TextPolicy | 233 | 0.5717 | 0.5020 | 0.4157 | 0.3487 | 0.5413 | 0.7794 | 678.45 | Failed; no full LoCoMo |

v2.15 diagnostics:

- Policy reason counts:
  - `selected`: 142
  - `temporal_query_text_suppressed`: 74
  - `too_many_candidate_atoms`: 16
  - `no_topic_event`: 1
- Topic evidence average event count: `0.6094`.
- v2.15 versus v2.1 on the same 233 rows:
  - mean F1 delta: `-0.0073`
  - improved rows: 6
  - hurt rows: 12
  - unchanged rows: 215
- By policy reason:
  - `too_many_candidate_atoms`: `+0.0094` mean F1 delta, 1 improve / 0 hurt.
  - `selected`: `+0.0017`, 4 improve / 5 hurt.
  - `temporal_query_text_suppressed`: `-0.0283`, 1 improve / 7 hurt.

Interpretation:

- Candidate-pool width is a plausible topic uncertainty signal: the rows actually suppressed by `too_many_candidate_atoms` were positive on average.
- The full run still failed because temporal-suppressed rows and some unchanged selected rows drifted downward in fresh answer generation.
- The repeated pattern across v2.14 and v2.15 is that narrow retrieval policy changes can look good in offline replay or smoke subsets but do not yield stable 233QA improvements.
- This points to answer-prompt sensitivity and memory modeling quality as bottlenecks, not just retrieval-time topic gating.

Decision:

- Do **not** run full LoCoMo for v2.15.
- Current best evolved/online anchor remains `OnlineGrowth-v2.1-CleanRouteNodeLocal-TopK3`:
  - `reports/agentic_memory/locomo2_evo_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_keywordtopk3_textpolicy_sample233_20260517.json`
- For the next iteration, avoid adding more small retrieval-time gates unless backed by a stable non-LLM validation signal. Prefer improving:
  - online topic growth quality and granularity;
  - self-QA-based topic exposure validation;
  - deterministic answer-context ordering and temporal handling;
  - topic tree evolution before retrieval exposure, rather than per-query suppression only.

## 2026-05-17 OnlineGrowth v2.16-v2.18 Answer Determinism And Topic Policy Retest

Purpose:

- Diagnose whether the v2.14/v2.15 regression came from retrieval-side topic policy or from fresh answer generation drift.
- Keep retrieval LLM-free. Answer model remained local `Qwen3-235B`; memory/topic evolution still used configured `gpt-5.4-mini` only for memory modeling tasks.
- Add deterministic OpenAI-compatible generation controls in config and test them on the same 233QA medium slice.

Code/config changes:

- `src/leaf/config.py`
  - Added optional `ModelConfig.top_p` and `ModelConfig.seed`.
- `src/leaf/clients.py`
  - Passes optional `top_p` and `seed` through to `/chat/completions`.
- `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
  - Set answer-side `llm.top_p: 1.0` and `llm.seed: 0`.
- `scripts/eval_locomo.py`, `scripts/eval_locomo_qa_only_parallel.py`
  - Added experimental answer style `structured_context_topic_auto`.
  - It uses `structured_context_topic_labeled` only when accepted topic-soft evidence exists; otherwise it falls back to plain `structured_context`.
  - Records `effective_answer_style` per row.

Validation:

```bash
scripts/with_tracenav_nlp.sh python -m py_compile \
  src/leaf/config.py src/leaf/clients.py \
  scripts/eval_locomo.py scripts/eval_locomo_qa_only_parallel.py
```

Local Qwen endpoint accepted a minimal request with `top_p=1.0` and `seed=0`.

Important diagnostic:

- Comparing v2.1 and v2.15 showed most hurt rows had identical `answer_prompt_messages`, `answer_context_lines`, `answer_view`, and raw-span signatures.
- Therefore some of the measured 233QA movement is answer-side nondeterminism or sensitivity, not retrieval differences.
- Example: v2.15 had 12 hurt rows versus v2.1; many were identical-prompt temporal rows whose outputs differed.

Commands:

```bash
# v2.16: same v2.1 DB/policy, but topic-auto answer style and Qwen seed0.
scripts/with_tracenav_nlp.sh python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_onlinegrowth_v2_16_v21db_topicauto_seed0_sample233_20260517.json \
  --sample-limit 2 --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-view-mode heuristic --answer-style structured_context_topic_auto \
  --retrieval-mode topic_soft_selective --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range --short-answer-postprocess precise --qa-workers 6

# v2.17: same as v2.1, original topic-labeled answer style, Qwen seed0.
scripts/with_tracenav_nlp.sh python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_onlinegrowth_v2_17_v21db_topiclabeled_seed0_repeat_sample233_20260517.json \
  --sample-limit 2 --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-view-mode heuristic --answer-style structured_context_topic_labeled \
  --retrieval-mode topic_soft_selective --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range --short-answer-postprocess precise --qa-workers 6

# v2.18: retest candidate-pool uncertainty gate under Qwen seed0.
scripts/with_tracenav_nlp.sh python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_onlinegrowth_v2_18_v21db_poolmax28_topiclabeled_seed0_sample233_20260517.json \
  --sample-limit 2 --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-view-mode heuristic --answer-style structured_context_topic_labeled \
  --retrieval-mode topic_soft_selective --topic-router keyword --topic-route-top-k 3 \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 28 \
  --temporal-postprocess range --short-answer-postprocess precise --qa-workers 6
```

Artifacts:

- v2.16 QA JSON: `reports/agentic_memory/locomo2_onlinegrowth_v2_16_v21db_topicauto_seed0_sample233_20260517.json`
- v2.16 log: `reports/agentic_memory/locomo2_onlinegrowth_v2_16_v21db_topicauto_seed0_sample233_20260517.log`
- v2.17 QA JSON: `reports/agentic_memory/locomo2_onlinegrowth_v2_17_v21db_topiclabeled_seed0_repeat_sample233_20260517.json`
- v2.17 log: `reports/agentic_memory/locomo2_onlinegrowth_v2_17_v21db_topiclabeled_seed0_repeat_sample233_20260517.log`
- v2.18 QA JSON: `reports/agentic_memory/locomo2_onlinegrowth_v2_18_v21db_poolmax28_topiclabeled_seed0_sample233_20260517.json`
- v2.18 log: `reports/agentic_memory/locomo2_onlinegrowth_v2_18_v21db_poolmax28_topiclabeled_seed0_sample233_20260517.log`

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Avg search ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 12295.73 full-run avg | Baseline anchor |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 2300.35 | Existing evolved anchor |
| v2.16 TopicAuto+Seed0 | 233 | 0.5559 | 0.4875 | 0.4097 | 0.3643 | 0.5213 | 0.7577 | 2112.71 | Failed |
| v2.17 TopicLabeled+Seed0 | 233 | 0.5799 | 0.5112 | 0.4010 | 0.3605 | 0.5488 | 0.8038 | 1845.79 | Best F1 repeat, not full candidate by BLEU |
| v2.18 PoolMax28+Seed0 | 233 | 0.5721 | 0.5041 | 0.3960 | 0.3605 | 0.5442 | 0.7863 | 1784.59 | Failed |

Interpretation:

- `structured_context_topic_auto` is not useful as implemented. Even though it looked semantically cleaner, it hurt single-hop and temporal heavily.
- Qwen `seed=0` plus original topic-labeled prompt gives a slightly higher F1 repeat than v2.1, but BLEU1 is lower. Treat it as answer-side stability hygiene, not a method gain.
- Candidate-pool uncertainty (`PoolMax28`) remains unstable and should not be promoted.
- The reliable evolved/online anchor is still the v2.1/v2.17 line: online topic growth with clean routing and topic-labeled answer payload. v2.17 can be used for future seeded repeats; v2.1 remains the better BLEU1 anchor.

Decision:

- Do **not** run full LoCoMo for v2.16 or v2.18.
- Do **not** claim v2.17 as a substantive method gain; it is a seeded repeat with marginal F1 improvement and BLEU1 regression.
- Keep `top_p`/`seed` config support for reproducibility.
- Do not use `structured_context_topic_auto` in the mainline unless later evidence supports it.

## 2026-05-17 Self-QA Criteria v1 Smoke

Purpose:

- Make self-QA usable as an explicit evolution signal, not only as a generated question/answer pair.
- Add per-task criteria that can diagnose:
  - whether required evidence was retrieved;
  - whether expected topic nodes were routed/retrieved;
  - whether the gold answer has checkable answer constraints;
  - whether a failure is likely evidence, topic routing, temporal, or entity related.

Implementation:

- `scripts/build_selfqa_from_memory.py`
  - Added `criteria_v1` to the requested LLM JSON shape.
  - Normalizes generated criteria into `metadata.criteria_v1`.
  - Adds local criteria validation before accepting a task.
  - Adds validator checks for `criteria_valid` and `criteria_complete`.
- `scripts/eval_memory_search.py`
  - Reads `metadata.criteria_v1`.
  - Reports required event/atom hit rates from `evidence_criteria` and `retrieval_criteria`.
  - Reports expected topic route/retrieval hit rates from `topic_criteria`.
  - Keeps original gold path and topic shadow metrics for backward compatibility.

Criteria schema:

```json
{
  "version": "criteria_v1",
  "answer_criteria": {
    "must_contain": ["..."],
    "acceptable_aliases": ["..."],
    "temporal_granularity": "none|date|month|year|relative|ordering",
    "wrong_if_contains": ["..."]
  },
  "evidence_criteria": {
    "required_event_ids": ["..."],
    "required_atom_ids": ["..."],
    "all_required": true,
    "evidence_roles": [
      {"event_id": "...", "atom_id": "...", "role": "anchor|bridge|temporal_anchor|support"}
    ]
  },
  "retrieval_criteria": {
    "success_at_k": 8,
    "must_retrieve_any_event_ids": ["..."],
    "must_retrieve_all_event_ids": ["..."],
    "must_retrieve_any_atom_ids": ["..."],
    "must_retrieve_all_atom_ids": ["..."]
  },
  "topic_criteria": {
    "expected_topic_ids": ["..."],
    "topic_should_help": true,
    "failure_mode_if_missing": "miss_bridge|wrong_time|wrong_entity|overbroad_topic|wrong_topic|unknown"
  }
}
```

Validation:

```bash
scripts/with_tracenav_nlp.sh python -m py_compile \
  scripts/build_selfqa_from_memory.py scripts/eval_memory_search.py
```

Smoke generation command:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/build_selfqa_from_memory.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 \
  --output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl \
  --summary-output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_summary_20260517.json \
  --limit 6 --candidate-limit 60 \
  --task-types single_fact multi_hop temporal \
  --validate --min-validation-score 0.75
```

Smoke generation result:

- Output JSONL: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl`
- Summary: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_summary_20260517.json`
- Log: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.log`
- Accepted tasks: `6`
- Candidate count: `1412`
- Candidate sample count: `60`
- Error count: `3`
- Task type counts:
  - `single_fact`: `3`
  - `temporal`: `3`
  - `multi_hop`: `0`

Observation:

- Criteria were written into all accepted tasks under `metadata.criteria_v1`.
- The local validator and LLM validator both accepted these tasks.
- Multi-hop remains the weak point: three attempted multi-hop candidates were rejected as invalid/unsupported in the smoke.

Search eval command:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 \
  --selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl \
  --output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_smoke_20260517.json \
  --snapshot-limit 6 --raw-span-limit 8 \
  --topic-routing-shadow --topic-router keyword
```

Search eval smoke result:

- Report: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_smoke_20260517.json`
- Task count: `6`
- Mean event recall: `0.6667`
- Mean atom recall: `0.6667`
- Criteria required event hit rate: `0.6667`
- Criteria required atom hit rate: `0.6667`
- Answer criteria static pass rate: `1.0000`
- Mean topic recall from routing: `0.3333`
- Criteria expected topic route hit rate: `0.3333`
- Criteria expected topic retrieval hit rate: `1.0000`

Interpretation:

- The new criteria metrics separate retrieval and topic routing failure modes.
- In this smoke, the final retrieved evidence covered the expected topics, but the keyword topic router often did not route directly to the expected topic.
- This is exactly the kind of signal we need for later topic-tree evolution:
  - if required evidence is missing, improve memory search or topic exposure;
  - if expected topic route is missing but retrieval topic is present, improve routing/profile terms;
  - if both are missing, consider topic growth, assignment, or atom extraction quality.

Decision:

- Keep `criteria_v1` in self-QA generation and search eval.
- Do not yet wire criteria directly into topic evolution decisions; next step should be an ablation that uses criteria failures to propose topic profile/keyword updates, then validates them with self-QA search metrics before LoCoMo QA.

## 2026-05-17 Criteria-Driven Topic Profile Patch v1.2

Purpose:

- Test whether `self-QA criteria_v1` can drive topic-tree self-evolution by patching topic route/profile keywords from criteria topic-routing failures.
- This is a memory-modeling experiment: the patch script creates a candidate memory view and does not promote it.

Implementation changes:

- `scripts/patch_topic_view_from_criteria.py`
  - Reads a criteria-bearing self-QA JSONL plus `scripts/eval_memory_search.py` output.
  - Finds tasks where `criteria_expected_topic_route_hit=false` while `criteria_expected_topic_retrieval_hit=true`.
  - Clones the base topic view into a candidate view and patches only route/node keywords for expected topics.
  - v1.2 term extraction is conservative: it uses question terms that overlap with source atom content/entities, filters speaker-name-only terms, pure numeric terms, and terms shared across multiple target topics.
- `scripts/eval_memory_search.py`
  - Resolves criteria expected topic ids through candidate-view aliases such as `base_topic_id` and `criteria_patch_clone_of_topic_id`.
- `src/leaf/normalize.py` and `src/leaf/topic_soft.py`
  - Added optional English Snowball stemmed content tokens for topic-soft atom scoring.
  - Important: after v2.19, stemming is no longer the default; it is gated by `use_stemmed_content_tokens` / `--topic-soft-use-stemmed-content-tokens` to avoid silently changing the anchor path.
- `scripts/eval_locomo_qa_only_parallel.py`
  - Added `--topic-view-map` so individual corpora can use candidate views while other corpora use their active views.

Self-QA criteria smoke inputs:

- DB: `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Corpus: `locomo_conv_26`
- Self-QA: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl`
- Baseline search eval: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_smoke_20260517.json`

Patch command:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/patch_topic_view_from_criteria.py \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 \
  --selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl \
  --search-eval-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_smoke_20260517.json \
  --output reports/agentic_memory/locomo_conv_26_criteria_topic_profile_patch_v1_2_smoke_20260517.json \
  --name criteria-topic-profile-patch-v1-2-smoke \
  --max-added-keywords-per-topic 6 \
  --max-route-keywords 24 \
  --max-node-keywords 32
```

Candidate view:

- View id: `view_ba5ca69fe27deecefc`
- Base view id: `view_623b1749dd19f66929`
- Patched topic count: `3`
- Patched topics:
  - `events_timeline`: added `photo`, `ago`, `mentions`, `met`
  - `preferences_opinions`: added `abstract art`, `abstract`, `art`, `enjoys`, `painting`
  - `lgbtq`: added `volunteer`
- Patch artifact: `reports/agentic_memory/locomo_conv_26_criteria_topic_profile_patch_v1_2_smoke_20260517.json`

Self-QA search eval results:

| Variant | Retrieval mode | View | Tasks | Expected topic route hit | Mean event recall | Mean atom recall |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline criteria search | baseline | `view_623b1749dd19f66929` | 6 | 0.3333 | 0.6667 | 0.6667 |
| CriteriaPatch-v1.2 shadow | baseline | `view_ba5ca69fe27deecefc` | 6 | 1.0000 | 0.6667 | 0.6667 |
| CriteriaPatch-v1.2 + topic_soft + stem | topic_soft | `view_ba5ca69fe27deecefc` | 6 | 1.0000 | 1.0000 | 1.0000 |

Search eval artifacts:

- Patched route shadow: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_patch_v1_2_smoke_20260517.json`
- Patched topic-soft + stem smoke: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_topicsoft_patch_v1_2_stem_smoke_20260517.json`

233QA command:

```bash
tmux new-session -d -s leaf_v219_criteria_patch_sample233_20260517 "cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && scripts/with_tracenav_nlp.sh python -u scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo2_onlinegrowth_v2_19_criteria_patch_v12_stem_sample233_20260517.json \
  --sample-limit 2 --snapshot-limit 8 --raw-span-limit 8 --non-temporal-raw-span-limit 12 \
  --answer-view-mode heuristic --answer-style structured_context_topic_labeled \
  --retrieval-mode topic_soft_selective --topic-router keyword --topic-route-top-k 3 \
  --topic-view-map '{\"locomo_conv_26\":\"view_ba5ca69fe27deecefc\"}' \
  --topic-soft-event-limit 1 --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 --topic-soft-allow-fallback-topic \
  --topic-soft-fallback baseline_on_unknown \
  --topic-soft-policy text_temporal_suppressed_v0 \
  --topic-soft-policy-min-selected-overlap 1 \
  --topic-soft-policy-max-candidate-atoms 999 \
  --temporal-postprocess range --short-answer-postprocess precise --qa-workers 6 \
  2>&1 | tee reports/agentic_memory/locomo2_onlinegrowth_v2_19_criteria_patch_v12_stem_sample233_20260517.log"
```

233QA outputs:

- QA JSON: `reports/agentic_memory/locomo2_onlinegrowth_v2_19_criteria_patch_v12_stem_sample233_20260517.json`
- QA log: `reports/agentic_memory/locomo2_onlinegrowth_v2_19_criteria_patch_v12_stem_sample233_20260517.log`
- QA progress: `reports/agentic_memory/locomo2_onlinegrowth_v2_19_criteria_patch_v12_stem_sample233_20260517.qa_progress.jsonl`

233QA medium results:

| Variant | Questions | F1 | BLEU1 | Multi-hop F1 | Open F1 | Single-hop F1 | Temporal F1 | Avg search ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base subset conv-26+30 | 233 | 0.5774 | 0.5110 | 0.4075 | 0.3589 | 0.5439 | 0.7991 | 2075.88 |
| OnlineGrowth-v2.1-CleanRouteNodeLocal TopK3 | 233 | 0.5790 | 0.5132 | 0.4146 | 0.3605 | 0.5422 | 0.8030 | 2300.35 |
| v2.17 TopicLabeled+Seed0 repeat | 233 | 0.5799 | 0.5112 | 0.4010 | 0.3605 | 0.5488 | 0.8038 | 1845.79 |
| v2.19 CriteriaPatch-v1.2+Stem | 233 | 0.5692 | 0.5024 | 0.3996 | 0.3487 | 0.5365 | 0.7898 | 1875.09 |

Per-corpus result vs anchors:

| Variant | Corpus | Questions | F1 | BLEU1 | Multi | Open | Single | Temporal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2.19 | conv-26 | 152 | 0.5759 | 0.5072 | 0.3917 | 0.3487 | 0.5866 | 0.7948 |
| v2.17 | conv-26 | 152 | 0.5858 | 0.5156 | 0.3935 | 0.3605 | 0.6010 | 0.8023 |
| v2.1 | conv-26 | 152 | 0.5907 | 0.5222 | 0.4118 | 0.3605 | 0.5999 | 0.8090 |
| v2.19 | conv-30 | 81 | 0.5568 | 0.4934 | 0.4227 | 0.0000 | 0.4568 | 0.7828 |
| v2.17 | conv-30 | 81 | 0.5690 | 0.5029 | 0.4227 | 0.0000 | 0.4657 | 0.8059 |
| v2.1 | conv-30 | 81 | 0.5571 | 0.4964 | 0.4227 | 0.0000 | 0.4506 | 0.7943 |

Row-level diagnosis vs v2.1 on patched corpus `conv-26`:

- Changed rows: `12`
- Harmed rows: `10`
- Improved rows: `2`
- Net F1 delta over changed rows: `-2.2543`
- Largest harms:
  - q29 temporal: `Friday before 15 July 2023` became `8 July, 2023` (`-0.5` F1)
  - q77 multi-hop: `19 October 2023` became `Weekend before 20 October 2023` (`-0.5` F1)
  - q135 single-hop adoption advice dropped several required details (`-0.3446` F1)
  - q31 open-domain LGBTQ membership flipped `No` to `Yes` (`-0.1538` F1)

Interpretation:

- The self-QA criteria patch did exactly what it was designed to do on the self-QA smoke: expected topic route hit improved from `0.3333` to `1.0000`.
- That routing gain did not transfer to LoCoMo QA. On the patched corpus (`conv-26`), F1 dropped against both v2.1 and v2.17.
- The main failure mode is overfitting from narrow self-QA route misses to broader benchmark questions. The patched view routes more often to `events_timeline` and `preferences_opinions`, but answer quality falls on temporal/open/single-hop rows.
- Therefore criteria should be used as a validation/training signal for topic evolution gates, not directly as a source of route keywords without a held-out self-QA or benchmark-independent promotion gate.

Decision:

- Do **not** promote `view_ba5ca69fe27deecefc`.
- Do **not** use v2.19 for full LoCoMo.
- Keep the patch script for controlled ablations and future offline learning, but require a promotion gate that evaluates candidate changes on held-out self-QA before retrieval exposure.
- Keep optional stemmed topic-soft scoring behind an explicit flag only. Default topic-soft retrieval remains the v2.1/v2.17 behavior.

Recommended next step:

- Build a two-split self-QA evolution loop:
  - generate criteria tasks periodically;
  - use one split to propose topic/profile changes;
  - evaluate on a held-out split;
  - promote only changes that improve criteria evidence recall and do not regress route precision/candidate-pool size.
- For route/profile updates, prefer topic-local learned scoring weights or dormant metadata suggestions over direct keyword exposure.

Reproduction note added after hygiene cleanup:

- The recorded v2.19 run was executed while stemmed topic-soft content tokens were temporarily the default during debugging.
- After the run, default topic-soft behavior was restored to the v2.1/v2.17 non-stem path, and stemming was gated behind `--topic-soft-use-stemmed-content-tokens` / `use_stemmed_content_tokens`.
- To reproduce the exact v2.19 run with the current code, add `--topic-soft-use-stemmed-content-tokens` to the recorded 233QA command.

## 2026-05-17 Two-Split Self-QA Topic Patch Gate Smoke

Purpose:

- Add a held-out self-QA gate so criteria-driven topic/profile changes are not promoted just because they improve the same self-QA tasks that proposed them.
- This follows the v2.19 failure: direct criteria patching improved self-QA route hit but regressed LoCoMo QA.

Implementation changes:

- `scripts/split_selfqa_eval.py`
  - Splits a self-QA JSONL and matching `eval_memory_search.py` report by `task_id`.
  - Writes train/heldout self-QA JSONL and matching train/heldout eval reports.
- `scripts/evaluate_selfqa_topic_patch_gate.py`
  - Compares candidate vs baseline held-out self-QA reports.
  - Gates on task count, route-hit delta, event/atom recall delta/regression, candidate atom pool ratio, and route count ratio.
  - Can optionally promote and/or record a gate run; smoke did not promote.
- `scripts/eval_memory_search.py`
  - Summary now reports `avg_topic_soft_candidate_atom_count`, `avg_topic_soft_raw_candidate_atom_count`, `avg_topic_soft_filtered_atom_count`, and `topic_soft_use_stemmed_content_tokens`.

Inputs:

- DB: `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Corpus: `locomo_conv_26`
- Source self-QA: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl`
- Source eval report: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_smoke_20260517.json`

Split command:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/split_selfqa_eval.py \
  --selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_smoke_20260517.jsonl \
  --eval-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_search_eval_smoke_20260517.json \
  --train-selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_train_smoke_20260517.jsonl \
  --heldout-selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_smoke_20260517.jsonl \
  --train-eval-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_train_search_eval_smoke_20260517.json \
  --heldout-eval-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_search_eval_smoke_20260517.json \
  --summary-output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_split_summary_20260517.json \
  --heldout-ratio 0.5 --seed 17 --heldout-min-count 3
```

Split result:

- Common tasks: `6`
- Train tasks: `3`
- Heldout tasks: `3`
- Split summary: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_split_summary_20260517.json`

Candidate patch command from train split:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/patch_topic_view_from_criteria.py \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 \
  --selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_train_smoke_20260517.jsonl \
  --search-eval-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_train_search_eval_smoke_20260517.json \
  --output reports/agentic_memory/locomo_conv_26_criteria_topic_profile_patch_v1_2_gate_train_smoke_20260517.json \
  --name criteria-topic-profile-patch-v1-2-gate-train-smoke \
  --max-added-keywords-per-topic 6 --max-route-keywords 24 --max-node-keywords 32
```

Candidate view:

- View id: `view_f5af5280c3741425a4`
- Base view id: `view_623b1749dd19f66929`
- Patched topics:
  - `events_timeline`: added `photo`
  - `lgbtq`: added `volunteer`
- Patch artifact: `reports/agentic_memory/locomo_conv_26_criteria_topic_profile_patch_v1_2_gate_train_smoke_20260517.json`

Heldout eval commands:

```bash
scripts/with_tracenav_nlp.sh python -u scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 \
  --selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_smoke_20260517.jsonl \
  --output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_baseline_topicsoft_smoke_20260517.json \
  --snapshot-limit 6 --raw-span-limit 8 \
  --retrieval-mode topic_soft --topic-routing-shadow --topic-router keyword \
  --topic-view-id view_623b1749dd19f66929

scripts/with_tracenav_nlp.sh python -u scripts/eval_memory_search.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 \
  --selfqa reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_smoke_20260517.jsonl \
  --output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_candidate_topicsoft_smoke_20260517.json \
  --snapshot-limit 6 --raw-span-limit 8 \
  --retrieval-mode topic_soft --topic-routing-shadow --topic-router keyword \
  --topic-view-id view_f5af5280c3741425a4
```

Heldout results:

| View | Tasks | Route hit | Event recall | Atom recall | Avg topic-soft candidates | Route count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `view_623b1749dd19f66929` | 3 | 0.3333 | 0.6667 | 0.6667 | 26.6667 | 5 |
| Candidate `view_f5af5280c3741425a4` | 3 | 0.6667 | 1.0000 | 1.0000 | 32.0000 | 6 |

Gate commands and outputs:

```bash
# Loose gate: allows up to 1.25x candidate pool / route count.
scripts/with_tracenav_nlp.sh python -u scripts/evaluate_selfqa_topic_patch_gate.py \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 --view-id view_f5af5280c3741425a4 \
  --candidate-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_candidate_topicsoft_smoke_20260517.json \
  --baseline-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_baseline_topicsoft_smoke_20260517.json \
  --output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_eval_loose_smoke_20260517.json \
  --min-heldout-task-count 3 --max-candidate-atom-ratio 1.25 --max-route-count-ratio 1.25 --record-run

# Strict gate: allows only 1.05x candidate pool / route count.
scripts/with_tracenav_nlp.sh python -u scripts/evaluate_selfqa_topic_patch_gate.py \
  --db data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_26 --view-id view_f5af5280c3741425a4 \
  --candidate-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_candidate_topicsoft_smoke_20260517.json \
  --baseline-report reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_heldout_baseline_topicsoft_smoke_20260517.json \
  --output reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_eval_strict_smoke_20260517.json \
  --min-heldout-task-count 3 --max-candidate-atom-ratio 1.05 --max-route-count-ratio 1.05
```

Gate results:

- Loose gate output: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_eval_loose_smoke_20260517.json`
  - Passed: `true`
  - Candidate atom ratio: `1.20`
  - Route count ratio: `1.20`
- Strict gate output: `reports/agentic_memory/locomo_conv_26_selfqa_criteria_v1_gate_eval_strict_smoke_20260517.json`
  - Passed: `false`
  - Failed checks: `topic_soft_candidate_atom_ratio`, `route_count_ratio`

Interpretation:

- The gate now exposes the exact tradeoff that v2.19 hid: evidence recall improves, but routing/candidate pool also grows.
- With a strict pool budget, the candidate is rejected before any LoCoMo QA run.
- This is the desired control mechanism for an agentic memory system: self-QA can propose changes, but held-out self-QA plus candidate-pool constraints decide exposure.

Decision:

- Keep the two-split gate infrastructure.
- Do not promote the smoke candidate.
- Next substantial experiment should generate a larger self-QA criteria set per corpus so the gate is meaningful beyond this 6-task smoke.

## 2026-05-17 Larger Criteria Gate V2 On LoCoMo Conv-26/30

Scope:

- Purpose: test whether self-QA criteria can safely promote topic/profile changes using a train/heldout gate.
- DB: `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Environment: `scripts/with_tracenav_nlp.sh`
- Memory LLM for self-QA generation/validation: `gpt-5.4-mini` via `https://codeai.ysaikeji.cn/v1`
- Search/heldout gate uses no LLM.

Code changes:

- `scripts/build_selfqa_from_memory.py`
  - Added `--candidate-sampling stratified`, `--seed`, and `--stratified-turn-bucket-size`.
  - Candidate sampling can now spread self-QA generation across topic/time buckets instead of only taking sorted early atoms.
- `scripts/patch_topic_view_from_criteria.py`
  - Accepts optional `--config` for pipeline compatibility.
- `scripts/eval_memory_search.py`
  - Added `--topic-router profile_hybrid` for self-QA search evaluation.

Commands:

```bash
# Main run was launched in tmux:
tmux new-session -d -s leaf_criteria_gate_v2_resume \
  'cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   bash /tmp/leaf_criteria_gate_v2_resume.sh 2>&1 | tee logs/criteria_gate_v2_resume_20260517.log'
```

Self-QA generation:

| Corpus | Active view | Candidate count | Sampled candidates | Accepted validated tasks | Train | Heldout | Output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `locomo_conv_26` | `view_623b1749dd19f66929` | 1412 | 120 | 24 | 14 | 10 | `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_validated_20260517.jsonl` |
| `locomo_conv_30` | `view_290c4e577f9376dc06` | 1136 | 120 | 24 | 14 | 10 | `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_validated_20260517.jsonl` |

Patch summaries:

| Corpus | Candidate view | Base view | Source train tasks | Patched topics | Patch artifact |
| --- | --- | --- | ---: | ---: | --- |
| `locomo_conv_26` | `view_56f8c221ce0aeb24ed` | `view_623b1749dd19f66929` | 7 | 4 | `reports/agentic_memory/locomo_conv_26_criteria_gate_v2_patch_summary_20260517.json` |
| `locomo_conv_30` | `view_25fa3d33fd6e9ec639` | `view_290c4e577f9376dc06` | 8 | 5 | `reports/agentic_memory/locomo_conv_30_criteria_gate_v2_patch_summary_20260517.json` |

Strict heldout gate results using keyword routing:

| Corpus | Tasks | Base route hit | Candidate route hit | Base event/atom recall | Candidate event/atom recall | Candidate pool ratio | Route count ratio | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `locomo_conv_26` | 10 | 0.5000 | 0.5000 | 0.9500 / 0.9500 | 0.9500 / 0.9500 | 1.0833 | 1.0833 | failed |
| `locomo_conv_30` | 10 | 0.4000 | 0.4000 | 0.7500 / 0.7500 | 0.7500 / 0.7500 | 1.2421 | 1.1429 | failed |

Strict gate artifacts:

- `reports/agentic_memory/locomo_conv_26_criteria_gate_v2_strict_gate_20260517.json`
- `reports/agentic_memory/locomo_conv_30_criteria_gate_v2_strict_gate_20260517.json`

Interpretation:

- The naive criteria patch is not useful at this scale. It does not improve heldout evidence recall or expected-topic route hit, and it increases the topic-soft candidate pool.
- Examples of weak added route terms include `look`, `creating`, `place`, and `eventually`; these are not stable topic-discriminative memory concepts.
- Both candidates were correctly rejected and not promoted.

Profile-hybrid router check:

| Corpus | View | Router | Route hit | Topic recall | Event/atom recall | Avg topic-soft candidates |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `locomo_conv_26` | base | keyword | 0.5000 | 0.4500 | 0.9500 / 0.9500 | 19.2 |
| `locomo_conv_26` | base | profile_hybrid | 0.5000 | 0.4500 | 0.9500 / 0.9500 | 20.8 |
| `locomo_conv_30` | base | keyword | 0.4000 | 0.2333 | 0.7500 / 0.7500 | 19.0 |
| `locomo_conv_30` | base | profile_hybrid | 0.6000 | 0.3833 | 0.7500 / 0.7500 | 19.0 |

Profile-hybrid artifacts:

- `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_heldout_baseline_topicsoft_profilehybrid_20260517.json`
- `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_heldout_candidate_topicsoft_profilehybrid_20260517.json`
- `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_heldout_baseline_topicsoft_profilehybrid_20260517.json`
- `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_heldout_candidate_topicsoft_profilehybrid_20260517.json`

Decision:

- Do not run LoCoMo QA for the V2 criteria-patch candidates.
- Keep the self-QA train/heldout gate as the evolution safety check.
- Next implementation should move away from direct criteria-to-route-keyword patches and toward topic-discriminative profile updates plus profile-hybrid routing.
- Candidate terms should be accepted only when supported by target-topic atoms and relatively discriminative against non-target topics.

## 2026-05-17 Criteria Gate V3 Discriminative Profile Patch

Scope:

- Purpose: replace direct question-term route patches with more conservative topic-discriminative profile terms.
- Inputs reused from V2:
  - Train self-QA/eval:
    - `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_train_20260517.jsonl`
    - `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_train_base_topicsoft_20260517.json`
    - `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_train_20260517.jsonl`
    - `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_train_base_topicsoft_20260517.json`
  - Heldout self-QA:
    - `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_heldout_20260517.jsonl`
    - `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_heldout_20260517.jsonl`
  - Baseline profile-hybrid heldout reports:
    - `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v2_heldout_baseline_topicsoft_profilehybrid_20260517.json`
    - `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v2_heldout_baseline_topicsoft_profilehybrid_20260517.json`

Code change:

- `scripts/patch_topic_view_from_criteria.py`
  - Added `--patch-strategy discriminative_profile_v2`.
  - The patch is triggered by train self-QA route misses, but candidate terms must be supported by assigned target-topic atoms and bounded by cross-topic background presence.

Command:

```bash
tmux new-session -d -s leaf_criteria_gate_v3_profile \
  'cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   bash /tmp/leaf_criteria_gate_v3_profile.sh 2>&1 | tee logs/criteria_gate_v3_profile_20260517.log'
```

Results:

| Corpus | Candidate view | Patched topics | Heldout tasks | Route hit delta | Event/atom recall delta | Candidate pool ratio | Route count ratio | Strict gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `locomo_conv_26` | `view_78ea8b47bdd13d7fd7` | 8 | 10 | 0.0000 | 0.0000 / 0.0000 | 1.0000 | 1.0000 | passed |
| `locomo_conv_30` | `view_f09e570799352032f3` | 5 | 10 | 0.0000 | 0.0000 / 0.0000 | 1.0000 | 1.0000 | passed |

Artifacts:

- Patch summaries:
  - `reports/agentic_memory/locomo_conv_26_criteria_gate_v3_discriminative_profile_patch_summary_20260517.json`
  - `reports/agentic_memory/locomo_conv_30_criteria_gate_v3_discriminative_profile_patch_summary_20260517.json`
- Candidate heldout reports:
  - `reports/agentic_memory/locomo_conv_26_selfqa_criteria_gate_v3_discriminative_profile_candidate_topicsoft_profilehybrid_20260517.json`
  - `reports/agentic_memory/locomo_conv_30_selfqa_criteria_gate_v3_discriminative_profile_candidate_topicsoft_profilehybrid_20260517.json`
- Gate reports:
  - `reports/agentic_memory/locomo_conv_26_criteria_gate_v3_discriminative_profile_profilehybrid_strict_gate_20260517.json`
  - `reports/agentic_memory/locomo_conv_30_criteria_gate_v3_discriminative_profile_profilehybrid_strict_gate_20260517.json`

Interpretation:

- V3 is safer than V2: it did not increase candidate pool or route count.
- V3 did not improve heldout self-QA route hit or evidence recall, so it is not worth promoting or taking to LoCoMo QA.
- The useful signal remains profile-hybrid routing itself: on `locomo_conv_30`, base `profile_hybrid` improved expected-topic route hit from keyword `0.4000` to `0.6000` without changing evidence recall or candidate pool.

Decision:

- Do not promote V3 candidate views.
- Do not run LoCoMo QA for V3.
- Next step: evaluate base online-growth view with `profile_hybrid` routing on the existing 233QA LoCoMo subset, without extra criteria patches.

## 2026-05-17 LoCoMo 233QA V20 Profile-Hybrid Router

Scope:

- Purpose: test the topic-profile router itself on the existing LoCoMo 233QA subset, without criteria patch candidates.
- Baseline anchor: `reports/agentic_memory/locomo2_onlinegrowth_v2_17_v21db_topiclabeled_seed0_repeat_sample233_20260517.json`
- DB: `data/agentic_smoke/locomo2_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Input: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Sample limit: `2`
- QA count: `233`
- Answer model: Qwen3-235B from `tmp/config_agentic_codeai_memory_qwen_answer.yaml`

Command:

```bash
tmux new-session -d -s leaf_locomo_v20_profilehybrid_233 \
  'cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   bash /tmp/leaf_locomo_v20_profilehybrid_233.sh 2>&1 | tee reports/agentic_memory/locomo2_onlinegrowth_v2_20_v21db_profilehybrid_topiclabeled_seed0_sample233_20260517.log'
```

Important run parameters:

- `--retrieval-mode topic_soft_selective`
- `--topic-router profile_hybrid`
- `--topic-route-top-k 3`
- `--topic-soft-event-limit 1`
- `--topic-soft-policy text_temporal_suppressed_v0`
- `--topic-soft-policy-min-selected-overlap 1`
- `--topic-soft-policy-max-candidate-atoms 999`
- `--answer-style structured_context_topic_labeled`
- `--answer-evidence-mode auto`
- `--temporal-postprocess range`
- `--short-answer-postprocess precise`

Results:

| Run | Router | F1 | BLEU1 | Avg topic events | Avg candidate atoms | Policy applied | Suppressed events |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2.17 repeat | keyword | 0.5799 | 0.5112 | 0.6781 | 17.7124 | 158 | 74 |
| V2.20 | profile_hybrid | 0.5811 | 0.5136 | 0.4206 | 13.4464 | 98 | 45 |

By question type:

| Type | Count | V2.17 F1 | V2.20 F1 | Delta | V2.17 BLEU1 | V2.20 BLEU1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `multi_hop` | 43 | 0.4010 | 0.4138 | +0.0128 | 0.3090 | 0.3216 |
| `open_domain` | 13 | 0.3605 | 0.3605 | 0.0000 | 0.1646 | 0.1646 |
| `single_hop` | 114 | 0.5488 | 0.5489 | +0.0001 | 0.4890 | 0.4935 |
| `temporal` | 63 | 0.8038 | 0.7993 | -0.0045 | 0.7607 | 0.7529 |

Artifacts:

- Result: `reports/agentic_memory/locomo2_onlinegrowth_v2_20_v21db_profilehybrid_topiclabeled_seed0_sample233_20260517.json`
- Progress: `reports/agentic_memory/locomo2_onlinegrowth_v2_20_v21db_profilehybrid_topiclabeled_seed0_sample233_20260517.qa_progress.jsonl`
- Log: `reports/agentic_memory/locomo2_onlinegrowth_v2_20_v21db_profilehybrid_topiclabeled_seed0_sample233_20260517.log`

Interpretation:

- Profile-hybrid gives a small but real 233QA gain over V2.17: `+0.0012` F1 and `+0.0024` BLEU1.
- It is notably more conservative: avg topic-soft candidate atoms drops from `17.7124` to `13.4464`, and topic evidence is applied on fewer questions.
- The main gain is multi-hop (`+0.0128` F1), while temporal slightly regresses (`-0.0045` F1).
- This matches the self-QA signal: topic profiles improve route specificity, but temporal should stay mostly protected by the existing temporal suppression policy.

Decision:

- Keep `profile_hybrid` as a serious non-hard-coded retrieval integration point.
- It is not yet a large enough gain to call final.
- Next profile-router variants should target multi-hop/open-domain gains while avoiding additional temporal exposure.

## 2026-05-17 LoCoMo 233QA V21 Profile-Hybrid TopK4 Negative Result

Scope:

- Purpose: test whether a wider `profile_hybrid` route set helps multi-hop.
- Same setup as V20 except `--topic-route-top-k 4`.

Artifact:

- Result: `reports/agentic_memory/locomo2_onlinegrowth_v2_21_v21db_profilehybrid_topk4_topiclabeled_seed0_sample233_20260517.json`
- Progress: `reports/agentic_memory/locomo2_onlinegrowth_v2_21_v21db_profilehybrid_topk4_topiclabeled_seed0_sample233_20260517.qa_progress.jsonl`
- Log: `reports/agentic_memory/locomo2_onlinegrowth_v2_21_v21db_profilehybrid_topk4_topiclabeled_seed0_sample233_20260517.log`

Results:

| Run | Router | TopK | F1 | BLEU1 | Avg topic events | Avg candidate atoms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V2.17 | keyword | 3 | 0.5799 | 0.5112 | 0.6781 | 17.7124 |
| V2.20 | profile_hybrid | 3 | 0.5811 | 0.5136 | 0.4206 | 13.4464 |
| V2.21 | profile_hybrid | 4 | 0.5742 | 0.5074 | 0.4206 | 13.5193 |

By question type:

| Type | V20 F1 | V21 F1 | Delta |
| --- | ---: | ---: | ---: |
| `multi_hop` | 0.4138 | 0.4078 | -0.0060 |
| `open_domain` | 0.3605 | 0.3487 | -0.0118 |
| `single_hop` | 0.5489 | 0.5411 | -0.0078 |
| `temporal` | 0.7993 | 0.7942 | -0.0051 |

Interpretation:

- Widening profile-hybrid routing to top-k 4 is worse across all question types.
- Candidate atom count barely changes, so the regression appears to come from changed route/evidence selection and answer instability rather than a simple candidate-pool explosion.

Decision:

- Reject V21.
- Keep V20 profile-hybrid top-k 3 as the current best profile-router variant.

## 2026-05-17 LoCoMo Full V20 Profile-Hybrid Router

Scope:

- Purpose: test whether the V20 `profile_hybrid` gain on the 233QA subset transfers to full LoCoMo.
- Input: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Sample count: `10`
- QA count: `1540`
- Answer model: Qwen3-235B from the config file.
- Memory/modeling LLM for online-growth ingest: `gpt-5.4-mini` from the config file.
- Baseline anchor: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`

Command:

```bash
tmux new-session -d -s leaf_locomo10_v20_profilehybrid_full \
  'cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && \
   bash /tmp/leaf_locomo10_v20_profilehybrid_full.sh 2>&1 | tee reports/agentic_memory/locomo10_onlinegrowth_v2_20_profilehybrid_topiclabeled_full1540_20260517.log'
```

Stage 1: full online-growth ingest.

- DB: `data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Output: `reports/agentic_memory/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.json`
- Main ingest parameters:
  - `--sample-limit 10`
  - `--chunk-turns 40`
  - `--turns-threshold 80`
  - `--atoms-threshold 48`
  - `--trigger-policy all`
  - `--growth-strategy node_local`
  - `--min-cluster-atoms 3`
  - `--max-new-topics 2`
  - `--max-depth 4`
  - `--window-atom-limit 120`
  - `--secondary-max-assignments 0`

Ingest counts from SQLite after completion:

| Corpus | Events | Atoms | Active topic nodes |
| --- | ---: | ---: | ---: |
| `locomo_conv_26` | 419 | 568 | 21 |
| `locomo_conv_30` | 369 | 437 | 19 |
| `locomo_conv_41` | 663 | 874 | 27 |
| `locomo_conv_42` | 629 | 811 | 25 |
| `locomo_conv_43` | 680 | 895 | 27 |
| `locomo_conv_44` | 675 | 751 | 27 |
| `locomo_conv_47` | 689 | 888 | 27 |
| `locomo_conv_48` | 681 | 905 | 27 |
| `locomo_conv_49` | 509 | 646 | 23 |
| `locomo_conv_50` | 568 | 810 | 25 |

Total DB counts:

- Corpora: `10`
- Events: `5882`
- Atoms: `7585`
- Memory views: `79`
- Active views: `10`

Stage 2: full V20 profile-hybrid QA.

- Result: `reports/agentic_memory/locomo10_onlinegrowth_v2_20_profilehybrid_topiclabeled_full1540_20260517.json`
- Progress: `reports/agentic_memory/locomo10_onlinegrowth_v2_20_profilehybrid_topiclabeled_full1540_20260517.qa_progress.jsonl`
- Log: `reports/agentic_memory/locomo10_onlinegrowth_v2_20_profilehybrid_topiclabeled_full1540_20260517.log`
- Main QA parameters:
  - `--qa-per-sample 0`
  - `--ingest-mode reuse_only`
  - `--retrieval-mode topic_soft_selective`
  - `--topic-router profile_hybrid`
  - `--topic-route-top-k 3`
  - `--topic-soft-event-limit 1`
  - `--topic-soft-per-topic-atom-limit 16`
  - `--topic-soft-min-content-overlap 1`
  - `--topic-soft-policy text_temporal_suppressed_v0`
  - `--topic-soft-secondary-policy all`
  - `--answer-style structured_context_topic_labeled`
  - `--answer-evidence-mode auto`
  - `--temporal-postprocess range`
  - `--short-answer-postprocess precise`
  - `--qa-workers 8`

Overall results:

| Run | Retrieval | QA | F1 | BLEU1 | Avg search ms | Avg answer ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base | `baseline` | 1540 | 0.5593 | 0.5022 | 12295.73 | 1443.19 |
| V20 full | `topic_soft_selective` + `profile_hybrid` | 1540 | 0.5556 | 0.4988 | 13315.27 | 1621.20 |

By question type:

| Type | Count | LEAF-Base F1 | V20 F1 | Delta | LEAF-Base BLEU1 | V20 BLEU1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `multi_hop` | 282 | 0.3499 | 0.3573 | +0.0074 | 0.2661 | 0.2687 |
| `open_domain` | 96 | 0.3153 | 0.3206 | +0.0053 | 0.2597 | 0.2610 |
| `single_hop` | 841 | 0.6243 | 0.6186 | -0.0057 | 0.5725 | 0.5690 |
| `temporal` | 321 | 0.6459 | 0.6351 | -0.0108 | 0.5981 | 0.5882 |

By corpus:

| Corpus | Count | LEAF-Base F1 | V20 F1 | Delta | LEAF-Base BLEU1 | V20 BLEU1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `conv-26` | 152 | 0.5757 | 0.5683 | -0.0074 | 0.5078 | 0.5051 |
| `conv-30` | 81 | 0.5806 | 0.5713 | -0.0093 | 0.5171 | 0.5140 |
| `conv-41` | 152 | 0.6052 | 0.5906 | -0.0146 | 0.5424 | 0.5264 |
| `conv-42` | 199 | 0.5021 | 0.4998 | -0.0023 | 0.4504 | 0.4466 |
| `conv-43` | 178 | 0.6106 | 0.5795 | -0.0311 | 0.5529 | 0.5232 |
| `conv-44` | 123 | 0.5290 | 0.5330 | +0.0040 | 0.4819 | 0.4787 |
| `conv-47` | 150 | 0.5790 | 0.5912 | +0.0122 | 0.5279 | 0.5440 |
| `conv-48` | 191 | 0.5763 | 0.5814 | +0.0051 | 0.5233 | 0.5264 |
| `conv-49` | 156 | 0.5094 | 0.5156 | +0.0061 | 0.4437 | 0.4493 |
| `conv-50` | 158 | 0.5364 | 0.5374 | +0.0010 | 0.4826 | 0.4849 |

Topic-soft stats for V20 full:

- Avg topic-soft event count: `0.3779`
- Avg topic-soft candidate atoms: `12.6565`
- Avg raw candidate atoms: `54.6136`
- Avg filtered atoms: `41.9571`
- Policy applied count: `582`
- Suppressed events: `327`
- Search latency p95/max: `48320.73 ms` / `146800.55 ms`

Interpretation:

- V20 does not beat LEAF-Base on full LoCoMo: `-0.0037` F1 and `-0.0034` BLEU1.
- The 233QA gain did not transfer to the full benchmark. The subset result was dominated by `conv-26` and `conv-30`, while the full run shows mixed corpus behavior.
- The useful signal is still real but narrower: V20 improves `multi_hop` and `open_domain`, which are the intended topic-routing beneficiaries.
- The overall loss comes from `single_hop` and especially `temporal`; temporal should probably bypass topic-soft more aggressively or receive a different retrieval path.
- Latency is also worse than LEAF-Base, with long-tail searches above 90 seconds on some late `single_hop` questions.

Decision:

- Do not promote V20 profile-hybrid as the full LoCoMo default.
- Keep it as an ablation showing that topic profiles help hard reasoning categories but currently hurt common/temporal questions.
- Next experiment should preserve the multi-hop/open-domain benefit while routing single-hop/temporal questions through a safer non-LLM selector. Avoid benchmark question-type cheating; the selector must be inferred from query/retrieval features available at runtime.

## 2026-05-17 EvolveMem-style Failure-Driven Memory Proposal

Scope: implement a non-destructive evolution loop inspired by EvolveMem/SimpleMem: collect retrieval/QA failures, ask the memory LLM for guardable memory patches, then require self-QA gates before promotion.

Code path:

- Project: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory`
- Branch: `agentic-memory-evolution`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- DB: `data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`
- Memory LLM: `gpt-5.4-mini` at `https://codeai.ysaikeji.cn/v1`
- Answer LLM for LoCoMo reports used as input: `Qwen3-235B` at `http://localhost:8001/v1`

New scripts:

- `scripts/build_memory_failure_log.py`
- `scripts/propose_memory_patches_from_failures.py`
- `scripts/apply_memory_patch_proposal.py`
- Updated `scripts/build_selfqa_from_memory.py` with `--per-task-type-limit` and `--stream-output`.

Inputs:

- LoCoMo candidate report: `reports/agentic_memory/locomo_conv41_42_onlinegrowth_v2_27_profilehybrid_answer_topic_gate_topiclabeled_351qa_20260517.json`
- LEAF-Base baseline report: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- Self-QA stream tasks:
  - `reports/agentic_memory/selfqa/locomo_conv41_selfqa_criteria_v3_stream_evolvemem_20260517.jsonl`
  - `reports/agentic_memory/selfqa/locomo_conv42_selfqa_criteria_v3_stream_evolvemem_20260517.jsonl`
- Self-QA search evals:
  - `reports/agentic_memory/selfqa/locomo_conv41_selfqa_criteria_v3_stream_topicsoft_eval_20260517.json`
  - `reports/agentic_memory/selfqa/locomo_conv42_selfqa_criteria_v3_stream_topicsoft_eval_20260517.json`

Commands:

```bash
./scripts/with_tracenav_nlp.sh python scripts/build_memory_failure_log.py \
  --db data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --locomo-report reports/agentic_memory/locomo_conv41_42_onlinegrowth_v2_27_profilehybrid_answer_topic_gate_topiclabeled_351qa_20260517.json \
  --baseline-locomo-report reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json \
  --selfqa-report reports/agentic_memory/selfqa/locomo_conv41_selfqa_criteria_v3_stream_topicsoft_eval_20260517.json \
  --selfqa-report reports/agentic_memory/selfqa/locomo_conv42_selfqa_criteria_v3_stream_topicsoft_eval_20260517.json \
  --output reports/agentic_memory/failure_logs/locomo_conv41_42_v27_selfqa_v3_stream_failure_log_20260517.jsonl \
  --summary-output reports/agentic_memory/failure_logs/locomo_conv41_42_v27_selfqa_v3_stream_failure_summary_20260517.json \
  --min-f1 0.5 \
  --max-items 220

./scripts/with_tracenav_nlp.sh python scripts/propose_memory_patches_from_failures.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --failure-log reports/agentic_memory/failure_logs/locomo_conv41_42_v27_selfqa_v3_stream_failure_log_20260517.jsonl \
  --output reports/agentic_memory/patch_proposals/locomo_conv41_42_v27_selfqa_v3_evolvemem_patch_proposals_20260517.json \
  --max-failures 100 \
  --max-topics 30 \
  --max-tokens 4096
```

Failure log summary:

- Output: `reports/agentic_memory/failure_logs/locomo_conv41_42_v27_selfqa_v3_stream_failure_summary_20260517.json`
- Failure count: `179`
- Source types: `170` LoCoMo QA, `9` self-QA search
- By corpus: `locomo_conv_41=69`, `locomo_conv_42=110`
- Important failure modes:
  - `answer_wrong=154`
  - `regressed_vs_baseline=39`
  - `topic_missing_or_unused=28`
  - `temporal_conflict_or_missing_evidence=25`
  - `topic_noise=20`
  - `topic_route_miss=7`
  - `gold_topic_path_miss=9`
  - `selfqa_event_miss=3`, `selfqa_atom_miss=3`

Self-QA generation/eval observations:

- First validated smoke generated only single-fact questions, so `build_selfqa_from_memory.py` was updated with accepted-task type caps and streaming output.
- Stream self-QA accepted:
  - `locomo_conv_41`: 4 tasks, `2` single_fact + `2` multi_hop, `0` temporal.
  - `locomo_conv_42`: 6 tasks, `2` single_fact + `2` multi_hop + `2` temporal.
- Self-QA route diagnostics:
  - `locomo_conv_41`: criteria expected topic route hit `0.0`; retrieval topic hit `1.0`; mean event recall `0.875`; mean atom recall `0.875`.
  - `locomo_conv_42`: criteria expected topic route hit `0.5`; retrieval topic hit `1.0`; mean event recall `0.75`; mean atom recall `0.75`.
- Interpretation: baseline retrieval often finds evidence whose assigned topic is correct, but `profile_hybrid` topic router misses those expected topics. The next optimization should focus on topic route/profile alignment, not just adding more topic evidence.

Proposal outputs:

- LoCoMo-only proposal: `reports/agentic_memory/patch_proposals/locomo_conv41_42_v27_evolvemem_patch_proposals_20260517.json`, `4` patches.
- LoCoMo+self-QA proposal: `reports/agentic_memory/patch_proposals/locomo_conv41_42_v27_selfqa_v3_evolvemem_patch_proposals_20260517.json`, `9` patches.
- The joint proposal added self-QA-aware ideas such as `topic_route_keywords` and `selfqa_criteria_policy`, not just broad topic splits.

Guard test for low-risk metadata patch:

- Applied proposal patches to a candidate view only:
  - `p07_route_keywords_coconut_icecream_bridge`
  - `p09_answer_exposure_shadow_boost_profile_specificity`
- Apply output: `reports/agentic_memory/patch_proposals/locomo_conv41_42_v27_selfqa_v3_metadata_patch_apply_20260517.json`
- Candidate view: `view_2cab6a6bbb897dff58`
- Baseline self-QA eval: `reports/agentic_memory/selfqa/locomo_conv42_selfqa_criteria_v3_stream_topicsoft_eval_20260517.json`
- Candidate self-QA eval: `reports/agentic_memory/selfqa/locomo_conv42_selfqa_criteria_v3_stream_topicsoft_eval_metadata_patch_p07_p09_20260517.json`
- Gate output: `reports/agentic_memory/patch_proposals/locomo_conv42_metadata_patch_p07_p09_selfqa_gate_20260517.json`
- Gate result: `passed=false`, `promoted=false`
- Gate failures:
  - route hit delta: `-0.5`
  - topic-soft candidate atom ratio: `1.40506` > threshold `1.25`
  - route count ratio: `1.333333` > threshold `1.25`

Decision:

- The EvolveMem-style proposal loop is now implemented end to end, but proposals are not automatically trusted.
- The first low-risk route/profile patch was correctly rejected by self-QA gate. This validates the guard mechanism and shows why proposal generation must be followed by shadow evaluation.
- Next work should implement a real shadow topic split evaluator for proposed split patches (`p02`-`p06`) and/or improve topic-router scoring so route/profile changes do not inflate candidate pools.

## 2026-05-17 Memory Overlay Sidecars Beyond Topic Tree

Scope:

- Project: `LEAF_agentic_memory`
- Branch: `agentic-memory-evolution`
- Purpose: implement non-destructive evolved sidecars beyond topic-tree: retrieval policy, entity/profile overlay, temporal overlay, evidence utility, atom facets, context assembly policy, and self-QA/failure-log policy proposal.
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
  - answer LLM: `Qwen3-235B` at `http://localhost:8001/v1`
  - memory LLM: `gpt-5.4-mini` at `https://codeai.ysaikeji.cn/v1`
  - embedding: `bge-m3` at `http://127.0.0.1:8080/v1`
- DB: `data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3`

Code changes:

- Added `src/leaf/memory_overlay.py`.
  - Builds atom/event utility and facets.
  - Builds `entity_profile_overlay` and `temporal_overlay`.
  - Defines default retrieval/context policy overlay.
- Added `scripts/build_memory_overlay_view.py`.
  - Clones active memory view into candidate view and stores sidecars in view metadata.
  - Does not change base events or atoms.
- Added `scripts/propose_overlay_policy_from_failures.py`.
  - Reads LoCoMo+self-QA failure log and derives runtime-feature-only retrieval policy patch.
  - Uses failure modes and query-visible features, not benchmark `question_type` as runtime routing.
- Added `scripts/apply_overlay_policy_patch.py`.
  - Applies policy patch to a cloned memory view metadata only.
- Updated `src/leaf/topic_soft.py`.
  - Added `overlay_expand_events`.
  - Added conservative primary-evidence suppression: if baseline evidence is strong, overlay is not added.
  - Added entity/profile sidecar expansion using entity/alias hit + event term/facet/utility score.
  - Added temporal sidecar expansion and optional previous/next timeline neighbor events.
- Updated `scripts/eval_locomo.py` and `scripts/eval_locomo_qa_only_parallel.py`.
  - Added `retrieval-mode=overlay_selective`.
  - Added overlay evidence to structured context as supplemental evidence only.
  - Added overlay metrics: event count, candidate count, suppressed count, source counts.

Mechanism status:

- Retrieval strategy evolution: implemented as sidecar policy proposal/application. Current policy patch:
  - `reports/agentic_memory/patch_proposals/locomo_conv41_42_overlay_policy_from_failure_selfqa_v4_20260517.json`
  - Runtime-visible features only; no LoCoMo question-type routing at inference.
- Entity/profile sidecar: implemented in v3/v4 overlay views.
  - Query expansion/rerank only; atoms/events unchanged.
  - v4 requires entity/alias hit and event relevance score, after v1/v3 were too broad.
- Temporal memory overlay: implemented in metadata and retrieval.
  - Includes date/month index and previous/next event links.
  - Neighbor expansion was added after v4 guard; not separately scored yet.
- Evidence utility/visibility score: implemented per atom/event as specificity/profile/temporal/relation/answerability.
  - Used in overlay scoring, not learned yet.
- Atom facets: implemented as multi-label sidecar facets.
  - Current facets include `temporal`, `profile`, `relationship`, `plan`, `media_hobby`, `place_travel`, `activity`, `general_fact`.
- Context assembly policy: implemented as supplemental overlay context and evidence usage policy.
  - Primary LEAF evidence is instructed to dominate; overlay labels are not standalone evidence.
- Self-QA/criteria generator as training signal: partially implemented.
  - Existing failure log mixes `170` LoCoMo failures and `9` self-QA search failures.
  - v4 policy proposal is derived from that log.
  - Still missing: automatic guard-gated promotion loop from self-QA criteria to policy/tree patch.

Overlay view artifacts:

- v3 sidecar overlay build:

```bash
./scripts/with_tracenav_nlp.sh python scripts/build_memory_overlay_view.py \
  --db data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --corpus-id locomo_conv_41 --corpus-id locomo_conv_42 \
  --output reports/agentic_memory/overlay/locomo_conv41_42_evolved_memory_overlay_v3_apply_20260517.json \
  --name evolved-memory-overlay-v3 \
  --max-topic-profile-terms 32 \
  --profile-term-target overlay_profile_terms
```

- v3 output:
  - `locomo_conv_41`: `view_fc30cb5708b6b870d5`
    - event overlays `663`, atom overlays `874`, entity profiles `224`, temporal events `663`.
  - `locomo_conv_42`: `view_7409e6a3027d6acaa2`
    - event overlays `629`, atom overlays `811`, entity profiles `248`, temporal events `629`.
- v4 policy patch output:
  - proposal: `reports/agentic_memory/patch_proposals/locomo_conv41_42_overlay_policy_from_failure_selfqa_v4_20260517.json`
  - conv41 candidate: `view_67c4f7cb3cf9359d7c`
  - conv42 candidate: `view_e639275ce5b6d8984e`

Conv41 guard:

```bash
./scripts/with_tracenav_nlp.sh python scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_online_growth_v2_1_cleanroute_nodelocal_evidenceonly_20260517.sqlite3 \
  --input tmp/locomo_conv41_noadv.json \
  --output reports/agentic_memory/locomo_conv41_overlay_policy_v4_qa_guard_20260517.json \
  --ingest-mode reuse_only \
  --retrieval-mode overlay_selective \
  --topic-view-id view_67c4f7cb3cf9359d7c \
  --overlay-event-limit 4 \
  --topic-soft-use-stemmed-content-tokens \
  --answer-style structured_context_topic_labeled \
  --answer-evidence-mode merged \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  --qa-workers 8
```

Conv41 result table:

| Run | Result file | N | F1 | BLEU1 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| active control | `reports/agentic_memory/locomo_conv41_activeview_event4_stem_qa_control_20260517.json` | 152 | 0.5848 | 0.5217 | matched active view control |
| strict topic split | `reports/agentic_memory/locomo_conv41_topic_split_shadow_strict_qa_guard_20260517.json` | 152 | 0.5956 | 0.5284 | helps open/single, hurts multi |
| overlay v1 broad | `reports/agentic_memory/locomo_conv41_evolved_memory_overlay_v1_qa_guard_20260517.json` | 152 | 0.5965 | 0.5328 | best conv41 guard, but broad profile terms made routing/candidates fat |
| overlay v4 conservative | `reports/agentic_memory/locomo_conv41_overlay_policy_v4_qa_guard_20260517.json` | 152 | 0.5933 | 0.5308 | suppresses overlay for 143/152 questions; avg overlay events 0.2368 |

Conv41 v4 by category:

- multi_hop: `31` questions, F1 `0.4398`, BLEU1 `0.3362`.
- open_domain: `8` questions, F1 `0.3709`, BLEU1 `0.3553`.
- single_hop: `86` questions, F1 `0.6522`, BLEU1 `0.6006`.
- temporal: `27` questions, F1 `0.6477`, BLEU1 `0.5838`.

Interpretation:

- v4 is more conservative and more product-realistic than v1.
- v4 improves over matched active control by `+0.0085` F1 and `+0.0091` BLEU1 on conv41.
- v4 is slightly below broad v1 (`-0.0032` F1, `-0.0020` BLEU1), but avoids v1's always-on topic/profile pollution.
- Current sidecar gains are modest. Biggest remaining gap is not sidecar creation but policy learning/promotion and better missing-evidence triggering.

Evolution frequency:

- Default config in `src/leaf/config.py`:
  - `online_evolution_turns_threshold=50`
  - `online_evolution_atoms_threshold=40`
  - `online_evolution_trigger_policy=any`
- This LoCoMo online-growth DB used:
  - chunk turns `40`
  - turns threshold `80`
  - atoms threshold `48`
  - trigger policy `all`
  - min cluster atoms `3`
  - max new topics `2`
  - max depth `4`
  - growth strategy `node_local`
  - window atom limit was set by the run command/config; active trigger metadata shows recent atom counts around `101-105`.
- Active pending state after final promoted online growth:
  - `locomo_conv_41`: pending `23` turns / `28` atoms after last evolution.
  - `locomo_conv_42`: pending `69` turns / `95` atoms after last evolution. Under `all`, it has enough atoms but not enough turns, so it did not trigger another online growth.
- `leaf_evolution_runs` count after this sidecar work: `84`.

Per-corpus run counts from `leaf_evolution_runs`:

| corpus | total runs | promoted online growth | split candidates | overlay candidates | overlay policy candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| locomo_conv_26 | 5 | 5 | 0 | 0 | 0 |
| locomo_conv_30 | 4 | 4 | 0 | 0 | 0 |
| locomo_conv_41 | 16 | 8 | 2 | 3 | 3 |
| locomo_conv_42 | 14 | 7 | 2 | 3 | 2 |
| locomo_conv_43 | 8 | 8 | 0 | 0 | 0 |
| locomo_conv_44 | 8 | 8 | 0 | 0 | 0 |
| locomo_conv_47 | 8 | 8 | 0 | 0 | 0 |
| locomo_conv_48 | 8 | 8 | 0 | 0 | 0 |
| locomo_conv_49 | 6 | 6 | 0 | 0 | 0 |
| locomo_conv_50 | 7 | 7 | 0 | 0 | 0 |

Caveats:

- v4 conv42 candidate exists but has not been LoCoMo-guarded yet.
- Temporal neighbor expansion was implemented after the conv41 v4 guard, so the exact v4 result above does not include temporal neighbor gains or regressions.
- Sidecar utility/facet scoring is heuristic/traditional NLP based; not yet learned from self-QA.
- No candidate was promoted active in this section.

## 2026-05-17 Old-DB Sidecar Overlay vs LEAF-Base Anchor

Purpose:

- Compare against the actual LEAF-Base anchor on the same old LEAF DB, without topic tree or online-evolved DB mismatch.
- Keep answer LLM fixed to local Qwen3-235B via `tmp/config_agentic_codeai_memory_qwen_answer.yaml`.
- Retrieval/search uses no LLM at query time; overlay evidence comes from offline sidecar metadata over existing LEAF events/atoms.

Input and DB:

- Input QA: `tmp/locomo_conv41_noadv.json`
- Old LEAF-Base DB: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Environment wrapper: `./scripts/with_tracenav_nlp.sh`

Sidecar views:

- v5 sidecar output: `reports/agentic_memory/overlay/locomo_conv41_leafbase_overlay_sidecar_v5_apply_20260517.json`
  - view id: `view_1acd08f0a99172d17f`
- v6stable sidecar output: `reports/agentic_memory/overlay/locomo_conv41_leafbase_overlay_sidecar_v6stable_apply_20260517.json`
  - view id: `view_a0cab5143b0c10678d`
  - metrics: event overlays `663`, atom overlays `910`, entity profiles `271`, temporal events `663`, avg answerability `0.7965`

Full conv41 commands:

```bash
./scripts/with_tracenav_nlp.sh python scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input tmp/locomo_conv41_noadv.json \
  --output reports/agentic_memory/locomo_conv41_leafbase_overlay_v6stable_temporal_geo_nt12_qa_guard_20260517.json \
  --ingest-mode reuse_only \
  --snapshot-limit 8 \
  --raw-span-limit 8 \
  --non-temporal-raw-span-limit 12 \
  --retrieval-mode overlay_selective \
  --topic-view-id view_a0cab5143b0c10678d \
  --overlay-event-limit 4 \
  --answer-style structured_context_topic_labeled \
  --answer-evidence-mode merged \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  --qa-workers 8
```

Deterministic repostprocess command:

```bash
./scripts/with_tracenav_nlp.sh python scripts/repostprocess_locomo_results.py \
  --input reports/agentic_memory/locomo_conv41_leafbase_overlay_v6stable_temporal_geo_nt12_qa_guard_20260517.json \
  --output reports/agentic_memory/locomo_conv41_leafbase_overlay_v6stable_temporal_guard_repost_scriptcheck_nt12_qa_guard_20260517.json \
  --note 'Script check for deterministic temporal guard repostprocess; no retrieval or LLM calls.'
```

Result table:

| Run | Result file | N | F1 | BLEU1 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| LEAF-Base current old DB | `reports/agentic_memory/locomo_conv41_olddb_baseline_nt12_qa_guard_20260517.json` | 152 | 0.6113 | 0.5470 | no topic tree, no overlay, same Qwen answer |
| Overlay v5 sidecar | `reports/agentic_memory/locomo_conv41_leafbase_overlay_sidecar_v5_nt12_qa_guard_20260517.json` | 152 | 0.6406 | 0.5698 | strongest raw full run before temporal guard |
| v6stable raw | `reports/agentic_memory/locomo_conv41_leafbase_overlay_v6stable_temporal_geo_nt12_qa_guard_20260517.json` | 152 | 0.6349 | 0.5732 | temporal/geo fixes; raw run before deterministic repostprocess |
| v6stable + temporal guard repost | `reports/agentic_memory/locomo_conv41_leafbase_overlay_v6stable_temporal_guard_repost_nt12_qa_guard_20260517.json` | 152 | 0.6511 | 0.5869 | best current conv41; no extra retrieval/LLM calls |
| v6stable + temporal guard script check | `reports/agentic_memory/locomo_conv41_leafbase_overlay_v6stable_temporal_guard_repost_scriptcheck_nt12_qa_guard_20260517.json` | 152 | 0.6511 | 0.5869 | reproduces repostprocess via script |

Best current result by category:

- multi_hop: `31`, F1 `0.5671`, BLEU1 `0.4772`
- open_domain: `8`, F1 `0.2852`, BLEU1 `0.2788`
- single_hop: `86`, F1 `0.6861`, BLEU1 `0.6222`
- temporal: `27`, F1 `0.7447`, BLEU1 `0.6917`

Delta vs LEAF-Base current old DB:

- Overall: `+0.0398` F1, `+0.0399` BLEU1.
- multi_hop: `+0.0686` F1, `+0.0686` BLEU1.
- single_hop: `+0.0326` F1, `+0.0259` BLEU1.
- temporal: `+0.0692` F1, `+0.0772` BLEU1.
- open_domain: `-0.0915` F1, `-0.0475` BLEU1. This remains the main weak slice.

Mechanisms included in v6stable + temporal guard:

- Offline sidecar event/atom utility and facets over unchanged LEAF events/atoms.
- Entity/profile sidecar used as retrieval expansion signal.
- Event lexical overlay route using sidecar terms, facets, geo terms, and answerability.
- Local same-session neighbor route for Q/A follow-up evidence.
- Temporal relative parsing for `about a year ago`, seasons such as `last summer`, and relative week/weekend/weekday expressions.
- Evidence-aware temporal answer guard:
  - avoids replacing a supported explicit answer with an unrelated relative cue,
  - allows replacement when current answer is unsupported and the candidate grounding has speaker/focus support,
  - ignores person names as distinctive content terms and checks speaker separately.
- Deterministic geo list extraction can use sidecar `geo_terms`, fixing the Spain/England case.

Repostprocess changes:

- Changed answer text on 7 rows: `qa-14`, `qa-21`, `qa-35`, `qa-39`, `qa-55`, `qa-57`, `qa-60`.
- Six rows changed metrics; `qa-55` changed punctuation only.
- Main temporal fixes:
  - grandmother death: `Week before 9 January 2023` -> `Week before 6 March 2023`
  - Pacific Northwest road trip: `16 December 2022` -> `April 2022`
  - gym join: `5 May 2023` -> `9-15 June 2023`
  - car accident: `Friday before 10 April 2023` -> `2 July 2023`
  - John 5K: `Weekend before 7 April 2023` -> `9 August 2023`
  - Max camping: `Weekend before 12 June 2023` -> `Summer 2022`

Caveats:

- The best result includes deterministic repostprocessing from the completed v6stable raw run; it does not rerun retrieval or answer LLM.
- v6stable still hurts open_domain vs LEAF-Base. Next work should evolve profile/inference sidecar rather than broad suppress rules.
- This is conv41 only. Full LoCoMo should be run after applying the same view-building and repostprocess path to all target corpora.

## 2026-05-17 Full LoCoMo LEAF-Base Overlay v6stable Launch

Purpose:

- Run the conv41-best overlay sidecar path on full LoCoMo without re-ingesting memory.
- Compare against the full LEAF-Base anchor on the same old DB and same local Qwen answer model.
- Keep retrieval/search LLM-free at query time; all overlay information is offline sidecar metadata over unchanged LEAF events/atoms.

Baseline anchor:

- Result file: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- Input QA: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- DB: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- Count: `10` samples, `1540` QA
- LEAF-Base full metric: F1 `0.5593`, BLEU1 `0.5022`
- Category counts: multi_hop `282`, open_domain `96`, single_hop `841`, temporal `321`

Old DB corpus inventory:

- `locomo_conv_26`: events `419`, atoms `560`
- `locomo_conv_30`: events `369`, atoms `470`
- `locomo_conv_41`: events `663`, atoms `910`
- `locomo_conv_42`: events `629`, atoms `842`
- `locomo_conv_43`: events `680`, atoms `932`
- `locomo_conv_44`: events `675`, atoms `902`
- `locomo_conv_47`: events `689`, atoms `910`
- `locomo_conv_48`: events `681`, atoms `902`
- `locomo_conv_49`: events `509`, atoms `673`
- `locomo_conv_50`: events `568`, atoms `822`
- Total: events `5882`, atoms `7923`

Sidecar build:

```bash
./scripts/with_tracenav_nlp.sh python scripts/build_memory_overlay_sidecar_view.py \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --name locomo10_leafbase_overlay_sidecar_v6stable_temporal_geo \
  --output reports/agentic_memory/overlay/locomo10_leafbase_overlay_sidecar_v6stable_apply_20260517.json \
  --corpus-id locomo_conv_26 \
  --corpus-id locomo_conv_30 \
  --corpus-id locomo_conv_41 \
  --corpus-id locomo_conv_42 \
  --corpus-id locomo_conv_43 \
  --corpus-id locomo_conv_44 \
  --corpus-id locomo_conv_47 \
  --corpus-id locomo_conv_48 \
  --corpus-id locomo_conv_49 \
  --corpus-id locomo_conv_50
```

Sidecar outputs:

- Sidecar apply JSON: `reports/agentic_memory/overlay/locomo10_leafbase_overlay_sidecar_v6stable_apply_20260517.json`
- Topic view map: `tmp/locomo10_leafbase_overlay_v6stable_view_map_20260517.json`
- View ids:
  - `locomo_conv_26`: `view_44e87c8d4512c9f918`
  - `locomo_conv_30`: `view_ad49193e06764b8686`
  - `locomo_conv_41`: `view_17453272c64a9ea566`
  - `locomo_conv_42`: `view_c183ad4de6110da975`
  - `locomo_conv_43`: `view_a2fa9354a79ee064af`
  - `locomo_conv_44`: `view_c14917fa80dab17613`
  - `locomo_conv_47`: `view_0fc0f640b0f8856ce8`
  - `locomo_conv_48`: `view_066beb9cd462560ec4`
  - `locomo_conv_49`: `view_9f3683c0b39e5329f9`
  - `locomo_conv_50`: `view_4d04ab15fea87eb3b7`

Full QA launch:

```bash
tmux new-session -d -s leaf_full_locomo_overlay_v6stable_20260517 \
  '/tmp/leaf_full_locomo_overlay_v6stable_20260517.sh 2>&1 | tee logs/leaf_full_locomo_overlay_v6stable_20260517.log'
```

Script content:

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
./scripts/with_tracenav_nlp.sh python scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json \
  --output reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_geo_nt8_qa_guard_full_20260517.json \
  --ingest-mode reuse_only \
  --snapshot-limit 8 \
  --raw-span-limit 8 \
  --non-temporal-raw-span-limit 12 \
  --retrieval-mode overlay_selective \
  --topic-view-map tmp/locomo10_leafbase_overlay_v6stable_view_map_20260517.json \
  --overlay-event-limit 4 \
  --answer-style structured_context_topic_labeled \
  --answer-evidence-mode merged \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  --qa-workers 8
```

Run artifacts:

- tmux session: `leaf_full_locomo_overlay_v6stable_20260517`
- Log: `logs/leaf_full_locomo_overlay_v6stable_20260517.log`
- Raw/full output: `reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_geo_nt8_qa_guard_full_20260517.json`
- Note: `nt8` in this filename means `qa-workers=8`; the run label should be read as `FullLoCoMo-LEAFBaseOverlay-v6stable-qaw8`.

Initial status:

- Launched at local time `2026-05-17 23:15`.
- Initial sanity check: output JSON was being written and had completed at least `65 / 1540` rows.
- The partial metric is not a final result and should not be used for claims.

Completion:

- Full raw output: `reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_geo_nt8_qa_guard_full_20260517.json`
- Full repostprocessed output: `reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_guard_repost_nt8_full_20260518.json`
- Repostprocess command:

```bash
./scripts/with_tracenav_nlp.sh python scripts/repostprocess_locomo_results.py \
  --input reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_geo_nt8_qa_guard_full_20260517.json \
  --output reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_guard_repost_nt8_full_20260518.json \
  --note 'Full LoCoMo deterministic repostprocess for v6stable overlay sidecar; no retrieval or LLM calls.'
```

Full result:

| Run | N | F1 | BLEU1 | Notes |
| --- | ---: | ---: | ---: | --- |
| LEAF-Base full anchor | 1540 | 0.5593 | 0.5022 | `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json` |
| Overlay v6stable no gate full | 1540 | 0.5595 | 0.5023 | `reports/agentic_memory/locomo10_leafbase_overlay_v6stable_temporal_guard_repost_nt8_full_20260518.json` |

By-category result for overlay v6stable no gate:

- multi_hop: `282`, F1 `0.3667`, BLEU1 `0.2781`
- open_domain: `96`, F1 `0.2552`, BLEU1 `0.2038`
- single_hop: `841`, F1 `0.6235`, BLEU1 `0.5696`
- temporal: `321`, F1 `0.6520`, BLEU1 `0.6123`

Caveats:

- Full overlay v6stable is effectively tied with LEAF-Base, not a meaningful full-LoCoMo gain.
- Repostprocess changed `0` answers on the full run.
- Full diff showed positive slices on some corpora, especially `conv41`, but negative slices on `conv43`, `conv47`, `conv48`, `conv49`, and `conv50`.

## 2026-05-18 Challenge Corpus Check And Runtime Gate Ablation

Purpose:

- Avoid relying on the repeatedly tuned `conv41`.
- Test on a harder, non-conv41 challenge set: `conv30`, `conv43`, `conv47`, `conv50`.
- Treat `pre_gate_v2_no_open` as a failed/diagnostic ablation unless it clearly beats current-code baseline.

Challenge input:

- Input: `tmp/locomo_challenge_conv30_43_47_50_noadv.json`
- Source: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Samples: `4`
- QA: `567`

Code change:

- Added optional overlay runtime gate `--overlay-runtime-policy pre_gate_v2_no_open`.
- Files touched for this gate:
  - `src/leaf/topic_soft.py`
  - `scripts/eval_locomo.py`
  - `scripts/eval_locomo_qa_only_parallel.py`
- Default is `none`, so old runs are not affected unless the flag is passed.
- Verified with:

```bash
./scripts/with_tracenav_nlp.sh python -m py_compile \
  src/leaf/topic_soft.py \
  scripts/eval_locomo.py \
  scripts/eval_locomo_qa_only_parallel.py
```

Challenge commands:

```bash
tmux new-session -d -s leaf_locomo_challenge_overlay_gate_v2_20260518 \
  '/tmp/leaf_locomo_challenge_overlay_gate_v2_20260518.sh 2>&1 | tee logs/leaf_locomo_challenge_overlay_gate_v2_20260518.log'
```

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
./scripts/with_tracenav_nlp.sh python scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input tmp/locomo_challenge_conv30_43_47_50_noadv.json \
  --output reports/agentic_memory/locomo_challenge_conv30_43_47_50_overlay_gate_v2_no_open_qaw8_20260518.json \
  --ingest-mode reuse_only \
  --snapshot-limit 8 \
  --raw-span-limit 8 \
  --non-temporal-raw-span-limit 12 \
  --retrieval-mode overlay_selective \
  --topic-view-map tmp/locomo10_leafbase_overlay_v6stable_view_map_20260517.json \
  --overlay-event-limit 4 \
  --overlay-runtime-policy pre_gate_v2_no_open \
  --answer-style structured_context_topic_labeled \
  --answer-evidence-mode merged \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  --qa-workers 8
```

```bash
tmux new-session -d -s leaf_locomo_challenge_baseline_current_20260518 \
  '/tmp/leaf_locomo_challenge_baseline_current_20260518.sh 2>&1 | tee logs/leaf_locomo_challenge_baseline_current_20260518.log'
```

```bash
cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory
./scripts/with_tracenav_nlp.sh python scripts/eval_locomo_qa_only_parallel.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3 \
  --input tmp/locomo_challenge_conv30_43_47_50_noadv.json \
  --output reports/agentic_memory/locomo_challenge_conv30_43_47_50_leafbase_current_qaw8_20260518.json \
  --ingest-mode reuse_only \
  --snapshot-limit 8 \
  --raw-span-limit 8 \
  --non-temporal-raw-span-limit 12 \
  --retrieval-mode baseline \
  --answer-style structured_context_topic_labeled \
  --answer-evidence-mode merged \
  --temporal-postprocess range \
  --short-answer-postprocess precise \
  --qa-workers 8
```

Challenge results:

| Run | N | F1 | BLEU1 | Notes |
| --- | ---: | ---: | ---: | --- |
| Old full LEAF-Base slice | 567 | 0.5773 | 0.5216 | slice from `locomo10_leafbase_evalv16...full_20260516.json`; not same-run current baseline |
| Old full overlay no-gate slice | 567 | 0.5602 | 0.5070 | slice from full overlay run |
| Current-code LEAF-Base challenge | 567 | 0.5520 | 0.4995 | same input/code timing as gate run |
| Current-code overlay gate v2 challenge | 567 | 0.5551 | 0.5020 | +0.0031 F1 / +0.0025 BLEU1 vs current-code challenge baseline |

By-category current-code overlay gate v2:

- multi_hop: `94`, F1 `0.3807`, BLEU1 `0.2894`
- open_domain: `34`, F1 `0.2635`, BLEU1 `0.2230`
- single_hop: `321`, F1 `0.6039`, BLEU1 `0.5541`
- temporal: `118`, F1 `0.6456`, BLEU1 `0.6102`

Conclusion:

- Stop after these runs per user request.
- Do not promote `pre_gate_v2_no_open` to full LoCoMo; the same-run gain is too small to count as a meaningful method improvement.
- No related `tmux` sessions were running after completion.

## 2026-05-18 GVD Current Topic-Soft Full100

Purpose:

- Run the current GVD-compatible evolved/topic-soft path on English GVD full100.
- Note: `scripts/eval_gvd.py` currently supports `baseline`, `topic_soft`, `topic_soft_gated`, and `topic_soft_selective`; it does not yet expose the LoCoMo `overlay_selective` sidecar path.

Inputs:

- Code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Answer LLM: `Qwen3-235B` at `http://localhost:8001/v1`
- Memory LLM in config: `gpt-5.4-mini` at `https://codeai.ysaikeji.cn/v1`
- Benchmark memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json`
- Benchmark questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl`
- Gold file: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json`
- DB: `data/agentic_smoke/gvd_agentic_full100_topicsoft_20260515.sqlite3`

Counts:

- Persona count: `15`
- Question count: `100`
- Reused corpora: `15`
- Newly ingested corpora: `0`
- Turn count total: `1132`

Command:

```bash
tmux new-session -d -s leaf_gvd_current_20260518 \
  'cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && { echo "[start] $(date -Is) eval_gvd current topic-soft full100"; ./scripts/with_tracenav_nlp.sh python scripts/eval_gvd.py --config tmp/config_agentic_codeai_memory_qwen_answer.yaml --db data/agentic_smoke/gvd_agentic_full100_topicsoft_20260515.sqlite3 --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json --questions /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl --output reports/agentic_memory/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_20260518.json --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 --answer-view-mode extractive --answer-style structured_context --answer-revision none --unknown-recovery none --retrieval-mode topic_soft --topic-router keyword --topic-route-top-k 3 --topic-soft-event-limit 2 --topic-soft-per-topic-atom-limit 16 --topic-soft-fallback baseline_on_unknown; echo "[eval_done] $(date -Is)"; ./scripts/with_tracenav_nlp.sh python /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/scripts/eval_gvd_gold_metrics.py --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json --report reports/agentic_memory/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_20260518.json --output reports/agentic_memory/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_gold_eval_20260518.json; echo "[done] $(date -Is)"; } 2>&1 | tee logs/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_20260518.log'
```

Outputs:

- Report: `reports/agentic_memory/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_20260518.json`
- Gold metrics: `reports/agentic_memory/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_gold_eval_20260518.json`
- Log: `logs/gvd_full100_current_topicsoft_keyword_limit2_fallback_structctx_20260518.log`

Metrics:

| Run | Q | F1 | BLEU1 | strict EM | accepted | avg search ms | avg answer tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-15 baseline structctx | 100 | 0.7379 | 0.6760 | 0.37 | 0.42 | 103.51 | 1041.60 |
| 2026-05-15 topic-soft keyword limit2 fallback | 100 | 0.7525 | 0.6904 | 0.37 | 0.42 | 94.06 | 1191.16 |
| 2026-05-18 current topic-soft keyword limit2 fallback | 100 | 0.7534 | 0.6982 | 0.39 | 0.43 | 102.84 | 1199.58 |

Current run summary:

- `retrieval_mode`: `topic_soft`
- `topic_soft_fallback`: `baseline_on_unknown`
- `avg_topic_soft_event_count`: `1.98`
- `topic_soft_fallback_used_count`: `0`
- `answer_llm_call_count_total`: `101`
- `avg_elapsed_ms`: `358.45`

Caveats:

- This is a current-code GVD-compatible topic-soft run, not the latest LoCoMo `overlay_selective` sidecar method.
- The run reused the existing full100 topic-soft DB, so this is QA/search evaluation over existing ingested/evolved memory views, not fresh ingestion.

## 2026-05-18 GVD LEAF-Base GPT-5.4-Mini Fresh Memory

Purpose:

- Add a strict LEAF-Base baseline with fresh memory construction using the current memory-side model.
- Keep QA answering fixed to local `Qwen3-235B`; retrieval mode remains `baseline` with no topic tree and no evolution.

Inputs:

- Code path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory`
- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Memory construction LLM: CodeAI `gpt-5.4-mini` at `https://codeai.ysaikeji.cn/v1`
- Answer LLM: local `Qwen3-235B` at `http://localhost:8001/v1`
- Benchmark memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json`
- Benchmark questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl`
- Gold file: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json`
- Fresh DB: `data/agentic_smoke/gvd_leafbase_gpt54mini_memory_full100_20260518.sqlite3`

Command:

```bash
tmux new-session -d -s leaf_gvd_leafbase_gpt54mini_20260518 \
  'cd /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory && { echo "[start] $(date -Is) LEAF-Base-GPT54MiniMemory fresh GVD full100"; ./scripts/with_tracenav_nlp.sh python scripts/eval_gvd.py --config tmp/config_agentic_codeai_memory_qwen_answer.yaml --db data/agentic_smoke/gvd_leafbase_gpt54mini_memory_full100_20260518.sqlite3 --memory-bank /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/memory_bank_en.json --questions /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/gvd/probing_questions_en.jsonl --output reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.json --ingest-mode migration --snapshot-limit 6 --raw-span-limit 8 --answer-view-mode extractive --answer-style structured_context --answer-revision none --unknown-recovery none; echo "[eval_done] $(date -Is)"; ./scripts/with_tracenav_nlp.sh python /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/scripts/eval_gvd_gold_metrics.py --gold /vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_full100_qwen235b_evidence.json --report reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.json --output reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_gold_eval_20260518.json; echo "[done] $(date -Is)"; } 2>&1 | tee logs/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.log'
```

Outputs:

- Fresh DB: `data/agentic_smoke/gvd_leafbase_gpt54mini_memory_full100_20260518.sqlite3`
- Report: `reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.json`
- Gold metrics: `reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_gold_eval_20260518.json`
- Log: `logs/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.log`

Counts:

- Persona count: `15`
- Question count: `100`
- Newly ingested corpora: `15`
- Reused corpora: `0`
- Turn/event count: `1132`
- Atoms written: `1887`
- Objects written in report summary: `1038`; final active `leaf_objects` rows in DB: `962`
- Snapshots: `1059`
- Evidence links: `1645`
- Memory LLM calls estimated from ingest summary: `302` total (`264` atom extraction, `38` reconciliation)
- Memory LLM prompt tokens from provider usage: `80765` total (`65010` atom extraction, `15755` reconciliation)

Metrics:

| Run | Q | F1 | BLEU1 | strict EM | accepted | avg search ms | avg answer tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF-Base old reused DB | 100 | 0.7379 | 0.6760 | 0.37 | 0.42 | 103.51 | 1041.60 |
| LEAF-Base-GPT54MiniMemory fresh | 100 | 0.7326 | 0.6693 | 0.36 | 0.42 | 73.21 | 1042.57 |
| Gated evolved topic-soft active views | 100 | 0.7416 | 0.6814 | 0.37 | 0.44 | 157.68 | 1198.79 |

Caveats:

- The older LEAF-Base baseline reused a 2026-04-18 DB; its original memory model is not fully recoverable from DB metadata.
- This fresh baseline provides the stricter comparison point for current `gpt-5.4-mini` memory construction.

## 2026-05-18 Chinese GVD Fresh Baseline And Gated Evolved Topic-Soft

Purpose:

- Repeat the strict baseline/evolved comparison on Chinese GVD full100.
- Use fresh `gpt-5.4-mini` memory construction for the LEAF-Base baseline, then build seed/evolved topic views on a copy of the same fresh DB.

Inputs:

- Config: `tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml`
- Memory construction/evolution LLM: CodeAI `gpt-5.4-mini`
- Answer LLM: local `Qwen3-235B` at `http://localhost:8001/v1`
- Memory bank: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/memory_bank_cn.json`
- Questions: `/vepfs-mlp2/c20250513/241404044/users/roytian/MemoryBank-SiliconFriend/eval_data/cn/probing_questions_cn.jsonl`
- Gold file: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF/reports/gvd_gold_cn_full100_gpt41mini_polished_v4_20260417.json`

Outputs:

- Fresh baseline DB: `data/agentic_smoke/gvd_cn_leafbase_gpt54mini_memory_full100_20260518.sqlite3`
- Evolved working DB: `data/agentic_smoke/gvd_cn_agentic_full100_all_evolved_20260518.sqlite3`
- Evolution output: `reports/agentic_memory/gvd_cn_all_evolved_20260518/pipeline.jsonl`
- Evolution summary: `reports/agentic_memory/gvd_cn_all_evolved_20260518/pipeline_summary.json`
- Fresh baseline report: `reports/agentic_memory/gvd_cn_full100_leafbase_gpt54mini_memory_structctx_20260518.json`
- Fresh baseline gold eval: `reports/agentic_memory/gvd_cn_full100_leafbase_gpt54mini_memory_structctx_gold_eval_20260518.json`
- Gated evolved report: `reports/agentic_memory/gvd_cn_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.json`
- Gated evolved gold eval: `reports/agentic_memory/gvd_cn_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_gold_eval_20260518.json`

Fresh baseline counts:

- Newly ingested corpora: `15`
- Reused corpora: `0`
- Turn/event count: `1132`
- Atoms written: `1637`
- Objects written: `79`
- Memory LLM calls estimated from ingest summary: `351` atom-extraction calls, `0` reconciliation calls
- Memory LLM prompt tokens: `86088`

Evolution counts:

- Corpus count: `15`
- Promoted evolved active views: `12`
- Gate failed: `3` (`gvd_李雪`, `gvd_郝明`, `gvd_陈阳`)
- Active view mix after gate: `12` evolved topic trees, `3` seed topic trees

Metrics:

| Run | Q | F1 | BLEU1 | strict EM | accepted | avg search ms | avg answer tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-15 Chinese baseline structctx | 100 | 0.3150 | 0.3150 | 0.31 | 0.34 | 94.63 | 852.40 |
| 2026-05-15 Chinese seed topic-soft structctx | 100 | 0.3250 | 0.3250 | 0.32 | 0.35 | 96.76 | 881.51 |
| 2026-05-18 Chinese LEAF-Base-GPT54MiniMemory fresh | 100 | 0.3150 | 0.3150 | 0.31 | 0.34 | 82.35 | 826.01 |
| 2026-05-18 Chinese gated evolved topic-soft | 100 | 0.3317 | 0.3287 | 0.32 | 0.35 | 114.80 | 865.98 |

Deltas:

- Gated evolved - fresh baseline: `+0.0167` F1, `+0.0137` BLEU1, `+0.01` strict EM, `+0.01` accepted.
- Gated evolved - 2026-05-15 seed topic-soft structctx: `+0.0067` F1, `+0.0037` BLEU1, equal strict EM and accepted.

Caveats:

- Chinese evolved topic generation still often emits English slugs/keywords, so keyword routing frequently lands in broad topics such as `misc`.
- The QA gain is positive under the same retrieval mode, but only two questions changed materially versus the fresh baseline; one exact-match formatting improvement and one content improvement.

## 2026-05-18 Current Reporting Scope

The current public/internal reporting table is now restricted to **GPT-5.4-mini Base vs Evolved only**.

- Canonical summary: `docs/gpt54mini_base_vs_evolved_results_20260518.md`
- Benchmarks covered: LoCoMo, GVD EN, GVD ZH
- QA answer model for these reports: local `Qwen3-235B`
- Memory construction/evolution model for these reports: `gpt-5.4-mini`

Older seed-only, reused-old-memory, oracle-type, partial-slice, failed-ablation, and intermediate-policy rows should not be included in the main result table.

## 2026-05-18 Current GPT-5.4-Mini Base/Evolved Judge-1 Supplement

Purpose:

- Add LLM-as-judge scores for the six current reporting rows only:
  - LoCoMo Base / Evolved
  - GVD EN Base / Evolved
  - GVD ZH Base / Evolved
- Keep the current reporting scope restricted to `gpt-5.4-mini` Base vs Evolved.

Judge setup:

- Judge model: local OpenAI-compatible `Qwen3-235B`
- Judge URL: `http://localhost:8002/v1`
- API key: local OpenAI-compatible placeholder, not recorded in git.
- Config: `tmp/config_agentic_qwen8002_judge.yaml`
- Launcher: `scripts/launch_current_judge6_parallel_20260518.sh`
- Execution: six parallel tmux jobs, one per report; final LoCoMo jobs used `16` workers each after GVD completed.
- Judge runs per item: `1`
- LoCoMo judge style: `legacy_binary`
- GVD judge style: gold-reference judge with scores in `{0, 0.5, 1}`

Outputs:

- LoCoMo Base Judge-1: `reports/agentic_memory/judge_20260518/locomo10_base_gpt54mini_qwen8002_judge1_legacybinary_20260518.json`
- LoCoMo Evolved Judge-1: `reports/agentic_memory/judge_20260518/locomo10_evolved_gpt54mini_qwen8002_judge1_legacybinary_20260518.json`
- GVD EN Base Judge-1: `reports/agentic_memory/judge_20260518/gvd_en_base_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- GVD EN Evolved Judge-1: `reports/agentic_memory/judge_20260518/gvd_en_evolved_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- GVD ZH Base Judge-1: `reports/agentic_memory/judge_20260518/gvd_zh_base_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- GVD ZH Evolved Judge-1: `reports/agentic_memory/judge_20260518/gvd_zh_evolved_gpt54mini_qwen8002_goldref_judge1_20260518.json`

Logs:

- `logs/judge_locomo_base_qwen8002_20260518.log`
- `logs/judge_locomo_evolved_qwen8002_20260518.log`
- `logs/judge_gvd_en_base_qwen8002_20260518.log`
- `logs/judge_gvd_en_evolved_qwen8002_20260518.log`
- `logs/judge_gvd_zh_base_qwen8002_20260518.log`
- `logs/judge_gvd_zh_evolved_qwen8002_20260518.log`

Judge-1 metrics:

| Benchmark | Run | Q | Judge-1 |
| --- | ---: | ---: | ---: |
| LoCoMo | Base | 1540 | 0.7714 |
| LoCoMo | Evolved | 1540 | 0.7727 |
| GVD EN | Base | 100 | 0.8900 |
| GVD EN | Evolved | 100 | 0.9050 |
| GVD ZH | Base | 100 | 0.9300 |
| GVD ZH | Evolved | 100 | 0.9400 |

Deltas:

- LoCoMo Evolved - Base Judge-1: `+0.0013`
- GVD EN Evolved - Base Judge-1: `+0.0150`
- GVD ZH Evolved - Base Judge-1: `+0.0100`

Canonical reporting document updated:

- `docs/gpt54mini_base_vs_evolved_results_20260518.md`

Notes:

- The interrupted early LoCoMo Judge-5 progress file was renamed to `reports/agentic_memory/judge_20260518/locomo10_base_gpt54mini_qwen8002_judge5_legacybinary_20260518.judge_progress.abandoned.jsonl` and is not part of the current reporting table.

## 2026-05-18 Mem0-Inspired Hybrid Sidecar

Purpose:

- Borrow useful ideas from current `mem0ai/mem0` without changing LEAF's base event/atom storage: ADD-only compact memory facts, lexical/BM25-style normalized terms, entity boosts, and search-time hybrid rerank without LLM calls.
- Evaluate whether this improves the 4-corpus LoCoMo challenge subset over the current LEAF-base anchor.

Code branch and backup:

- Project path: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_agentic_memory`
- Working branch: `feature/mem0-inspired-hybrid-memory`
- Backup branch pushed before implementation: `backup/agentic-memory-before-mem0-inspired-20260518`
- Backup commit: `982f7b1 backup: preserve agentic memory evolution state`

Main code changes:

- `src/leaf/records.py`: added `AdditiveMemoryRecord`.
- `src/leaf/store.py`: added `leaf_additive_memories` persistence helpers.
- `src/leaf/service.py`: cache support for all atoms and additive memories.
- `src/leaf/hybrid_retrieval.py`: new local hybrid index over events + additive memories with BM25-style term matching and entity boost.
- `src/leaf/search.py`: optional mem0-style hybrid boost, additive-memory source filtering, temporal guard, and shared-profile support overlay.
- `scripts/build_additive_memory_sidecar.py`: new sidecar builder with `event_atom`, `dialog_window`, and LLM ADD-only fact modes; `llm_fact_v2` uses a mem0-inspired extraction prompt and parallel workers.
- `scripts/eval_locomo.py`: records mem0 hybrid diagnostics and strips additive sidecar spans from deterministic postprocess.

Input artifacts:

- Config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`
- Challenge input: `tmp/locomo_challenge_conv30_43_47_50_noadv.json`
- DB used for mem0-inspired run: `data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_llmfacts_20260518.sqlite3`
- Baseline anchor report: `reports/agentic_memory/locomo_challenge_conv30_43_47_50_leafbase_current_qaw8_20260518.json`
- Previous overlay anchor: `reports/agentic_memory/locomo_challenge_conv30_43_47_50_overlay_gate_v2_no_open_qaw8_20260518.json`

Sidecar build:

```bash
./scripts/with_tracenav_nlp.sh python scripts/build_additive_memory_sidecar.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_llmfacts_20260518.sqlite3 \
  --corpus-id locomo_conv_30 --corpus-id locomo_conv_43 \
  --corpus-id locomo_conv_47 --corpus-id locomo_conv_50 \
  --mode llm_facts \
  --llm-source llm_fact_v2 \
  --llm-extraction-policy mem0_additive_v2 \
  --llm-window-size 8 \
  --llm-max-facts-per-window 8 \
  --llm-workers 6 \
  --llm-timeout 90 \
  --max-atoms-per-event 4 \
  --flush-every-windows 10 \
  --output reports/agentic_memory/locomo_challenge4_llmfact_v2_sidecar_build_retry_20260518.json
```

Sidecar output:

- Report: `reports/agentic_memory/locomo_challenge4_llmfact_v2_sidecar_build_retry_20260518.json`
- Log: `logs/locomo_challenge4_llmfact_v2_sidecar_build_retry_20260518.log`
- `locomo_conv_30`: 343 new `llm_fact_v2` facts in retry run; 52 windows; 1 API-format error. Current DB has 723 `llm_fact_v2` rows because an earlier successful build wrote 380 rows before retry.
- `locomo_conv_43`: 680 new `llm_fact_v2` facts; 96 windows; 0 errors. Current DB has 695 rows including a 15-row smoke build.
- `locomo_conv_47`: 655 `llm_fact_v2` facts; 100 windows; 0 errors.
- `locomo_conv_50`: 587 `llm_fact_v2` facts; 84 windows; 1 API-format error.

Primary QA command:

```bash
LEAF_MEM0_HYBRID=1 \
LEAF_MEM0_HYBRID_CANDIDATE_ONLY=1 \
LEAF_MEM0_HYBRID_FINAL_WEIGHT=0.20 \
LEAF_MEM0_HYBRID_MAX_BONUS=0.14 \
LEAF_ADDITIVE_SIDECAR_SOURCES=llm_fact_v2 \
LEAF_SHARED_PROFILE_OVERLAY=1 \
LEAF_SHARED_PROFILE_OVERLAY_MODE=support \
LEAF_SHARED_PROFILE_OVERLAY_LIMIT=2 \
LEAF_SHARED_PROFILE_GEO_MODE=broad \
./scripts/with_tracenav_nlp.sh python scripts/eval_locomo.py \
  --config tmp/config_agentic_codeai_memory_qwen_answer.yaml \
  --db data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_llmfacts_20260518.sqlite3 \
  --input tmp/locomo_challenge_conv30_43_47_50_noadv.json \
  --output reports/agentic_memory/locomo4_mem0v2_llmfact_hybrid_sharedsupport_evolved_profileauto_fullsubset_20260518.json \
  --sample-limit 4 \
  --snapshot-limit 8 \
  --raw-span-limit 12 \
  --non-temporal-raw-span-limit 12 \
  --retrieval-mode topic_soft_selective \
  --topic-router evolved_profile_first \
  --topic-route-top-k 3 \
  --topic-soft-event-limit 4 \
  --topic-soft-per-topic-atom-limit 16 \
  --topic-soft-min-content-overlap 1 \
  --topic-soft-secondary-policy strict_text_v0 \
  --topic-soft-secondary-min-content-overlap 2 \
  --topic-soft-secondary-min-route-keyword-overlap 1 \
  --topic-soft-policy route_uncertainty_semantic_v0 \
  --topic-soft-policy-min-selected-semantic-similarity 0.12 \
  --topic-soft-policy-suppress-multi-route \
  --topic-soft-fallback baseline_on_unknown \
  --answer-style structured_context_topic_labeled \
  --answer-evidence-mode merged \
  --answer-view-mode heuristic \
  --short-answer-postprocess precise
```

Primary QA result:

- Report: `reports/agentic_memory/locomo4_mem0v2_llmfact_hybrid_sharedsupport_evolved_profileauto_fullsubset_20260518.json`
- Log: `logs/locomo4_mem0v2_llmfact_hybrid_sharedsupport_fullsubset_20260518.log`
- Completed: `true`
- Evaluated questions: `567`
- Overall: F1 `0.5641`, BLEU1 `0.5091`
- Average search latency: `319.72 ms`

Comparison:

| Run | Q | F1 | BLEU1 |
| --- | ---: | ---: | ---: |
| LEAF-base current challenge anchor | 567 | 0.5520 | 0.4995 |
| Previous overlay anchor | 567 | 0.5551 | 0.5020 |
| Mem0-inspired `llm_fact_v2` hybrid + shared support | 567 | 0.5641 | 0.5091 |

Per-corpus F1/BLEU1:

| Corpus | LEAF-base | Mem0-inspired |
| --- | ---: | ---: |
| conv-30 | 0.5711 / 0.5158 | 0.5886 / 0.5311 |
| conv-43 | 0.5988 / 0.5423 | 0.6059 / 0.5473 |
| conv-47 | 0.5315 / 0.4847 | 0.5487 / 0.4990 |
| conv-50 | 0.5088 / 0.4569 | 0.5189 / 0.4643 |

Per-category F1/BLEU1:

| Category | LEAF-base | Mem0-inspired |
| --- | ---: | ---: |
| multi_hop | 0.3784 / 0.2874 | 0.4043 / 0.3146 |
| open_domain | 0.2582 / 0.2294 | 0.2712 / 0.2392 |
| single_hop | 0.5992 / 0.5505 | 0.6169 / 0.5648 |
| temporal | 0.6465 / 0.6075 | 0.6319 / 0.5903 |

Caveats:

- The improvement is broad across all four corpora and non-temporal categories, but temporal questions regressed.
- `llm_fact_v2` rows for conv-30 and conv-43 include smoke/retry leftovers; future clean runs should either use a fresh DB or add window-level source/run cleanup to avoid semantically near-duplicate facts.
- A temporal runtime-skip ablation was started and stopped early because conv43 degraded sharply in the first 107 questions; partial output is `reports/agentic_memory/locomo4_mem0v2_temporalskip_llmfact_hybrid_sharedsupport_evolved_profileauto_fullsubset_20260518.json` and should not be used as a final result.
- Directly adding `llm_fact_v1` sidecar facts into the answer context on conv-30 underperformed the hybrid-only path: `reports/agentic_memory/locomo4_mem0style_llmfact_directsupport2_sharedsupport_evolved_profileauto_conv30_20260518.json` gave F1 `0.5712`, BLEU1 `0.5135`, versus the prior conv-30 hybrid/shared support result F1 `0.5819`, BLEU1 `0.5271`.
