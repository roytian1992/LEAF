# GPT-5.4-Mini Base vs Evolved Results - 2026-05-18

## Reporting Scope

Current reporting scope is intentionally restricted to two run families only:

- **Base**: LEAF base memory, built with `gpt-5.4-mini` where fresh memory construction was required, answered by local `Qwen3-235B`.
- **Evolved**: agentic/evolved memory variant, with evolution handled by `gpt-5.4-mini`, answered by local `Qwen3-235B`.

Do not include seed-only runs, reused-old-memory runs from other model settings, oracle-type runs, partial slices, failed ablations, or intermediate policy variants in the main result table.

Common model/config assumptions:

- Memory construction / evolution LLM: `gpt-5.4-mini` via CodeAI `https://codeai.ysaikeji.cn/v1`.
- QA answer LLM: local OpenAI-compatible `Qwen3-235B` at `http://localhost:8001/v1`.
- LLM-as-judge: local OpenAI-compatible `Qwen3-235B` at `http://localhost:8002/v1`, Judge-1.
- Embedding service: `bge-m3` at `http://127.0.0.1:8080/v1`.
- Main config: `tmp/config_agentic_codeai_memory_qwen_answer.yaml`.
- Chinese config: `tmp/config_agentic_codeai_memory_qwen_answer_zh.yaml`.
- Judge config: `tmp/config_agentic_qwen8002_judge.yaml`.

## Overall Results

| Benchmark | Run | Questions | F1 | BLEU1 | Judge-1 | Strict EM | Accepted | Avg search ms | Avg answer input tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | Base | 1540 | 0.5593 | 0.5022 | 0.7714 | - | - | 12295.73 | 2190.67 |
| LoCoMo | Evolved | 1540 | 0.5596 | 0.5022 | 0.7727 | - | - | 12423.78 | 2273.21 |
| GVD EN | Base | 100 | 0.7326 | 0.6693 | 0.8900 | 0.36 | 0.42 | 73.21 | 1042.57 |
| GVD EN | Evolved | 100 | 0.7416 | 0.6814 | 0.9050 | 0.37 | 0.44 | 157.68 | 1198.79 |
| GVD ZH | Base | 100 | 0.3150 | 0.3150 | 0.9300 | 0.31 | 0.34 | 82.35 | 826.01 |
| GVD ZH | Evolved | 100 | 0.3317 | 0.3287 | 0.9400 | 0.32 | 0.35 | 114.80 | 865.98 |

## Deltas

| Benchmark | Evolved - Base F1 | Evolved - Base BLEU1 | Evolved - Base Judge-1 | Evolved - Base Strict EM | Evolved - Base Accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoCoMo | +0.0003 | +0.0000 | +0.0013 | - | - |
| GVD EN | +0.0090 | +0.0121 | +0.0150 | +0.01 | +0.02 |
| GVD ZH | +0.0167 | +0.0137 | +0.0100 | +0.01 | +0.01 |

## LoCoMo Detail

Input:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`

Base:

- Report: `reports/agentic_memory/locomo10_leafbase_evalv16_textonly_notype_adaptive_rawspan_precise_qwen_answer_full_20260516.json`
- Judge-1 report: `reports/agentic_memory/judge_20260518/locomo10_base_gpt54mini_qwen8002_judge1_legacybinary_20260518.json`
- DB: `data/agentic_smoke/locomo10_oldmode_gpt54mini_memory_qwen_answer_20260515.sqlite3`
- Retrieval mode: `baseline`
- Answer style: `structured_context_topic_labeled`
- Evidence limits: `snapshot_limit=8`, `raw_span_limit=8`, `non_temporal_raw_span_limit=12`
- Postprocess: `temporal_postprocess=range`, `short_answer_postprocess=precise`

Evolved:

- Report: `reports/agentic_memory/locomo10_evolvedtopic_secondary_profileauto_topicsoft_textpolicy_full_20260516.json`
- Judge-1 report: `reports/agentic_memory/judge_20260518/locomo10_evolved_gpt54mini_qwen8002_judge1_legacybinary_20260518.json`
- DB: `data/agentic_smoke/locomo10_evolvedtopic_secondary_profileauto_evalv16_textonly_20260516.sqlite3`
- Evolution summaries:
  - `reports/agentic_memory/locomo2_evolvedtopic_secondary_profileauto_pipeline_summary_20260516.json`
  - `reports/agentic_memory/locomo8_evolvedtopic_secondary_profileauto_pipeline_summary_20260516.json`
- Evolution status: 9 / 10 corpora promoted; 1 corpus kept base active view after gate failure.
- Retrieval mode: `topic_soft_selective`
- Topic router: `keyword`, `topic_route_top_k=3`
- Topic policy: `text_temporal_suppressed_v0`
- Evidence limits: `snapshot_limit=8`, `raw_span_limit=8`, `non_temporal_raw_span_limit=12`
- Postprocess: `temporal_postprocess=range`, `short_answer_postprocess=precise`

LoCoMo category metrics:

| Run | Multi-hop F1 | Open-domain F1 | Single-hop F1 | Temporal F1 | Multi-hop Judge-1 | Open-domain Judge-1 | Single-hop Judge-1 | Temporal Judge-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.3499 | 0.3153 | 0.6243 | 0.6459 | 0.6489 | 0.6042 | 0.8335 | 0.7664 |
| Evolved | 0.3583 | 0.3209 | 0.6234 | 0.6405 | 0.6348 | 0.6146 | 0.8419 | 0.7601 |

## GVD EN Detail

Base:

- Report: `reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.json`
- Gold eval: `reports/agentic_memory/gvd_full100_leafbase_gpt54mini_memory_structctx_gold_eval_20260518.json`
- Judge-1 report: `reports/agentic_memory/judge_20260518/gvd_en_base_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- Log: `logs/gvd_full100_leafbase_gpt54mini_memory_structctx_20260518.log`
- DB: `data/agentic_smoke/gvd_leafbase_gpt54mini_memory_full100_20260518.sqlite3`
- Personas: 15
- Questions: 100
- Ingest: 15 new / 0 reused
- Turns/events: 1132
- Atoms written: 1887
- Objects written: 1038
- Snapshots: 1059
- Memory LLM calls: 302 total (`atom_extraction=264`, `reconciliation=38`)
- Memory LLM prompt tokens: 80765 total (`atom_extraction=65010`, `reconciliation=15755`)
- Retrieval mode: `baseline`
- Answer style: `structured_context`
- Evidence limits: `snapshot_limit=6`, `raw_span_limit=8`

Evolved:

- Report: `reports/agentic_memory/gvd_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.json`
- Gold eval: `reports/agentic_memory/gvd_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_gold_eval_20260518.json`
- Judge-1 report: `reports/agentic_memory/judge_20260518/gvd_en_evolved_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- Log: `logs/gvd_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.log`
- DB: `data/agentic_smoke/gvd_agentic_full100_all_evolved_20260518.sqlite3`
- Evolution pipeline: `reports/agentic_memory/gvd_all_evolved_20260518/pipeline.jsonl`
- Evolution summary: `reports/agentic_memory/gvd_all_evolved_20260518/pipeline_summary.json`
- Evolution status: 15 corpora total; 13 promoted, 1 already evolved active, 1 gate failed; 14 active evolved views after pipeline.
- QA ingest: 0 new / 15 reused
- Retrieval mode: `topic_soft`
- Topic-soft average events: 1.95
- Topic-soft average candidate atoms: 15.98
- Topic-soft fallback count: 0
- Answer style: `structured_context`

## GVD ZH Detail

Base:

- Report: `reports/agentic_memory/gvd_cn_full100_leafbase_gpt54mini_memory_structctx_20260518.json`
- Gold eval: `reports/agentic_memory/gvd_cn_full100_leafbase_gpt54mini_memory_structctx_gold_eval_20260518.json`
- Judge-1 report: `reports/agentic_memory/judge_20260518/gvd_zh_base_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- Log: `logs/gvd_cn_full100_leafbase_gpt54mini_memory_structctx_20260518.log`
- DB: `data/agentic_smoke/gvd_cn_leafbase_gpt54mini_memory_full100_20260518.sqlite3`
- Personas: 15
- Questions: 100
- Ingest: 15 new / 0 reused
- Turns/events: 1132
- Atoms written: 1637
- Objects written: 79
- Snapshots: 787
- Memory LLM calls: 351 total (`atom_extraction=351`, `reconciliation=0`)
- Memory LLM prompt tokens: 86088 total (`atom_extraction=86088`, `reconciliation=0`)
- Retrieval mode: `baseline`
- Answer style: `structured_context`
- Evidence limits: `snapshot_limit=6`, `raw_span_limit=8`

Evolved:

- Report: `reports/agentic_memory/gvd_cn_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.json`
- Gold eval: `reports/agentic_memory/gvd_cn_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_gold_eval_20260518.json`
- Judge-1 report: `reports/agentic_memory/judge_20260518/gvd_zh_evolved_gpt54mini_qwen8002_goldref_judge1_20260518.json`
- Log: `logs/gvd_cn_full100_gated_evolved_topicsoft_keyword_limit2_fallback_structctx_20260518.log`
- DB: `data/agentic_smoke/gvd_cn_agentic_full100_all_evolved_20260518.sqlite3`
- Evolution pipeline: `reports/agentic_memory/gvd_cn_all_evolved_20260518/pipeline.jsonl`
- Evolution summary: `reports/agentic_memory/gvd_cn_all_evolved_20260518/pipeline_summary.json`
- Evolution status: 15 corpora total; 12 promoted, 3 gate failed; 12 active evolved views after pipeline.
- QA ingest: 0 new / 15 reused
- Retrieval mode: `topic_soft`
- Topic-soft average events: 2.00
- Topic-soft average candidate atoms: 16.00
- Topic-soft fallback count: 0
- Answer style: `structured_context`

## Notes

- LoCoMo Base/Evolved reports are full 1540-question QA reports over already-ingested GPT-5.4-mini memory DBs.
- GVD EN/ZH Base reports are fresh GPT-5.4-mini memory construction runs; Evolved reports reuse their corresponding evolved DBs for QA after the evolution pipeline.
- Search-time QA still answers with local `Qwen3-235B`; the evolved runs use `gpt-5.4-mini` for offline memory/evolution work, not for answering.
- Judge-1 was run on `2026-05-18` with six parallel jobs. LoCoMo used `legacy_binary` report-level judging; GVD used gold-reference judging with scores in `{0, 0.5, 1}`.
