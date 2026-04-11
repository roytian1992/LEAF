# LoCoMo Results Log

这个文件用于持续记录 `LEAF` 在 `LoCoMo` 上的实验结果，后续新的实现版本直接在这里追加或更新。

## Evaluation Scope

- 默认记录非 `adv` 类题目结果。
- 如果某次实验口径不同，必须在对应条目里显式说明。

## Historical Baseline

来源：你提供的上一版结果，先作为历史基线保留；其中 `LEAF` 的结果已经替换为官方版口径。

### Overall

| Method | F1 | BLEU-1 | Avg search (ms) | Avg answer input tokens | Judge mean | Judge std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF | 0.4590 | 0.4101 | 628.30 | 576.33 | - | - |
| mem0 | 0.4060 | 0.3455 | 75.29 | 819.24 | 0.5891 | 0.0030 |
| MemoryOS | 0.3844 | 0.3340 | 379.57 | 1789.64 | - | - |

### By-Category F1

| Method | multi_hop | temporal | open_domain | single_hop |
| --- | ---: | ---: | ---: | ---: |
| LEAF | 0.3019 | 0.5254 | 0.2565 | 0.5095 |
| mem0 | 0.3670 | 0.4564 | 0.2447 | 0.4182 |
| MemoryOS | 0.2472 | 0.3976 | 0.2280 | 0.4432 |

### By-Category BLEU-1

| Method | multi_hop | temporal | open_domain | single_hop |
| --- | ---: | ---: | ---: | ---: |
| LEAF | 0.2361 | 0.4712 | 0.2273 | 0.4660 |
| mem0 | 0.2831 | 0.3968 | 0.1894 | 0.3647 |
| MemoryOS | 0.1790 | 0.3380 | 0.1957 | 0.4003 |

### Ingest Baseline

| Method | Ingest total (ms) | Avg ingest per sample (ms) | Cache / Reuse | Notes |
| --- | ---: | ---: | --- | --- |
| LEAF | 2344149.85 | 234414.99 | `fresh_ingest = 10/10`, `state_cache_hits = 1630`, `state_cache_misses = 7025` | Official LEAF, `migration` ingest, `workers = 4`, `migration_prepare_workers = 8`, `embedding_workers = 8`, snapshot embedding uses `summary`, batch wall total `680081.03 ms` |
| mem0 | 80582750.00 | 8058275.00 | `cache_hits = 3`, `cache_misses = 7`, `cache_resumes = 0` | From `mem0_ingest_seconds`; non-adv run |
| MemoryOS | 1803762.09 | 180376.21 | `reused = 0`, `ingested = 10` | Fresh non-adv rerun on `2026-04-11`; storage root `runs/memoryos_cache_withingest_20260411` |

### LEAF Ingest Per Sample

| Sample | Ingest elapsed (ms) | Events | Atoms | Objects written | State candidates | Evidence links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conv-26 | 170201.20 | 419 | 2037 | 546 | 625 | 625 |
| conv-30 | 152937.87 | 369 | 1592 | 393 | 455 | 455 |
| conv-41 | 275851.45 | 663 | 3215 | 684 | 794 | 794 |
| conv-42 | 243378.43 | 629 | 2681 | 767 | 890 | 890 |
| conv-43 | 260398.18 | 680 | 3184 | 932 | 1126 | 1126 |
| conv-44 | 267513.42 | 675 | 3017 | 943 | 1107 | 1107 |
| conv-47 | 272583.83 | 689 | 3166 | 925 | 1073 | 1073 |
| conv-48 | 261900.51 | 681 | 2803 | 819 | 964 | 964 |
| conv-49 | 198028.14 | 509 | 2218 | 609 | 699 | 699 |
| conv-50 | 241356.82 | 568 | 2850 | 766 | 922 | 922 |

### Latest Official LEAF Ingest Run

- Run date: `2026-04-11`
- Code commit: `b06041c`
- Dataset: non-`adv` LoCoMo, `10` samples, `5882` turns, `272` sessions
- Config: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/configs/config.additional_llm_gpt41mini.yaml`
- Input: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/benchmarks/locomo/locomo10_noadv.json`
- Report: [locomo10_nonadv_official_migration_ingest_summary_w4_prepare8_embed8_20260411.json](/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/reports/locomo10_nonadv_official_migration_ingest_summary_w4_prepare8_embed8_20260411.json)
- DB dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/data/locomo10_nonadv_official_migration_ingest_summary_w4_prepare8_embed8_20260411`
- Prepare cache dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/LEAF_dev/data/locomo10_nonadv_official_migration_ingest_summary_w4_prepare8_embed8_20260411_cache`
- `stage_totals_ms`: `prepare_total = 2201178.53`, `apply_total = 121103.86`, `snapshot_total = 10425.39`, `prepare_cache_write = 10596.88`
- `state_action_counts_total`: `ADD = 7025`, `NONE = 1271`, `PATCH = 42`, `SUPERSEDE = 315`, `TENTATIVE = 2`
- `events_written_total = 5882`, `atoms_written_total = 26763`, `objects_written_total = 7384`
- `state_candidates_total = 8655`, `evidence_links_written_total = 8655`

注：

- 当前文档默认只记录非 `adv` 结果；总表与分类别表统一使用标准四类口径，不混入 `adv`。
- 总表中的 `LEAF` 结果已经替换为官方版结果；官方版没有 `Judge mean/std`，因此这两列记为 `-`。
- 总表中的时延列使用 `Avg search (ms)`，不再使用端到端 `elapsed_ms`。
- `LEAF` 的 `Overall` 指标来自官方版全量 QA 结果中过滤 `adv` 后的汇总；当前这次新增的是 ingest 统计，不改动 QA 指标。
- 当前代码默认使用 `summary` 做 snapshot embedding；`title + summary` 的 QA 对比结果更差，因此已回退。
- 本次 `LEAF` ingest 是全量冷启动运行，且保留了 prepared-turn cache，后续可以直接加载 prepared cache 跳过 prepare 阶段。
- `Ingest total (ms)` 使用各 sample 的 ingest elapsed 求和；并行批量实际墙钟时间单独记录为 `elapsed_wall_total_ms = 680081.03`。
- 与上一版全量 ingest 基线相比，`ingest_elapsed_ms_total` 从 `4482834.29` 降到了 `2344149.85`，批量墙钟时间从 `1293685.95` 降到了 `680081.03`。
- `objects written` 记录的是 ingest 过程中写入或更新的 object 次数，来自报告字段 `objects_written`，不等同于最终库内去重后的 `leaf_objects` 行数。
- `mem0` 的 `Avg search (ms)` 来自 `search_total_ms` 的逐题均值；其 ingest 统计来自同一份非 `adv` 报告顶层 `timing.mem0_ingest_seconds`。
- `MemoryOS` 的当前数值来自 `2026-04-11` 的新复跑结果，包含完整 ingest 字段。

## Update Template

后续新增结果时，按下面格式追加一节：

```md
## Run: YYYY-MM-DD <short tag>

- Scope:
- Config:
- Dataset filter:
- Notes:

### Overall

| Method | F1 | BLEU-1 | Avg search (ms) | Avg answer input tokens | Judge mean | Judge std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LEAF |  |  |  |  |  |  |

### By-Category F1

| Method | multi_hop | temporal | open_domain | single_hop |
| --- | ---: | ---: | ---: | ---: |
| LEAF |  |  |  |  |

### By-Category BLEU-1

| Method | multi_hop | temporal | open_domain | single_hop |
| --- | ---: | ---: | ---: | ---: |
| LEAF |  |  |  |  |

### Ingest Baseline

| Method | Ingest total (ms) | Avg ingest per sample (ms) | Cache / Reuse | Notes |
| --- | ---: | ---: | --- | --- |
| LEAF |  |  |  |  |

### LEAF Ingest Per Sample

| Sample | Ingest elapsed (ms) | Events | Atoms | Objects written | State candidates | Evidence links |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |
```
