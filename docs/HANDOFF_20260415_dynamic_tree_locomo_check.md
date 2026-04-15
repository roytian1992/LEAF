# LEAF Dynamic-Tree LoCoMo Check - 2026-04-15

## Scope

This note records the `LEAF` repository `dynamic-tree` branch LoCoMo QA run completed on 2026-04-15, with two goals:

- verify whether this branch is affected by the earlier benchmark-label-aware answer-routing issue
- preserve the final run paths and the added QA-time logging outputs


## Bottom Line

The `dynamic-tree` branch does **not** appear to use benchmark labels such as:

- `single_hop`
- `multi_hop`
- `open_domain`
- `temporal`

for runtime answer routing in `scripts/eval_locomo.py`.

Current conclusion:

- this run appears **free of the earlier benchmark-label answer-routing contamination**
- the answer path is driven by:
  - the question text
  - retrieved evidence
  - general inference heuristics such as `is_inference_query(question)`
- benchmark category fields are present for reporting / grouping only

This is a code-path conclusion, not a claim that every heuristic in the branch is philosophically "pure." But it does rule out the specific cheating pattern discussed in `LEAF_exp_offline_equiv`, namely benchmark-label-aware routing during answer generation.


## Why This Looks Clean

In `LEAF/scripts/eval_locomo.py`:

- `category` / `category_name` are loaded from benchmark rows and carried into result records
- they are used in logging and summary aggregation
- the final answer prompt is built from `question + evidence`
- the answer path does not branch on benchmark category

Relevant file:

- `LEAF/scripts/eval_locomo.py`

Important practical distinction:

- `is_inference_query(question)` is still used
- that is question-text-based routing, not benchmark-label routing


## Run Status

The full LoCoMo run completed successfully on 2026-04-15.

Final summary:

- Question count: `1540`
- Answer F1: `0.4346`
- BLEU-1: `0.3847`
- Avg elapsed: `1764.23 ms`
- Avg search: `1393.71 ms`
- Avg answer: `370.53 ms`

By category:

- Multi-hop F1: `0.2516`
- Open-domain F1: `0.2168`
- Single-hop F1: `0.4854`
- Temporal F1: `0.5271`


## Artifacts

Main DB:

- `LEAF/data/locomo10_dynamic_tree_20260415.sqlite3`

Main report:

- `LEAF/reports/locomo10_dynamic_tree_qa_20260415.json`

Per-QA progress log:

- `LEAF/reports/locomo10_dynamic_tree_qa_20260415.qa_progress.jsonl`

tmux stdout capture:

- `LEAF/tmp/locomo10_dynamic_tree_qa_20260415_tmux.log`


## Follow-Up Variant Anchors

### Merge-atom + YAKE-tag + official-style prompt

This follow-up variant kept the same `dynamic-tree` retrieval stack but changed memory construction:

- merge turns before atom extraction
- use non-LLM extractive summaries
- build tags with `yake`
- keep an official-style answer prompt

Final summary:

- Question count: `1540`
- Answer F1: `0.4355`
- BLEU-1: `0.3871`
- Multi-hop F1: `0.2724`
- Open-domain F1: `0.2373`
- Single-hop F1: `0.4958`
- Temporal F1: `0.4798`

Artifacts:

- `LEAF/data/locomo10_dynamic_tree_mergeatom_yaketag_official_20260415.sqlite3`
- `LEAF/reports/locomo10_dynamic_tree_mergeatom_yaketag_official_20260415.json`


### Merge-atom + YAKE-tag + mem0-style prompt

This run reused the same merged-atom / YAKE-tag DB, but switched the answer layer to a mem0-style prompt that asks for the single best-supported answer from evidence and does not explicitly permit `UNKNOWN`.

Final summary:

- Question count: `1540`
- Answer F1: `0.4461`
- BLEU-1: `0.3923`
- Avg elapsed: `2046.10 ms`
- Multi-hop F1: `0.2626`
- Open-domain F1: `0.2758`
- Single-hop F1: `0.4927`
- Temporal F1: `0.5362`

Comparison against the two earlier anchors in this note:

- vs clean dynamic-tree: `+0.0115` F1
- vs merge-atom + official-style prompt: `+0.0106` F1

Artifacts:

- `LEAF/data/locomo10_dynamic_tree_mergeatom_yaketag_official_20260415.sqlite3`
- `LEAF/reports/locomo10_dynamic_tree_mergeatom_yaketag_mem0prompt_20260415.json`
- `LEAF/reports/locomo10_dynamic_tree_mergeatom_yaketag_mem0prompt_20260415.qa_progress.jsonl`
- `LEAF/tmp/locomo10_dynamic_tree_mergeatom_yaketag_mem0prompt_20260415_tmux.log`

Interpretation:

- this result is still compatible with the earlier "no benchmark-label answer routing" claim
- the gain appears concentrated in `open_domain` and `temporal`
- because this variant changed the answer prompt semantics, it should be treated as a prompt-policy comparison, not a pure retrieval-only comparison


## Added Logging For Future Prompt Comparison

For this run, `scripts/eval_locomo.py` was updated so that QA results are written incrementally and include the concrete answer input bundle.

Each QA row now preserves:

- retrieved memories
- `answer_context_lines`
- `answer_view`
- `answer_view_text`
- `answer_prompt_messages`

This should make later answer-prompt comparison much easier because retrieval outputs and final answer-prompt inputs are now inspectable per question.


## Important Caveat

This note only supports the narrow claim that the branch does **not** appear to cheat via benchmark-label-aware answer routing.

It does **not** mean:

- the branch is the best LoCoMo baseline
- the summary / metadata design is ideal
- the run is directly comparable to every `LEAF_exp_offline_equiv` baseline without additional setup notes

It only means the earlier "using benchmark category at runtime to answer" problem does not appear to be present here.


## 2026-04-16 Retrieval-Focused Follow-Up

After the earlier prompt-policy comparisons, a retrieval-focused follow-up was tested on top of the merged-atom + YAKE-tag + mem0-style `6/8` setup.

Key retrieval changes:

- add light query term normalization for simple inflectional variants such as `research` / `researching`, `painted` / `painting`, `camped` / `camping`
- score event support text using `blip_caption` and `semantic_refs` explicitly instead of relying on raw metadata stringification
- keep the final answer evidence budget unchanged at `snapshot_limit=6`, `raw_span_limit=8`

Smoke on `conv-26`:

- baseline `missed_in_retrieval`: `63`
- retrieval-focused variant `missed_in_retrieval`: `50`
- baseline avg F1: `0.4183`
- retrieval-focused variant avg F1: `0.4360`

Full LoCoMo run on 2026-04-16:

- Question count: `1540`
- Answer F1: `0.4700`
- BLEU-1: `0.4130`
- Multi-hop F1: `0.2766`
- Open-domain F1: `0.2765`
- Single-hop F1: `0.5268`
- Temporal F1: `0.5490`

Comparison against the previous best `6/8` mem0-style prompt run:

- previous best: `F1 0.4461`, `BLEU-1 0.3923`
- retrieval-focused follow-up: `F1 0.4700`, `BLEU-1 0.4130`
- delta: `+0.0239` F1, `+0.0207` BLEU-1

Important cost:

- avg search latency rose sharply to about `30.5s` per question in the `8`-worker QA-only full run
- the run completed successfully with `0` explicit `__ERROR__` answer rows

Artifacts:

- `LEAF/reports/locomo10_dynamic_tree_mergeatom_yaketag_mem0prompt_retrievalboost_6x8_parallel8_20260416.json`
- `LEAF/reports/locomo10_dynamic_tree_mergeatom_yaketag_mem0prompt_retrievalboost_6x8_parallel8_20260416.qa_progress.jsonl`
- `LEAF/tmp/locomo10_dynamic_tree_mergeatom_yaketag_mem0prompt_retrievalboost_6x8_parallel8_20260416_tmux.log`

Interpretation:

- this is the best LoCoMo result observed so far on this branch during this investigation
- the gain appears to come from retrieval, not from widening the final answer evidence budget
- answer-layer quality is still not fully saturated, because some questions moved from `missed_in_retrieval` into `surfaced_but_answer_wrong` or `context_but_not_answer_view`
