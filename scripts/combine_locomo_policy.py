from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import eval_locomo as locomo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine two LoCoMo result JSONs with a retrieval-policy selector.")
    parser.add_argument("--baseline", required=True, help="Result JSON used when the policy rejects the variant.")
    parser.add_argument("--variant", required=True, help="Result JSON used when the policy accepts the variant.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deny-topic-slugs",
        default="",
        help="Comma-separated active topic slugs for which the baseline row should be used.",
    )
    parser.add_argument(
        "--policy-name",
        default="deny_topic_slug_v0",
        help="Human-readable policy name stored in the output payload and rows.",
    )
    return parser.parse_args()


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("sample_id") or ""), int(row.get("question_index") or 0)


def active_topic_slug(row: dict[str, Any]) -> str:
    slugs = [str(item).strip() for item in (row.get("topic_soft_active_topic_slugs") or []) if str(item).strip()]
    return slugs[0] if slugs else ""


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    variant_path = Path(args.variant)
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    variant_payload = json.loads(variant_path.read_text(encoding="utf-8"))
    baseline_rows = {row_key(row): dict(row) for row in baseline_payload.get("results") or []}
    variant_rows = {row_key(row): dict(row) for row in variant_payload.get("results") or []}
    if set(baseline_rows) != set(variant_rows):
        missing_baseline = sorted(set(variant_rows) - set(baseline_rows))[:5]
        missing_variant = sorted(set(baseline_rows) - set(variant_rows))[:5]
        raise RuntimeError(
            "Baseline and variant row keys differ. "
            f"missing_baseline={missing_baseline}; missing_variant={missing_variant}"
        )

    deny_slugs = {
        str(item).strip()
        for item in str(args.deny_topic_slugs or "").split(",")
        if str(item).strip()
    }
    combined_rows: list[dict[str, Any]] = []
    selected_variant_count = 0
    selected_baseline_count = 0
    denied_counts: dict[str, int] = {}
    for key in sorted(baseline_rows, key=lambda item: (item[0], item[1])):
        baseline_row = baseline_rows[key]
        variant_row = variant_rows[key]
        slug = active_topic_slug(variant_row)
        use_variant = slug not in deny_slugs
        source_row = dict(variant_row if use_variant else baseline_row)
        source_row["policy_combiner"] = str(args.policy_name)
        source_row["policy_combiner_source"] = "variant" if use_variant else "baseline"
        source_row["policy_combiner_active_topic_slug"] = slug
        if use_variant:
            selected_variant_count += 1
        else:
            selected_baseline_count += 1
            denied_counts[slug] = denied_counts.get(slug, 0) + 1
        combined_rows.append(source_row)

    output_payload = dict(variant_payload)
    output_payload["results"] = combined_rows
    output_payload["policy_combiner"] = str(args.policy_name)
    output_payload["policy_combiner_baseline"] = str(baseline_path)
    output_payload["policy_combiner_variant"] = str(variant_path)
    output_payload["policy_combiner_deny_topic_slugs"] = sorted(deny_slugs)
    output_payload["policy_combiner_selected_variant_count"] = selected_variant_count
    output_payload["policy_combiner_selected_baseline_count"] = selected_baseline_count
    output_payload["policy_combiner_denied_counts"] = dict(sorted(denied_counts.items()))
    output_payload["summary"] = locomo.build_summary(
        samples=[],
        ingest_rows=list(variant_payload.get("ingest") or baseline_payload.get("ingest") or []),
        results=combined_rows,
        judge_with_llm=False,
        judge_runs=0,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
