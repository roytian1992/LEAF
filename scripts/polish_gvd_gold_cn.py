from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from leaf.clients import ChatClient, extract_json_object  # noqa: E402
from leaf.config import ModelConfig  # noqa: E402


BAD_PREFIXES = (
    "我建议",
    "建议你",
    "你在",
    "我们聊了",
    "最近你在",
    "我分享了",
    "我推荐了",
    "我给你推荐了",
)
STOP_TOKENS = {
    "",
    "对",
    "不对",
    "不是",
    "是",
    "了",
    "的",
    "和",
    "或",
    "以及",
    "一个",
    "一种",
    "一些",
    "这个",
    "那个",
    "事情",
    "经历",
    "话题",
}


def load_config(path: str | Path) -> ChatClient:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    llm_cfg = cfg.get("llm") or {}
    model_cfg = ModelConfig(
        provider=str(llm_cfg.get("provider") or "openai"),
        model_name=str(llm_cfg["model_name"]),
        api_key=str(llm_cfg["api_key"]),
        base_url=str(llm_cfg["base_url"]),
        timeout=int(llm_cfg.get("timeout", 180)),
        temperature=0.0,
        max_tokens=384,
    )
    return ChatClient(model_cfg)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_memory_bank(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized[str(key).strip()] = value
    return normalized


def normalize_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[。！？；]+$", "", cleaned)
    cleaned = cleaned.replace(" ,", ",")
    return cleaned.strip()


def has_explicit_date(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{4}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?", normalized))


def normalize_list_punctuation(text: str) -> str:
    normalized = normalize_text(text)
    normalized = normalized.replace("以及", "、")
    normalized = normalized.replace("及", "、")
    normalized = normalized.replace("和", "、")
    normalized = normalized.replace(",", "，")
    normalized = re.sub(r"[，、]{2,}", "、", normalized)
    normalized = re.sub(r"^、+|、+$", "", normalized)
    normalized = re.sub(r"^，+|，+$", "", normalized)
    return normalize_text(normalized)


def compress_canonical(question: str, answer: str) -> str:
    question_type = detect_question_type(question)
    canonical = normalize_text(answer)
    question_text = normalize_text(question)
    if not canonical:
        return canonical

    if question_type == "yes_no":
        match = re.match(r"^(不对|不是|对|是)[，,]?(.*)$", canonical)
        if match:
            prefix, rest = match.groups()
            rest = normalize_text(rest)
            rest = re.sub(r"^(你|我|他|她)(最近|曾经|当时|在)?", "", rest)
            rest = re.sub(
                r"^(分享的是|分享了|喜欢|尝试了|读了|聊了|看到了|推荐了|提到的是|提到了|告诉过我|给我推荐了)",
                "",
                rest,
            )
            if has_explicit_date(question_text):
                rest = re.sub(r"^(?:在)?(?:\d{4}-\d{2}-\d{2}|\d{4}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?)(?:这天)?", "", rest)
            rest = normalize_text(rest.lstrip("，, "))
            rest = re.sub(r"^(?:你)?分享的是", "", rest)
            rest = normalize_text(rest)
            return prefix if not rest else f"{prefix}，{rest}"

    rewrite_prefixes = (
        "我们聊了",
        "聊了",
        "我推荐了",
        "我给你推荐了",
        "推荐了",
        "你最近尝试了",
        "最近你在尝试",
        "看到了",
        "灵感来自",
        "因为",
        "你感到",
        "我建议你",
        "建议你",
        "我分享了",
        "你分享了",
        "你曾分享过",
    )
    for prefix in rewrite_prefixes:
        if canonical.startswith(prefix):
            canonical = normalize_text(canonical[len(prefix) :].lstrip("，, "))
            break
    for prefix in ("你最近", "你曾经", "你当时", "我最近", "我曾经", "我当时"):
        if canonical.startswith(prefix):
            canonical = normalize_text(canonical[len(prefix) :])
            break
    if has_explicit_date(question_text):
        canonical = re.sub(r"^(?:在)?(?:\d{4}-\d{2}-\d{2}|\d{4}年\d{1,2}月\d{1,2}[日号]?|\d{1,2}月\d{1,2}[日号]?)(?:这天)?", "", canonical)
    if "为什么" in question_text and canonical.startswith("因为"):
        canonical = normalize_text(canonical[2:])
    if "聊了什么话题" in question_text or "聊了哪些话题" in question_text:
        canonical = re.sub(r"^聊了", "", canonical)
        canonical = canonical.replace("你的", "")
        canonical = canonical.replace("兴趣爱好和建议", "兴趣爱好")
        canonical = canonical.replace("棋牌类游戏的技巧", "棋牌技巧")
        canonical = canonical.replace("类游戏的技巧", "游戏技巧")
        canonical = canonical.replace("葡萄酒的推荐", "葡萄酒推荐")
        canonical = normalize_list_punctuation(canonical)
    if "推荐了哪些音乐" in question_text:
        canonical = re.sub(r"的?(?:一些)?流行音乐(?:歌曲)?", "", canonical)
        canonical = normalize_list_punctuation(canonical)
    if "推荐了哪些电影" in question_text:
        canonical = re.sub(r"(?:这两部|这部)?电影", "", canonical)
        canonical = normalize_list_punctuation(canonical)
    if "推荐了哪些意大利菜" in question_text:
        canonical = normalize_list_punctuation(canonical)
    if "推荐过哪些郊外徒步的好地方" in question_text:
        canonical = re.sub(r"作为郊外徒步的好地方", "", canonical)
        canonical = normalize_list_punctuation(canonical)
    if "推荐过哪些画家" in question_text:
        canonical = re.sub(r"文艺复兴时期的画家", "", canonical)
        canonical = normalize_list_punctuation(canonical)
    if any(marker in question_text for marker in ("什么建议", "什么方法", "哪些方式", "如何放松", "如何克服焦虑和压抑")):
        canonical = canonical.replace("必要时", "")
        canonical = canonical.replace("可以", "")
        canonical = canonical.replace("尝试", "")
        canonical = canonical.replace("适当", "")
        canonical = canonical.replace("等放松技巧", "")
        canonical = normalize_list_punctuation(canonical)
    if "什么开心的事情" in question_text and "，" in canonical:
        canonical = canonical.split("，", 1)[0]
    if "谁帮助了我" in question_text and "什么麻烦" in question_text:
        canonical = canonical.replace("帮助我", "")
    if "谁的演唱会" in question_text and "什么歌" in question_text:
        singer_match = re.search(r"([\u4e00-\u9fffA-Za-z0-9·]+)的演唱会", canonical)
        song_match = re.search(r"(《[^》]+》)", canonical)
        if singer_match and song_match:
            canonical = f"{singer_match.group(1)} / {song_match.group(1)}"
        canonical = re.sub(r"^去看了", "", canonical)
    if "具体日期" in question_text or "具体的日期" in question_text:
        date_match = re.search(r"\d{4}-\d{2}-\d{2}|\d{4}年\d{1,2}月\d{1,2}[日号]?", canonical)
        if date_match:
            canonical = date_match.group(0)
    canonical = normalize_text(canonical)
    return canonical


def build_span_index(persona_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    history = persona_payload.get("history") or {}
    span_index: dict[str, dict[str, str]] = {}
    for session_id in sorted(history.keys()):
        rows = history.get(session_id) or []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            span_id = f"{session_id}#{idx}"
            query = normalize_text(str(row.get("query") or ""))
            response = normalize_text(str(row.get("response") or ""))
            speaker = normalize_text(str(row.get("speaker") or ""))
            span_index[span_id] = {
                "speaker": speaker,
                "query": query,
                "response": response,
                "combined": "\n".join(
                    part for part in (f"USER: {query}" if query else "", f"ASSISTANT: {response}" if response else "") if part
                ),
            }
    return span_index


def detect_question_type(question: str) -> str:
    text = normalize_text(question)
    if any(marker in text for marker in ("对吗", "是不是", "是否", "有没有", "难道")):
        return "yes_no"
    if any(marker in text for marker in ("哪些", "哪几", "什么书", "什么礼物", "什么建议", "什么方法", "什么技巧")):
        return "list"
    if any(marker in text for marker in ("什么时候", "哪天", "几月几号", "哪一年", "哪月哪日")):
        return "time"
    if any(marker in text for marker in ("谁", "哪里", "哪本", "哪个", "什么", "是？", "是?")):
        return "factoid"
    return "open"


def extract_candidate_terms(answer: str) -> list[str]:
    terms: list[str] = []
    terms.extend(re.findall(r"《[^》]+》", answer))
    terms.extend(re.findall(r"[A-Za-z0-9]{2,}", answer))
    parts = re.split(r"[、，,；;：:和与及或/（）() ]+", answer)
    for part in parts:
        part = normalize_text(part)
        if len(part) < 2 or part in STOP_TOKENS:
            continue
        terms.append(part)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def answer_length_limit(question_type: str) -> int:
    if question_type == "yes_no":
        return 18
    if question_type == "time":
        return 18
    if question_type == "factoid":
        return 22
    if question_type == "list":
        return 32
    return 36


def audit_answer(*, question: str, answer: str, evidence_text: str) -> list[str]:
    flags: list[str] = []
    canonical = compress_canonical(question, answer)
    question_type = detect_question_type(question)
    if not canonical:
        return ["empty"]
    if len(canonical) >= 20:
        flags.append("long_ge_20")
    if len(canonical) > answer_length_limit(question_type):
        flags.append("too_long_for_type")
    if "。" in canonical or "；" in canonical:
        flags.append("sentence_punctuation")
    if any(canonical.startswith(prefix) for prefix in BAD_PREFIXES):
        flags.append("bad_prefix")
    if question_type == "yes_no" and not re.match(r"^(对|不对|是|不是)", canonical):
        flags.append("yes_no_not_normalized")
    if question_type not in {"list", "yes_no"} and ("，" in canonical or "," in canonical):
        flags.append("clause_like")
    if question_type != "yes_no":
        candidate_terms = extract_candidate_terms(canonical)
        unsupported = [term for term in candidate_terms if term not in evidence_text]
        if unsupported and len(unsupported) == len(candidate_terms):
            flags.append("weak_evidence_overlap")
    return flags


def summarize_audit(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    flagged = 0
    for row in rows:
        flags = list(row.get(key) or [])
        if flags:
            flagged += 1
        for flag in flags:
            counts[flag] = counts.get(flag, 0) + 1
    return {
        "row_count": len(rows),
        "flagged_rows": flagged,
        "flag_counts": counts,
    }


def build_messages(
    *,
    row: dict[str, Any],
    question_type: str,
    required_evidence: list[dict[str, str]],
    optional_evidence: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are polishing a Chinese gold-reference answer sheet for a memory benchmark. "
                "Do not change evidence spans or meaning. "
                "Rewrite only into a shorter, benchmark-friendly canonical answer and compact acceptable variants. "
                "Return JSON only. "
                "Avoid long sentences, explanations, narration, and filler. "
                "For yes/no correction questions, use forms like '对' or '不对，<corrected fact>'. "
                "For list questions, return only the compact list items. "
                "Use only facts supported by the provided evidence spans."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question type: {question_type}\n"
                f"Question: {row['question']}\n"
                f"Current canonical_answer: {row.get('canonical_answer') or ''}\n"
                f"Current acceptable_answers: {json.dumps(row.get('acceptable_answers') or [], ensure_ascii=False)}\n"
                f"Current key_facts: {json.dumps(row.get('key_facts') or [], ensure_ascii=False)}\n"
                f"Required spans: {json.dumps(required_evidence, ensure_ascii=False)}\n"
                f"Optional spans: {json.dumps(optional_evidence, ensure_ascii=False)}\n\n"
                "Return JSON with this schema:\n"
                "{\n"
                '  "canonical_answer": "string",\n'
                '  "acceptable_answers": ["string"],\n'
                '  "key_facts": ["string"]\n'
                "}\n\n"
                "Constraints:\n"
                "- canonical_answer must be the shortest directly judgeable answer.\n"
                "- acceptable_answers must be short variants only, max 5.\n"
                "- key_facts must be atomic short facts, max 3.\n"
                "- Do not output full explanatory sentences unless the answer cannot be shortened without losing meaning.\n"
                "- Do not include span ids in the answer.\n"
                "- Keep Chinese natural and concise."
            ),
        },
    ]


def dedupe_texts(items: list[str], limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def polish_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = normalize_text(str(payload.get("canonical_answer") or ""))
    acceptable = dedupe_texts([str(item) for item in (payload.get("acceptable_answers") or [])], limit=5)
    key_facts = dedupe_texts([str(item) for item in (payload.get("key_facts") or [])], limit=3)
    acceptable = [item for item in acceptable if item != canonical]
    return {
        "canonical_answer": canonical,
        "acceptable_answers": acceptable,
        "key_facts": key_facts,
    }


def rewrite_row(
    client: ChatClient,
    *,
    row: dict[str, Any],
    question_type: str,
    required_evidence: list[dict[str, str]],
    optional_evidence: list[dict[str, str]],
    retries: int,
) -> dict[str, Any]:
    last_error = None
    for _ in range(max(1, retries)):
        try:
            text = client.text(
                build_messages(
                    row=row,
                    question_type=question_type,
                    required_evidence=required_evidence,
                    optional_evidence=optional_evidence,
                ),
                max_tokens=384,
                temperature=0.0,
            ).strip()
            return polish_payload(extract_json_object(text))
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    raise RuntimeError(last_error or "rewrite_failed")


def process_rows(
    *,
    rows: list[dict[str, Any]],
    span_indexes: dict[str, dict[str, dict[str, str]]],
    client_factory,
    retries: int,
    max_workers: int,
    rewrite_all: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client_local = threading.local()

    def get_client() -> ChatClient:
        client = getattr(client_local, "client", None)
        if client is None:
            client = client_factory()
            client_local.client = client
        return client

    def process_one(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        updated = dict(row)
        question_text = str(row.get("question") or "")
        persona = str(row["persona"])
        span_index = span_indexes[persona]
        required_ids = [str(item) for item in (row.get("required_spans") or [])]
        optional_ids = [str(item) for item in (row.get("optional_spans") or [])]
        required_evidence = [{"span_id": span_id, "text": span_index.get(span_id, {}).get("combined", "")} for span_id in required_ids]
        optional_evidence = [{"span_id": span_id, "text": span_index.get(span_id, {}).get("combined", "")} for span_id in optional_ids[:4]]
        evidence_text = "\n".join(item["text"] for item in required_evidence if item["text"])
        question_type = detect_question_type(question_text)
        original_canonical = str(row.get("canonical_answer") or "")
        original_acceptable = [str(item) for item in (row.get("acceptable_answers") or [])]
        original_key_facts = [str(item) for item in (row.get("key_facts") or [])]
        shaped_canonical = compress_canonical(question_text, original_canonical)
        shaped_acceptable = dedupe_texts(
            [compress_canonical(question_text, item) for item in original_acceptable],
            limit=5,
        )
        shaped_acceptable = [item for item in shaped_acceptable if item != shaped_canonical]
        shaped_key_facts = dedupe_texts(original_key_facts, limit=3)
        updated["canonical_answer"] = shaped_canonical
        updated["acceptable_answers"] = shaped_acceptable
        updated["key_facts"] = shaped_key_facts
        before_flags = audit_answer(question=question_text, answer=shaped_canonical, evidence_text=evidence_text)
        updated["polish_question_type"] = question_type
        updated["polish_flags_before"] = before_flags
        updated["polish_changed"] = (
            shaped_canonical != normalize_text(original_canonical)
            or shaped_acceptable != dedupe_texts(original_acceptable, limit=5)
            or shaped_key_facts != dedupe_texts(original_key_facts, limit=3)
        )
        updated["polish_status"] = "rule_shaped" if updated["polish_changed"] else "skipped_clean"
        updated["polish_error"] = ""

        if not rewrite_all and not before_flags:
            updated["polish_flags_after"] = before_flags
            return index, updated

        try:
            rewritten = rewrite_row(
                get_client(),
                row=row,
                question_type=question_type,
                required_evidence=required_evidence,
                optional_evidence=optional_evidence,
                retries=retries,
            )
            after_flags = audit_answer(
                question=question_text,
                answer=rewritten["canonical_answer"],
                evidence_text=evidence_text,
            )
            if after_flags and len(after_flags) >= len(before_flags or ["placeholder"]):
                updated["polish_status"] = "fallback_invalid"
                updated["polish_flags_after"] = before_flags
                updated["polish_error"] = ",".join(after_flags)
                return index, updated
            updated["canonical_answer"] = compress_canonical(question_text, rewritten["canonical_answer"])
            updated["acceptable_answers"] = dedupe_texts(
                [compress_canonical(question_text, item) for item in rewritten["acceptable_answers"]],
                limit=5,
            )
            updated["acceptable_answers"] = [
                item for item in updated["acceptable_answers"] if item != updated["canonical_answer"]
            ]
            updated["key_facts"] = rewritten["key_facts"]
            updated["polish_changed"] = True
            updated["polish_status"] = "rewritten"
            updated["polish_flags_after"] = after_flags
            return index, updated
        except Exception as exc:  # noqa: BLE001
            updated["polish_status"] = "fallback_rule_shaped" if updated["polish_changed"] else "error"
            updated["polish_flags_after"] = before_flags
            updated["polish_error"] = str(exc)
            return index, updated

    processed: list[dict[str, Any] | None] = [None] * len(rows)
    if max_workers <= 1 or len(rows) <= 1:
        for index, row in enumerate(rows):
            _, updated = process_one(index, row)
            processed[index] = updated
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(rows))) as executor:
            futures = [executor.submit(process_one, index, row) for index, row in enumerate(rows)]
            for completed, future in enumerate(as_completed(futures), start=1):
                index, updated = future.result()
                processed[index] = updated
                if completed % 10 == 0 or completed == len(futures):
                    print(f"polished {completed}/{len(futures)} rows", flush=True)

    finalized = [row for row in processed if row is not None]
    status_counts: dict[str, int] = {}
    for row in finalized:
        status = str(row.get("polish_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "status_counts": status_counts,
        "audit_before": summarize_audit(finalized, "polish_flags_before"),
        "audit_after": summarize_audit(finalized, "polish_flags_after"),
    }
    return finalized, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polish CN GVD gold answers into shorter benchmark-friendly forms.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--memory-bank", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rewrite-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_json(args.input)
    memory_bank = load_memory_bank(args.memory_bank)
    span_indexes = {persona: build_span_index(payload) for persona, payload in memory_bank.items()}
    rows = list(gold.get("results") or [])
    processed, polish_summary = process_rows(
        rows=rows,
        span_indexes=span_indexes,
        client_factory=lambda: load_config(args.config),
        retries=args.retries,
        max_workers=args.max_workers,
        rewrite_all=bool(args.rewrite_all),
    )
    payload = dict(gold)
    payload["results"] = processed
    payload["polish_summary"] = {
        **polish_summary,
        "source_gold": str(args.input),
        "memory_bank": str(args.memory_bank),
        "rewrite_all": bool(args.rewrite_all),
        "max_workers": int(args.max_workers),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "polish_summary": payload["polish_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
