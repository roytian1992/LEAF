from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf import LEAFService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal incremental LEAF chat-loop example")
    parser.add_argument("--config", default="examples/config.yaml")
    parser.add_argument("--db", default="data/chat_loop.sqlite3")
    parser.add_argument("--corpus-id", default="demo-chat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = LEAFService(config_path=args.config, db_path=args.db)

    conversation = [
        {
            "session_id": "session-1",
            "speaker": "Caroline",
            "text": "I am considering psychology because I want to work in mental health.",
            "timestamp": "2025-01-10T10:00:00",
        },
        {
            "session_id": "session-1",
            "speaker": "Daniel",
            "text": "That makes sense. You are usually the one helping friends through stressful moments.",
            "timestamp": "2025-01-10T10:01:00",
        },
        {
            "session_id": "session-2",
            "speaker": "Caroline",
            "text": "I am also looking at counseling certification programs as a practical path.",
            "timestamp": "2025-02-03T18:20:00",
        },
    ]

    try:
        for turn in conversation:
            result = service.append_turns(
                corpus_id=args.corpus_id,
                title="Incremental Chat Loop",
                turns=[turn],
            )
            print(json.dumps({"ingested_turn": turn["text"], "result": result}, ensure_ascii=False, indent=2))

        question = "What is Caroline planning to study or pursue?"
        retrieved = service.search(
            corpus_id=args.corpus_id,
            question=question,
            raw_span_limit=6,
        )
        print(json.dumps({"question": question, "retrieval": retrieved}, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
