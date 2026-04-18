from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leaf import LEAFService
from leaf.records import MemoryObjectRecord, MemoryObjectVersionRecord, StateCandidate
from leaf.schemas import RawSpan


def _make_embedding(text: str) -> list[float]:
    lowered = text.lower()
    keywords = [
        "psychology",
        "mental health",
        "counseling",
        "dancing",
        "reading",
        "caroline",
        "daniel",
    ]
    vector = [float(lowered.count(token)) for token in keywords]
    vector.append(float(len(text.split())))
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


class _MockHandler(BaseHTTPRequestHandler):
    atom_chat_call_count = 0
    reconcile_chat_call_count = 0

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw or "{}")
        if self.path.endswith("/embeddings"):
            text = str(payload.get("input") or "")
            self._send({"data": [{"embedding": _make_embedding(text)}]})
            return
        if self.path.endswith("/chat/completions"):
            messages = payload.get("messages") or []
            system_text = str(messages[0].get("content") if messages else "")
            user_text = str(messages[-1].get("content") if messages else "")
            if "Allowed values: NONE, PATCH, SUPERSEDE, TENTATIVE" in system_text:
                type(self).reconcile_chat_call_count += 1
                content = '{"action": "NONE"}'
            elif "Extract up to 5 memory atoms" in system_text:
                type(self).atom_chat_call_count += 1
                content = '{"atoms": []}'
                lowered = user_text.lower()
                if "psychology" in lowered:
                    content = (
                        '{"atoms": [{"type": "goal", "content": '
                        '"Caroline plans to study psychology.", '
                        '"entities": ["Caroline", "psychology"], "confidence": 0.8}]}'
                    )
                elif "counseling" in lowered:
                    content = (
                        '{"atoms": [{"type": "goal", "content": '
                        '"Caroline is considering counseling certification programs.", '
                        '"entities": ["Caroline", "counseling certification"], '
                        '"confidence": 0.75}]}'
                    )
            else:
                content = '{"ok": true}'
            self._send({"choices": [{"message": {"content": content}}]})
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class LEAFSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_service_ingest_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.yaml"
            db_path = tmp / "leaf.sqlite3"
            input_path = tmp / "conversation.json"
            base_url = f"http://127.0.0.1:{self.port}/v1"

            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  provider: openai",
                        "  model_name: mock-chat",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "embedding:",
                        "  provider: openai",
                        "  model_name: mock-embed",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "additional_llm:",
                        "  provider: openai",
                        "  model_name: mock-memory",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            input_path.write_text(
                json.dumps(
                    {
                        "turns": [
                            *[
                                {
                                    "session_id": "session-1",
                                    "speaker": "Caroline",
                                    "text": (
                                        "I have been thinking about studying psychology because I want to work in mental health. "
                                        f"This is planning note {index}."
                                    ),
                                    "timestamp": f"2025-01-{10 + index:02d}T10:00:00",
                                }
                                for index in range(7)
                            ],
                            {
                                "session_id": "session-2",
                                "speaker": "Caroline",
                                "text": "I am also looking into counseling certification programs in case I want a more applied path.",
                                "timestamp": "2025-02-03T18:20:00",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            service = LEAFService(config_path=config_path, db_path=db_path)
            try:
                ingest = service.append_json(
                    corpus_id="demo",
                    title="Smoke Demo",
                    path=input_path,
                )
                self.assertEqual(ingest["events_written"], 8)
                self.assertIn("caroline", ingest["touched_subjects"])

                corpora = service.list_corpora()
                self.assertEqual(corpora, ["demo"])

                root = service.get_root_snapshot("demo")
                self.assertIsNotNone(root)
                self.assertEqual(root["snapshot_kind"], "root")

                session = service.get_session_snapshot("demo", "session-1")
                self.assertIsNotNone(session)
                self.assertTrue(session["child_ids"])
                session_block_ids = session["child_ids"]
                seen_kinds = set()
                for block_id in session_block_ids:
                    block = service.store.conn.execute(
                        "select snapshot_kind from leaf_snapshots where snapshot_id = ?",
                        (block_id,),
                    ).fetchone()
                    self.assertIsNotNone(block)
                    seen_kinds.add(str(block["snapshot_kind"]))
                self.assertIn("session_block", seen_kinds)
                self.assertTrue(seen_kinds.issubset({"session_block", "session_page"}))

                retrieval = service.search(
                    corpus_id="demo",
                    question="What is Caroline planning to study?",
                    raw_span_limit=4,
                )
                self.assertTrue(retrieval["pages"])
                self.assertTrue(retrieval["raw_spans"])
                self.assertIn("timing", retrieval)
                allowed_page_kinds = {"root", "session", "session_block", "session_page", "entity", "entity_slot"}
                self.assertTrue(
                    all(page["page_kind"] in allowed_page_kinds for page in retrieval["pages"]),
                    msg=f"Unexpected retrieval page kinds: {retrieval['pages']}",
                )
            finally:
                service.close()

    def test_atom_extraction_disk_cache_reuses_llm_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.yaml"
            db_path = tmp / "leaf.sqlite3"
            cache_dir = tmp / "atom-cache"
            base_url = f"http://127.0.0.1:{self.port}/v1"

            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  provider: openai",
                        "  model_name: mock-chat",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "embedding:",
                        "  provider: openai",
                        "  model_name: mock-embed",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "additional_llm:",
                        "  provider: openai",
                        "  model_name: mock-memory",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            previous_cache_dir = os.environ.get("LEAF_ATOM_CACHE_DIR")
            previous_disable_cache = os.environ.get("LEAF_DISABLE_ATOM_CACHE")
            os.environ["LEAF_ATOM_CACHE_DIR"] = str(cache_dir)
            os.environ.pop("LEAF_DISABLE_ATOM_CACHE", None)
            _MockHandler.atom_chat_call_count = 0
            service = LEAFService(config_path=config_path, db_path=db_path)
            try:
                span = RawSpan(
                    span_id="span-1",
                    corpus_id="demo",
                    session_id="session-1",
                    speaker="Caroline",
                    text="I have been thinking about studying psychology because I want to work in mental health.",
                    turn_index=0,
                    timestamp="2025-01-10T10:00:00",
                    metadata={},
                    embedding=None,
                )
                first = service.indexer.atom_extractor.extract_atoms(span)
                second = service.indexer.atom_extractor.extract_atoms(span)
                self.assertTrue(first)
                self.assertEqual(
                    [(atom.atom_type, atom.content) for atom in first],
                    [(atom.atom_type, atom.content) for atom in second],
                )
                self.assertEqual(_MockHandler.atom_chat_call_count, 1)
                self.assertTrue(any(cache_dir.iterdir()))
            finally:
                service.close()
                if previous_cache_dir is None:
                    os.environ.pop("LEAF_ATOM_CACHE_DIR", None)
                else:
                    os.environ["LEAF_ATOM_CACHE_DIR"] = previous_cache_dir
                if previous_disable_cache is None:
                    os.environ.pop("LEAF_DISABLE_ATOM_CACHE", None)
                else:
                    os.environ["LEAF_DISABLE_ATOM_CACHE"] = previous_disable_cache

    def test_reconcile_disk_cache_reuses_llm_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.yaml"
            db_path = tmp / "leaf.sqlite3"
            cache_dir = tmp / "reconcile-cache"
            base_url = f"http://127.0.0.1:{self.port}/v1"

            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  provider: openai",
                        "  model_name: mock-chat",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "embedding:",
                        "  provider: openai",
                        "  model_name: mock-embed",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "additional_llm:",
                        "  provider: openai",
                        "  model_name: mock-memory",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            previous_cache_dir = os.environ.get("LEAF_RECONCILE_CACHE_DIR")
            previous_disable_cache = os.environ.get("LEAF_DISABLE_RECONCILE_CACHE")
            os.environ["LEAF_RECONCILE_CACHE_DIR"] = str(cache_dir)
            os.environ.pop("LEAF_DISABLE_RECONCILE_CACHE", None)
            _MockHandler.reconcile_chat_call_count = 0
            service = LEAFService(config_path=config_path, db_path=db_path)
            try:
                current = MemoryObjectRecord(
                    object_id="obj-1",
                    corpus_id="demo",
                    subject="caroline",
                    slot="occupation",
                    memory_kind="state",
                    policy="singleton",
                    latest_version_id="ver-1",
                    status="active",
                    aliases=[],
                    canonical_entities=["caroline"],
                    created_at_event_id="evt-1",
                    updated_at_event_id="evt-1",
                    metadata={},
                )
                current_version = MemoryObjectVersionRecord(
                    version_id="ver-1",
                    object_id="obj-1",
                    corpus_id="demo",
                    value="Caroline works in education.",
                    normalized_value="caroline works in education",
                    summary="caroline [occupation]: Caroline works in education.",
                    operation="ADD",
                    status="active",
                    confidence=0.9,
                    valid_from="2025-01-10T10:00:00",
                    valid_to=None,
                    event_id="evt-1",
                    atom_id="atom-1",
                    metadata={},
                )
                candidate = StateCandidate(
                    candidate_id="cand-1",
                    corpus_id="demo",
                    event_id="evt-2",
                    span_id="span-2",
                    atom_id="atom-2",
                    subject="caroline",
                    slot="occupation",
                    value="Caroline works in education and mentoring.",
                    normalized_value="caroline works in education and mentoring",
                    memory_kind="state",
                    policy="singleton",
                    status="active",
                    confidence=0.8,
                    valid_from="2025-01-11T10:00:00",
                    metadata={},
                )
                first = service.indexer._llm_reconcile(
                    candidate=candidate,
                    current=current,
                    current_version=current_version,
                )
                second = service.indexer._llm_reconcile(
                    candidate=candidate,
                    current=current,
                    current_version=current_version,
                )
                self.assertEqual(first, "NONE")
                self.assertEqual(second, "NONE")
                self.assertEqual(_MockHandler.reconcile_chat_call_count, 1)
                self.assertTrue(any(cache_dir.iterdir()))
            finally:
                service.close()
                if previous_cache_dir is None:
                    os.environ.pop("LEAF_RECONCILE_CACHE_DIR", None)
                else:
                    os.environ["LEAF_RECONCILE_CACHE_DIR"] = previous_cache_dir
                if previous_disable_cache is None:
                    os.environ.pop("LEAF_DISABLE_RECONCILE_CACHE", None)
                else:
                    os.environ["LEAF_DISABLE_RECONCILE_CACHE"] = previous_disable_cache

    def test_service_migration_ingest_still_builds_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.yaml"
            db_path = tmp / "leaf.sqlite3"
            base_url = f"http://127.0.0.1:{self.port}/v1"

            config_path.write_text(
                "\n".join(
                    [
                        "llm:",
                        "  provider: openai",
                        "  model_name: mock-chat",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "embedding:",
                        "  provider: openai",
                        "  model_name: mock-embed",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "additional_llm:",
                        "  provider: openai",
                        "  model_name: mock-memory",
                        "  api_key: mock-key",
                        f"  base_url: {base_url}",
                        "",
                        "ingest:",
                        "  mode: migration",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            turns = [
                {
                    "session_id": "session-1",
                    "speaker": "Caroline",
                    "text": "I have been thinking about studying psychology because I want to work in mental health.",
                    "timestamp": "2025-01-10T10:00:00",
                },
                {
                    "session_id": "session-1",
                    "speaker": "AI Companion",
                    "text": "That sounds meaningful. Psychology could fit that goal.",
                    "timestamp": "2025-01-10T10:01:00",
                },
                {
                    "session_id": "session-2",
                    "speaker": "Caroline",
                    "text": "I am also looking into counseling certification programs.",
                    "timestamp": "2025-02-03T18:20:00",
                },
            ]

            service = LEAFService(config_path=config_path, db_path=db_path)
            try:
                ingest = service.append_turns(
                    corpus_id="demo_migration",
                    title="Migration Demo",
                    turns=turns,
                    ingest_mode="migration",
                )
                self.assertEqual(ingest["ingest_mode"], "migration")
                self.assertTrue(ingest["snapshot_refresh_skipped"])
                self.assertIn("migration", ingest)

                root = service.get_root_snapshot("demo_migration")
                self.assertIsNotNone(root)
                self.assertEqual(root["snapshot_kind"], "root")

                session = service.get_session_snapshot("demo_migration", "session-1")
                self.assertIsNotNone(session)
                self.assertTrue(session["child_ids"])
            finally:
                service.close()

    def test_cli_help(self) -> None:
        env = dict(os.environ)
        pythonpath = str(SRC)
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = pythonpath
        result = subprocess.run(
            [sys.executable, "-m", "leaf.cli", "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("LEAF CLI", result.stdout)


if __name__ == "__main__":
    unittest.main()
