from __future__ import annotations

import json
import os
import sqlite3
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
                content = '{"action": "NONE"}'
            elif "Extract up to 5 memory atoms" in system_text:
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
                            {
                                "session_id": "session-1",
                                "speaker": "Caroline",
                                "text": "I have been thinking about studying psychology because I want to work in mental health.",
                                "timestamp": "2025-01-10T10:00:00",
                            },
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
                self.assertEqual(ingest["events_written"], 2)
                self.assertIn("caroline", ingest["touched_subjects"])

                corpora = service.list_corpora()
                self.assertEqual(corpora, ["demo"])

                root = service.get_root_snapshot("demo")
                self.assertIsNotNone(root)
                self.assertEqual(root["snapshot_kind"], "root")

                retrieval = service.search(
                    corpus_id="demo",
                    question="What is Caroline planning to study?",
                    raw_span_limit=4,
                )
                self.assertTrue(retrieval["pages"])
                self.assertTrue(retrieval["raw_spans"])
                self.assertIn("timing", retrieval)
            finally:
                service.close()

    def test_online_and_migration_ingest_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config_path = tmp / "config.yaml"
            online_db_path = tmp / "online.sqlite3"
            migration_db_path = tmp / "migration.sqlite3"
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

            turns = [
                {
                    "session_id": "session-1",
                    "speaker": "Caroline",
                    "text": "I live in Boston and work at Acme.",
                    "timestamp": "2025-01-01T10:00:00",
                },
                {
                    "session_id": "session-2",
                    "speaker": "Daniel",
                    "text": "Caroline now lives in Seattle.",
                    "timestamp": "2025-01-02T10:00:00",
                },
                {
                    "session_id": "session-1",
                    "speaker": "Caroline",
                    "text": "I now live in Seattle and still work at Acme.",
                    "timestamp": "2025-01-03T10:00:00",
                },
            ]

            online_service = LEAFService(config_path=config_path, db_path=online_db_path)
            migration_service = LEAFService(config_path=config_path, db_path=migration_db_path)
            try:
                online_result = online_service.append_turns_online(
                    corpus_id="demo",
                    title="Mode Match",
                    turns=turns,
                )
                migration_result = migration_service.migrate_turns(
                    corpus_id="demo",
                    title="Mode Match",
                    turns=turns,
                )
                self.assertEqual(online_result["ingest_mode"], "online")
                self.assertEqual(migration_result["ingest_mode"], "migration")
                self.assertEqual(migration_result["apply_strategy"], "state_cache_serial")
                self.assertTrue(migration_result["state_cache_metrics"]["enabled"])
            finally:
                online_service.close()
                migration_service.close()

            self.assertEqual(
                self._dump_core_tables(online_db_path),
                self._dump_core_tables(migration_db_path),
            )

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

    @staticmethod
    def _dump_core_tables(path: Path) -> dict[str, list[dict[str, object]]]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            tables = [
                "leaf_events",
                "leaf_atoms",
                "leaf_objects",
                "leaf_object_versions",
                "leaf_evidence_links",
                "leaf_snapshots",
            ]
            dumped: dict[str, list[dict[str, object]]] = {}
            for table in tables:
                rows = conn.execute(f"select * from {table} order by 1").fetchall()
                dumped[table] = [dict(row) for row in rows]
            return dumped
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
