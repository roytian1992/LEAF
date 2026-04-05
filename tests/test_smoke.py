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
