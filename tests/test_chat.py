import http.server
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from llmquant import chat_engine as ce
from llmquant.gui import MainWindow


def test_list_local_mlx_models(tmp_path: Path):
    m1 = tmp_path / "base-model"
    m1.mkdir()
    (m1 / "config.json").write_text("{}", encoding="utf-8")
    (m1 / "model.safetensors").write_bytes(b"x")

    m2 = tmp_path / "my-MLX-4bit-g64"
    m2.mkdir()
    (m2 / "config.json").write_text("{}", encoding="utf-8")
    (m2 / "model-00001-of-00001.safetensors").write_bytes(b"x")

    empty = tmp_path / "empty-dir"
    empty.mkdir()

    found = ce.list_local_mlx_models(tmp_path)
    assert [p.name for p in found] == ["my-MLX-4bit-g64", "base-model"]


def test_format_chat_prompt():
    class TokWithTemplate:
        chat_template = "tmpl"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    msgs = [{"role": "user", "content": "こんにちは"}]
    assert ce.format_chat_prompt(TokWithTemplate(), msgs) == "user:こんにちは|assistant:"

    class TokNoTemplate:
        chat_template = None

    fallback = ce.format_chat_prompt(TokNoTemplate(), msgs)
    assert "User: こんにちは" in fallback
    assert fallback.endswith("Assistant:")


def test_mlx_chat_session_stream_and_unload(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "tiny-mlx"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("llmquant.tools.mlx_lm_available", lambda: True)

    load_calls = {"n": 0}

    class FakeTok:
        chat_template = None

    import mlx_lm

    def fake_load(path):
        load_calls["n"] += 1
        return object(), FakeTok()

    def fake_stream_generate(model, tokenizer, prompt, max_tokens=256, sampler=None):
        yield SimpleNamespace(text="こん", generation_tokens=1, generation_tps=20.0)
        yield SimpleNamespace(text="にちは", generation_tokens=2, generation_tps=25.0)

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)

    sess = ce.MlxChatSession()
    chunks: list[str] = []
    metrics = sess.stream_chat(model_dir, [{"role": "user", "content": "hi"}], on_chunk=chunks.append)
    assert chunks == ["こん", "にちは"]
    assert metrics.tokens == 2
    assert metrics.tps == 25.0
    assert load_calls["n"] == 1

    # Second call reuses the loaded model
    chunks2: list[str] = []
    sess.stream_chat(model_dir, [{"role": "user", "content": "again"}], on_chunk=chunks2.append)
    assert load_calls["n"] == 1

    sess.unload()
    assert not sess.is_loaded()


def test_stream_ollama_chat_http(monkeypatch):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            assert body["model"] == "gemma:latest"
            assert body["stream"] is True

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            lines = [
                {"message": {"role": "assistant", "content": "Hello "}, "done": False},
                {"message": {"role": "assistant", "content": "world"}, "done": True, "eval_count": 2, "eval_duration": 100_000_000},
            ]
            for obj in lines:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()

        def log_message(self, format, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
        chunks: list[str] = []
        metrics = ce.stream_ollama_chat("gemma:latest", [{"role": "user", "content": "hi"}], on_chunk=chunks.append)
        assert "".join(chunks) == "Hello world"
        assert metrics.tokens == 2
        assert metrics.tps == pytest.approx(20.0, rel=0.05)
    finally:
        srv.shutdown()
        srv.server_close()


def test_gui_chat_tab_controls(monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    monkeypatch.setattr("llmquant.tools.ollama_list_models", lambda max_age=0.0: [])

    w = MainWindow()
    assert w.tabs.count() == 4
    w.tabs.setCurrentIndex(3)
    w.chat_input.setPlainText("テストメッセージ")
    w._clear_chat()
    assert w.chat_view.toPlainText() == ""
