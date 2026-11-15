"""Tests for llmquant.benchmark and the Benchmark GUI tab."""

from __future__ import annotations

import base64
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llmquant import benchmark as bm


def test_builtin_vision_png_is_valid():
    for case in bm.builtin_cases():
        if case.category == bm.CATEGORY_VISION:
            assert case.image_b64
            raw = base64.b64decode(case.image_b64)
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_evaluate_case_qa_and_tool_call():
    qa_json = bm.BenchmarkCase(
        id="q1",
        category=bm.CATEGORY_QA,
        title="JSON",
        prompt="...",
        expected={"json_fields": {"name": "佐藤健太", "age": 34}},
    )
    ok, score, _ = bm.evaluate_case(qa_json, '```json\n{"name": "佐藤健太", "age": 34}\n```')
    assert ok and score == 1.0

    ok_bad, score_bad, _ = bm.evaluate_case(qa_json, '{"name": "佐藤健太", "age": 20}')
    assert not ok_bad and score_bad == 0.5

    tc_case = bm.BenchmarkCase(
        id="t1",
        category=bm.CATEGORY_TOOL_CALL,
        title="Weather",
        prompt="...",
        expected={
            "tool_name": "get_weather",
            "required_args": ["city"],
            "arg_regex": {"city": r"(?i)tokyo"},
        },
    )
    # Native Ollama tool_calls
    ok_native, score_native, _ = bm.evaluate_case(
        tc_case,
        "",
        tool_calls=[{"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}],
    )
    assert ok_native and score_native == 1.0

    # Fallback JSON in text
    ok_text, score_text, _ = bm.evaluate_case(
        tc_case,
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
    )
    assert ok_text and score_text == 1.0

    # Missing required arg
    ok_miss, score_miss, _ = bm.evaluate_case(
        tc_case,
        '{"name": "get_weather", "arguments": {}}',
    )
    assert not ok_miss and score_miss == 0.5


def test_expand_sweep_configs():
    temps = bm.parse_number_list("0.0, 0.7", as_int=False)
    tops = bm.parse_number_list("0.9, 1.0", as_int=False)
    maxs = bm.parse_number_list("256", as_int=True)
    grid = bm.expand_sweep_configs(temps, tops, maxs)
    assert len(grid) == 4
    assert grid[0] == bm.BenchmarkParams(temperature=0.0, top_p=0.9, max_tokens=256)
    assert grid[-1] == bm.BenchmarkParams(temperature=0.7, top_p=1.0, max_tokens=256)


def test_load_custom_cases(tmp_path: Path):
    img_file = tmp_path / "sample.png"
    img_file.write_bytes(base64.b64decode(bm._RED_SQUARE_PNG_B64))

    jsonl = tmp_path / "custom.jsonl"
    jsonl.write_text(
        json.dumps({
            "id": "c1",
            "category": "qa",
            "title": "Custom QA",
            "prompt": "1+1?",
            "expected": {"regex": r"\b2\b"},
        })
        + "\n"
        + json.dumps({
            "id": "c2",
            "category": "vision",
            "title": "Custom Vision",
            "prompt": "Color?",
            "image_path": "sample.png",
            "expected": {"keywords_any": ["red", "赤"]},
        })
        + "\n",
        encoding="utf-8",
    )
    cases = bm.load_custom_cases(jsonl)
    assert len(cases) == 2
    assert cases[0].id == "c1"
    assert cases[1].image_b64 is not None


def test_ollama_benchmark_sweep_http(monkeypatch):
    seen_payloads: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            seen_payloads.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            msgs = body.get("messages") or []
            user_msg = msgs[-1] if msgs else {}
            prompt = user_msg.get("content", "")

            if body.get("tools"):
                if "天気" in prompt:
                    tc = {"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
                elif "計算" in prompt:
                    tc = {"function": {"name": "calculate", "arguments": {"expression": "15 * 24 + 8"}}}
                else:
                    tc = {"function": {"name": "search_files", "arguments": {"pattern": "*.py"}}}
                lines = [
                    {"message": {"role": "assistant", "content": "", "tool_calls": [tc]}, "done": False},
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "eval_count": 12,
                        "eval_duration": 200_000_000,
                        "load_duration": 500_000_000,
                    },
                ]
            elif user_msg.get("images"):
                lines = [
                    {"message": {"role": "assistant", "content": "赤色です"}, "done": False},
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "eval_count": 5,
                        "eval_duration": 100_000_000,
                    },
                ]
            else:
                ans = "720 次郎 24 " + '{"name": "佐藤健太", "age": 34} ' + ("Python高速API " * 6)
                lines = [
                    {"message": {"role": "assistant", "content": ans}, "done": False},
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "eval_count": 30,
                        "eval_duration": 300_000_000,
                        "load_duration": 400_000_000,
                    },
                ]

            for obj in lines:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
        runs = bm.run_benchmark_sweep(
            targets=[bm.BenchmarkTarget(backend="ollama", model_id="mock:latest")],
            categories=[bm.CATEGORY_BASIC, bm.CATEGORY_QA, bm.CATEGORY_TOOL_CALL, bm.CATEGORY_VISION],
            param_grid=[bm.BenchmarkParams(temperature=0.0, top_p=0.95, max_tokens=128)],
        )
        assert len(runs) == 1
        r = runs[0]
        assert r.load_time_s == pytest.approx(0.4, abs=0.05)
        assert r.avg_tps and r.avg_tps > 0
        assert r.category_scores[bm.CATEGORY_QA] == 100.0
        assert r.category_scores[bm.CATEGORY_TOOL_CALL] == 100.0
        assert r.category_scores[bm.CATEGORY_VISION] >= 50.0
        assert any(p.get("tools") for p in seen_payloads)
        assert any(p["messages"][-1].get("images") for p in seen_payloads)
    finally:
        srv.shutdown()
        srv.server_close()


def test_mlx_benchmark_skips_vision_and_scores_text(monkeypatch, tmp_path: Path):
    mdir = tmp_path / "mlx-test"
    mdir.mkdir()
    (mdir / "config.json").write_text("{}")
    (mdir / "model.safetensors").write_bytes(b"0")

    class FakeResp:
        def __init__(self, text: str, tokens: int, tps: float):
            self.text = text
            self.generation_tokens = tokens
            self.generation_tps = tps

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda p: ("MODEL", "TOK")
    fake_mlx_lm.stream_generate = lambda m, tok, prompt, max_tokens=128, sampler=None: iter([
        FakeResp("720 次郎 24 ", 10, 50.0),
        FakeResp('{"name": "佐藤健太", "age": 34}', 20, 55.0),
    ])
    fake_sample = types.ModuleType("mlx_lm.sample_utils")
    fake_sample.make_sampler = lambda temp=0.0, top_p=0.95: ("SAMPLER", temp, top_p)
    fake_mx = types.ModuleType("mlx")
    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.clear_cache = lambda: None
    fake_mx.core = fake_mx_core

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample)
    monkeypatch.setitem(sys.modules, "mlx", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)
    monkeypatch.setattr("llmquant.tools.mlx_lm_available", lambda: True)

    cases = [c for c in bm.builtin_cases() if c.category in (bm.CATEGORY_QA, bm.CATEGORY_VISION)]
    run = bm.run_benchmark_for_target(
        target=bm.BenchmarkTarget(backend="mlx", model_id=str(mdir)),
        cases=cases,
        params=bm.BenchmarkParams(temperature=0.0, top_p=0.95, max_tokens=128),
    )
    assert run.category_scores[bm.CATEGORY_QA] == 100.0
    assert run.category_scores[bm.CATEGORY_VISION] is None
    vision_res = [c for c in run.case_results if c.category == bm.CATEGORY_VISION]
    assert all(c.passed is None for c in vision_res)


def test_store_ranking_compare_and_csv(tmp_path: Path):
    store = bm.BenchmarkStore(tmp_path / "history.jsonl")
    r1 = bm.BenchmarkRun(
        run_id="run_a",
        timestamp="2026-11-15T10:00:00+09:00",
        backend="mlx",
        model_id="/tmp/m1",
        model_name="gemma-4-mxfp4",
        params={"temperature": 0.0, "top_p": 0.95, "max_tokens": 512},
        load_time_s=2.5,
        avg_ttft_s=0.3,
        avg_tps=48.0,
        total_elapsed_s=15.0,
        category_scores={"basic": 100.0, "qa": 75.0, "tool_call": 100.0, "vision": None},
        category_tps={"basic": 50.0, "qa": 46.0, "tool_call": 48.0, "vision": None},
        overall_score=87.5,
        case_results=[bm.CaseResult("qa_math", "qa", "Math", True, 1.0, "ok", tps=45.0)],
    )
    r2 = bm.BenchmarkRun(
        run_id="run_b",
        timestamp="2026-11-15T11:00:00+09:00",
        backend="ollama",
        model_id="qwen2.5:7b",
        model_name="qwen2.5:7b",
        params={"temperature": 0.7, "top_p": 0.95, "max_tokens": 512},
        load_time_s=1.2,
        avg_ttft_s=0.2,
        avg_tps=62.0,
        total_elapsed_s=10.0,
        category_scores={"basic": 100.0, "qa": 100.0, "tool_call": 100.0, "vision": None},
        category_tps={"basic": 65.0, "qa": 60.0, "tool_call": 61.0, "vision": None},
        overall_score=100.0,
        case_results=[bm.CaseResult("qa_math", "qa", "Math", True, 1.0, "ok", tps=60.0)],
    )
    store.append(r1)
    store.append(r2)

    loaded = store.load_all()
    assert [r.run_id for r in loaded] == ["run_a", "run_b"]

    ranked_score = bm.rank_runs(loaded, sort_by="overall_score")
    assert ranked_score[0].run_id == "run_b"

    ranked_mlx = bm.rank_runs(loaded, backend_filter="mlx")
    assert len(ranked_mlx) == 1 and ranked_mlx[0].run_id == "run_a"

    cmp_res = bm.compare_runs(r1, r2)
    assert cmp_res["summary"]["overall_score"][2] == pytest.approx(12.5)
    assert cmp_res["summary"]["avg_tps"][2] == pytest.approx(14.0)
    assert cmp_res["summary"]["load_time_s"][2] == pytest.approx(-1.3)

    csv_path = tmp_path / "out.csv"
    store.export_csv(csv_path)
    assert "qwen2.5:7b" in csv_path.read_text(encoding="utf-8")

    assert store.delete_run("run_a")
    assert len(store.load_all()) == 1
    store.clear()
    assert store.load_all() == []


def test_gui_bench_tab_controls(monkeypatch, tmp_path: Path):
    from PySide6.QtWidgets import QApplication

    from llmquant.gui import MainWindow
    from llmquant.tools import OllamaModelItem

    QApplication.instance() or QApplication([])

    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_lm_available", lambda: True)
    monkeypatch.setattr("llmquant.chat_engine.list_local_mlx_models", lambda: [tmp_path / "m-mlx"])
    (tmp_path / "m-mlx").mkdir()
    monkeypatch.setattr(
        "llmquant.tools.ollama_list_models",
        lambda max_age=0.0: [OllamaModelItem(name="llama3:8b", id="1", size="4 GB", modified="now")],
    )

    w = MainWindow()
    w.bench_store = bm.BenchmarkStore(tmp_path / "bench_gui.jsonl")
    w.tabs.setCurrentIndex(4)

    assert w._bench_initialized
    assert w.bench_target_table.rowCount() == 2
    assert w.bench_cat_checks[bm.CATEGORY_BASIC].isChecked()
    assert w.bench_cat_checks[bm.CATEGORY_VISION].isChecked()
    assert w.bench_subtabs.count() == 3
    w.close()
