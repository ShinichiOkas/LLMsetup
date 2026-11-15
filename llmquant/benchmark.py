"""Benchmark suite for MLX and Ollama models.

Measures load time, TTFT, throughput (tokens/s), QA accuracy, tool-calling
accuracy, and vision (VLM) accuracy across parameter sweeps, and persists
results for historical comparison and ranking.
"""

from __future__ import annotations

import base64
import csv
import itertools
import json
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import chat_engine, tools

CATEGORY_BASIC = "basic"
CATEGORY_QA = "qa"
CATEGORY_TOOL_CALL = "tool_call"
CATEGORY_VISION = "vision"

ALL_CATEGORIES = (CATEGORY_BASIC, CATEGORY_QA, CATEGORY_TOOL_CALL, CATEGORY_VISION)

CATEGORY_LABELS = {
    CATEGORY_BASIC: "基礎速度",
    CATEGORY_QA: "質問応答",
    CATEGORY_TOOL_CALL: "ツールコール",
    CATEGORY_VISION: "画像理解",
}


def _make_solid_png_b64(width: int, height: int, rgb: tuple[int, int, int]) -> str:
    """Generates a minimal valid RGB PNG encoded in base64 using only stdlib."""
    r, g, b = rgb
    row = b"\x00" + bytes([r, g, b]) * width
    raw = row * height

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


# Pre-generated tiny solid-color PNGs for deterministic VLM benchmarks
_RED_SQUARE_PNG_B64 = _make_solid_png_b64(64, 64, (220, 30, 30))
_BLUE_SQUARE_PNG_B64 = _make_solid_png_b64(64, 64, (30, 90, 220))


@dataclass
class BenchmarkCase:
    id: str
    category: str
    title: str
    prompt: str
    system_prompt: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    image_b64: str | None = None
    image_path: str | None = None
    expected: dict[str, Any] = field(default_factory=dict)
    max_tokens_override: int | None = None


@dataclass
class CaseResult:
    case_id: str
    category: str
    title: str
    passed: bool | None  # None means skipped (e.g. VLM unsupported on backend)
    score: float  # 0.0 - 1.0 (0.0 when skipped)
    reason: str
    ttft_s: float | None = None
    total_s: float = 0.0
    tokens: int = 0
    tps: float | None = None
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkParams:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass
class BenchmarkTarget:
    backend: str  # "mlx" | "ollama"
    model_id: str  # path for mlx, model:tag for ollama
    model_name: str = ""

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = (
                Path(self.model_id).name if self.backend == "mlx" else self.model_id
            )


@dataclass
class BenchmarkRun:
    run_id: str
    timestamp: str
    backend: str
    model_id: str
    model_name: str
    params: dict[str, Any]
    load_time_s: float | None
    avg_ttft_s: float | None
    avg_tps: float | None
    total_elapsed_s: float
    category_scores: dict[str, float | None]
    category_tps: dict[str, float | None]
    overall_score: float | None
    case_results: list[CaseResult]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "backend": self.backend,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "params": self.params,
            "load_time_s": self.load_time_s,
            "avg_ttft_s": self.avg_ttft_s,
            "avg_tps": self.avg_tps,
            "total_elapsed_s": self.total_elapsed_s,
            "category_scores": self.category_scores,
            "category_tps": self.category_tps,
            "overall_score": self.overall_score,
            "case_results": [asdict(c) for c in self.case_results],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRun:
        cases = [CaseResult(**c) for c in data.get("case_results", [])]
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            backend=data["backend"],
            model_id=data["model_id"],
            model_name=data.get("model_name") or data["model_id"],
            params=dict(data.get("params") or {}),
            load_time_s=data.get("load_time_s"),
            avg_ttft_s=data.get("avg_ttft_s"),
            avg_tps=data.get("avg_tps"),
            total_elapsed_s=float(data.get("total_elapsed_s") or 0.0),
            category_scores=dict(data.get("category_scores") or {}),
            category_tps=dict(data.get("category_tps") or {}),
            overall_score=data.get("overall_score"),
            case_results=cases,
            note=str(data.get("note") or ""),
        )


def builtin_cases() -> list[BenchmarkCase]:
    """Returns the built-in deterministic benchmark suite."""
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定した都市の現在の天気を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "都市名 (例: Tokyo)"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
    calc_tool = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数式を計算して結果を返す",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "計算式"},
                },
                "required": ["expression"],
            },
        },
    }
    search_tool = {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "ワークスペース内のファイルを検索する",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "検索パターン"},
                    "directory": {"type": "string", "description": "対象ディレクトリ"},
                },
                "required": ["pattern"],
            },
        },
    }

    return [
        # --- Basic speed ---
        BenchmarkCase(
            id="basic_short",
            category=CATEGORY_BASIC,
            title="短文応答速度",
            prompt="「準備完了」とだけ短く返答してください。",
            expected={"min_chars": 1},
            max_tokens_override=64,
        ),
        BenchmarkCase(
            id="basic_throughput",
            category=CATEGORY_BASIC,
            title="長文スループット",
            prompt="Pythonで高速なAPIサーバを作る際のポイントを5つ、箇条書きで簡潔に説明してください。",
            expected={"min_chars": 40},
        ),
        # --- QA accuracy ---
        BenchmarkCase(
            id="qa_math",
            category=CATEGORY_QA,
            title="四則演算・文章題",
            prompt="りんごを1個120円で4個、みかんを1個80円で3個買いました。合計金額はいくらですか？数字（単位なしまたは円）で答えてください。",
            expected={"regex": r"(720|７２０)"},
            max_tokens_override=128,
        ),
        BenchmarkCase(
            id="qa_logic",
            category=CATEGORY_QA,
            title="論理推論",
            prompt="太郎は花子より背が高い。花子は次郎より背が高い。一番背が低いのは誰ですか？名前だけ答えてください。",
            expected={"keywords_any": ["次郎"]},
            max_tokens_override=64,
        ),
        BenchmarkCase(
            id="qa_extract",
            category=CATEGORY_QA,
            title="情報抽出・JSON出力",
            prompt='次の文から名前と年齢を抽出し、{"name": "...", "age": 数値} のJSONだけを出力してください。\n文: 「佐藤健太さんは今年34歳になりました。」',
            expected={"json_fields": {"name": "佐藤健太", "age": 34}},
            max_tokens_override=128,
        ),
        BenchmarkCase(
            id="qa_code",
            category=CATEGORY_QA,
            title="コード読解",
            prompt="次のPythonコードの出力結果は何ですか？\n```python\nnums = [2, 4, 6]\nprint(sum(x * 2 for x in nums))\n```\n数値のみ答えてください。",
            expected={"regex": r"\b24\b"},
            max_tokens_override=128,
        ),
        # --- Tool call accuracy ---
        BenchmarkCase(
            id="tool_weather",
            category=CATEGORY_TOOL_CALL,
            title="天気ツール呼出",
            prompt="東京の現在の天気を摂氏で調べてください。",
            tools=[weather_tool, calc_tool],
            expected={
                "tool_name": "get_weather",
                "required_args": ["city"],
                "arg_regex": {"city": r"(?i)(tokyo|東京)"},
            },
            max_tokens_override=256,
        ),
        BenchmarkCase(
            id="tool_calc",
            category=CATEGORY_TOOL_CALL,
            title="計算ツール呼出",
            prompt="ツールを使って 15 * 24 + 8 を計算してください。",
            tools=[weather_tool, calc_tool],
            expected={
                "tool_name": "calculate",
                "required_args": ["expression"],
                "arg_regex": {"expression": r"15.*24.*8"},
            },
            max_tokens_override=256,
        ),
        BenchmarkCase(
            id="tool_search",
            category=CATEGORY_TOOL_CALL,
            title="ファイル検索ツール呼出",
            prompt="src ディレクトリから *.py ファイルを検索してください。",
            tools=[weather_tool, search_tool],
            expected={
                "tool_name": "search_files",
                "required_args": ["pattern"],
                "arg_regex": {"pattern": r"\*\.py"},
            },
            max_tokens_override=256,
        ),
        # --- Vision accuracy ---
        BenchmarkCase(
            id="vision_red",
            category=CATEGORY_VISION,
            title="色認識（赤）",
            prompt="この画像全体は何色ですか？日本語または英語で色名を答えてください。",
            image_b64=_RED_SQUARE_PNG_B64,
            expected={"keywords_any": ["赤", "レッド", "red", "Red", "RED"]},
            max_tokens_override=128,
        ),
        BenchmarkCase(
            id="vision_blue",
            category=CATEGORY_VISION,
            title="色認識（青）",
            prompt="この画像全体は何色ですか？日本語または英語で色名を答えてください。",
            image_b64=_BLUE_SQUARE_PNG_B64,
            expected={"keywords_any": ["青", "ブルー", "blue", "Blue", "BLUE"]},
            max_tokens_override=128,
        ),
    ]


def load_custom_cases(path: Path) -> list[BenchmarkCase]:
    """Loads custom benchmark cases from a JSON or JSONL file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        items = parsed if isinstance(parsed, list) else parsed.get("cases", [])

    cases: list[BenchmarkCase] = []
    for idx, item in enumerate(items):
        img_b64 = item.get("image_b64")
        img_path = item.get("image_path")
        if not img_b64 and img_path:
            p = Path(img_path)
            if not p.is_absolute():
                p = path.parent / p
            if p.is_file():
                img_b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        cases.append(
            BenchmarkCase(
                id=str(item.get("id") or f"custom_{idx + 1}"),
                category=str(item.get("category") or CATEGORY_QA),
                title=str(item.get("title") or item.get("id") or f"Custom {idx + 1}"),
                prompt=str(item["prompt"]),
                system_prompt=str(item.get("system_prompt") or ""),
                tools=list(item.get("tools") or []),
                image_b64=img_b64,
                image_path=img_path,
                expected=dict(item.get("expected") or {}),
                max_tokens_override=item.get("max_tokens_override"),
            )
        )
    return cases


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extracts JSON objects from raw response text or fenced blocks."""
    objs: list[dict[str, Any]] = []
    # Strip <tool_call>...</tool_call> wrappers if present
    candidates: list[str] = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        candidates.append(m.group(1))
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        candidates.append(m.group(1))
    candidates.append(text.strip())

    decoder = json.JSONDecoder()
    for cand in candidates:
        pos = 0
        while pos < len(cand):
            brace = cand.find("{", pos)
            if brace < 0:
                break
            try:
                obj, end = decoder.raw_decode(cand, brace)
                if isinstance(obj, dict):
                    objs.append(obj)
                pos = end
            except json.JSONDecodeError:
                pos = brace + 1
    return objs


def normalize_tool_calls(
    native_tool_calls: list[dict[str, Any]] | None,
    response_text: str,
) -> list[dict[str, Any]]:
    """Normalizes native Ollama tool_calls or fallback JSON tool calls from text."""
    normalized: list[dict[str, Any]] = []
    for tc in native_tool_calls or []:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = str(fn.get("name") or "")
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {}
        if name:
            normalized.append({"name": name, "arguments": args})

    if normalized:
        return normalized

    # Fallback: parse JSON objects in response text
    for obj in _extract_json_objects(response_text):
        if "function" in obj and isinstance(obj["function"], dict):
            fn = obj["function"]
            name = str(fn.get("name") or "")
            args = fn.get("arguments") or {}
        else:
            name = str(obj.get("name") or obj.get("tool") or "")
            args = obj.get("arguments") if "arguments" in obj else obj.get("parameters", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if name and isinstance(args, dict):
            normalized.append({"name": name, "arguments": args})
    return normalized


def evaluate_case(
    case: BenchmarkCase,
    response_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[bool, float, str]:
    """Deterministic rule-based scoring returning (passed, score_0_to_1, reason)."""
    exp = case.expected or {}
    text = (response_text or "").strip()

    if case.category == CATEGORY_TOOL_CALL:
        calls = normalize_tool_calls(tool_calls, text)
        target_name = exp.get("tool_name")
        if not calls:
            return False, 0.0, "ツール呼び出しが検出されませんでした"
        matched = next((c for c in calls if c["name"] == target_name), None)
        if matched is None:
            got = ", ".join(c["name"] for c in calls)
            return False, 0.25, f"関数名不一致 (期待: {target_name}, 実際: {got})"
        args = matched.get("arguments") or {}
        req_args = list(exp.get("required_args") or [])
        missing = [k for k in req_args if k not in args or args[k] in (None, "")]
        if missing:
            return False, 0.5, f"必須引数不足: {', '.join(missing)}"
        arg_regex = dict(exp.get("arg_regex") or {})
        for k, pat in arg_regex.items():
            val = str(args.get(k, ""))
            if not re.search(pat, val):
                return False, 0.7, f"引数 {k}={val!r} が期待パターン /{pat}/ と不一致"
        return True, 1.0, f"一致 ({target_name}: {json.dumps(args, ensure_ascii=False)})"

    if not text:
        return False, 0.0, "応答が空です"

    if "min_chars" in exp:
        need = int(exp["min_chars"])
        if len(text) >= need:
            return True, 1.0, f"出力文字数 OK ({len(text)}文字)"
        return False, max(0.0, len(text) / max(1, need)), f"出力が短すぎます ({len(text)}/{need})"

    if "exact" in exp:
        ok = text == str(exp["exact"]).strip()
        return ok, (1.0 if ok else 0.0), ("完全一致" if ok else f"不一致: {text[:60]}")

    if "regex" in exp:
        ok = bool(re.search(str(exp["regex"]), text))
        return ok, (1.0 if ok else 0.0), ("正規表現一致" if ok else f"パターン不一致: {text[:80]}")

    if "keywords_all" in exp:
        kws = [str(k) for k in exp["keywords_all"]]
        hits = sum(1 for k in kws if k.lower() in text.lower())
        score = hits / max(1, len(kws))
        ok = hits == len(kws)
        return ok, score, f"キーワード {hits}/{len(kws)} 一致"

    if "keywords_any" in exp:
        kws = [str(k) for k in exp["keywords_any"]]
        hit = next((k for k in kws if k.lower() in text.lower()), None)
        if hit is not None:
            return True, 1.0, f"一致キーワード: {hit}"
        return False, 0.0, f"期待キーワードなし ({', '.join(kws)}): {text[:80]}"

    if "json_fields" in exp:
        want = dict(exp["json_fields"])
        objs = _extract_json_objects(text)
        if not objs:
            return False, 0.0, "JSONオブジェクトを抽出できませんでした"
        best_score = 0.0
        for obj in objs:
            matched = sum(1 for k, v in want.items() if obj.get(k) == v)
            best_score = max(best_score, matched / max(1, len(want)))
            if matched == len(want):
                return True, 1.0, "JSONフィールド全一致"
        return False, best_score, f"JSONフィールド一部不一致 (score={best_score:.2f})"

    return True, 1.0, "応答あり"


def parse_number_list(text: str, *, as_int: bool = False, default: list[Any] | None = None) -> list[Any]:
    """Parses comma-separated numeric values (e.g. '0.0, 0.3, 0.7')."""
    vals: list[Any] = []
    for raw in (text or "").split(","):
        s = raw.strip()
        if not s:
            continue
        vals.append(int(s) if as_int else float(s))
    if not vals and default is not None:
        return list(default)
    return vals


def expand_sweep_configs(
    temperatures: list[float],
    top_ps: list[float],
    max_tokens_list: list[int],
) -> list[BenchmarkParams]:
    """Generates cartesian product of inference parameter sweep values."""
    temps = temperatures or [0.0]
    tops = top_ps or [0.95]
    maxs = max_tokens_list or [512]
    return [
        BenchmarkParams(temperature=t, top_p=p, max_tokens=m)
        for t, p, m in itertools.product(temps, tops, maxs)
    ]


def _run_ollama_case(
    model_name: str,
    case: BenchmarkCase,
    params: BenchmarkParams,
    cancel_event: threading.Event | None = None,
) -> tuple[CaseResult, float | None]:
    """Executes one benchmark case against Ollama /api/chat (streaming)."""
    base = chat_engine.ollama_base_url()
    url = f"{base}/api/chat"

    messages: list[dict[str, Any]] = []
    if case.system_prompt:
        messages.append({"role": "system", "content": case.system_prompt})
    user_msg: dict[str, Any] = {"role": "user", "content": case.prompt}
    if case.image_b64:
        user_msg["images"] = [case.image_b64]
    messages.append(user_msg)

    max_tok = case.max_tokens_override or params.max_tokens
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_predict": max_tok,
        },
    }
    if case.tools:
        payload["tools"] = case.tools

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    first_token_t: float | None = None
    chunks: list[str] = []
    native_tools: list[dict[str, Any]] = []
    eval_count = 0
    eval_duration_ns = 0
    load_duration_ns = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("キャンセルされました")
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                msg = obj.get("message") or {}
                delta = msg.get("content") or ""
                if delta:
                    if first_token_t is None:
                        first_token_t = time.monotonic()
                    chunks.append(delta)
                if msg.get("tool_calls"):
                    if first_token_t is None:
                        first_token_t = time.monotonic()
                    native_tools.extend(msg["tool_calls"])
                if obj.get("done"):
                    eval_count = int(obj.get("eval_count") or 0)
                    eval_duration_ns = int(obj.get("eval_duration") or 0)
                    load_duration_ns = int(obj.get("load_duration") or 0)
                    break
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return (
            CaseResult(
                case_id=case.id,
                category=case.category,
                title=case.title,
                passed=False,
                score=0.0,
                reason=f"HTTP {e.code}: {body[:120]}",
                total_s=time.monotonic() - t0,
                error=body[:200],
            ),
            None,
        )
    except Exception as e:
        return (
            CaseResult(
                case_id=case.id,
                category=case.category,
                title=case.title,
                passed=False,
                score=0.0,
                reason=str(e),
                total_s=time.monotonic() - t0,
                error=str(e),
            ),
            None,
        )

    total_s = max(1e-6, time.monotonic() - t0)
    ttft_s = (first_token_t - t0) if first_token_t is not None else total_s
    text = "".join(chunks)
    if eval_count <= 0:
        eval_count = max(1, len(text) // 4) if text else 0
    if eval_duration_ns > 0 and eval_count > 0:
        tps = eval_count / (eval_duration_ns / 1e9)
    else:
        gen_s = max(1e-4, time.monotonic() - (first_token_t or t0))
        tps = eval_count / gen_s if eval_count > 0 else 0.0

    load_s = (load_duration_ns / 1e9) if load_duration_ns > 0 else None
    passed, score, reason = evaluate_case(case, text, native_tools)
    norm_calls = normalize_tool_calls(native_tools, text) if case.category == CATEGORY_TOOL_CALL else []

    return (
        CaseResult(
            case_id=case.id,
            category=case.category,
            title=case.title,
            passed=passed,
            score=score,
            reason=reason,
            ttft_s=round(ttft_s, 4),
            total_s=round(total_s, 4),
            tokens=eval_count,
            tps=round(tps, 2),
            response_text=text,
            tool_calls=norm_calls,
        ),
        load_s,
    )


def _format_mlx_prompt(tokenizer: object, case: BenchmarkCase) -> str:
    messages: list[dict[str, str]] = []
    sys_parts: list[str] = []
    if case.system_prompt:
        sys_parts.append(case.system_prompt)
    if case.tools:
        tools_json = json.dumps(case.tools, ensure_ascii=False)
        sys_parts.append(
            "利用可能なツール定義(JSON):\n"
            f"{tools_json}\n"
            'ツールを呼ぶ場合は {"name": "関数名", "arguments": {...}} のJSON形式で出力してください。'
        )
    if sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
    messages.append({"role": "user", "content": case.prompt})

    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    has_tmpl = bool(getattr(tokenizer, "chat_template", None))
    if callable(apply_fn) and has_tmpl and case.tools:
        try:
            out = apply_fn(
                messages,
                tools=case.tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(out, str):
                return out
        except Exception:
            pass
    return chat_engine.format_chat_prompt(tokenizer, messages)


def _run_mlx_case(
    model: object,
    tokenizer: object,
    case: BenchmarkCase,
    params: BenchmarkParams,
    cancel_event: threading.Event | None = None,
) -> CaseResult:
    """Executes one benchmark case against a loaded mlx-lm model."""
    if case.category == CATEGORY_VISION or case.image_b64:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            title=case.title,
            passed=None,
            score=0.0,
            reason="スキップ: mlx-vlm 未導入のため MLX 画像理解はスキップされました",
        )

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    prompt = _format_mlx_prompt(tokenizer, case)
    max_tok = case.max_tokens_override or params.max_tokens
    sampler = make_sampler(temp=float(params.temperature), top_p=float(params.top_p))

    t0 = time.monotonic()
    first_token_t: float | None = None
    chunks: list[str] = []
    tokens = 0
    last_tps: float | None = None

    try:
        for resp in mlx_lm.stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=int(max_tok),
            sampler=sampler,
        ):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("キャンセルされました")
            text_part = getattr(resp, "text", "") or ""
            if text_part:
                if first_token_t is None:
                    first_token_t = time.monotonic()
                chunks.append(text_part)
            tokens = getattr(resp, "generation_tokens", tokens + 1)
            gen_tps = getattr(resp, "generation_tps", None)
            if gen_tps is not None:
                last_tps = float(gen_tps)
    except Exception as e:
        return CaseResult(
            case_id=case.id,
            category=case.category,
            title=case.title,
            passed=False,
            score=0.0,
            reason=str(e),
            total_s=time.monotonic() - t0,
            error=str(e),
        )

    total_s = max(1e-6, time.monotonic() - t0)
    ttft_s = (first_token_t - t0) if first_token_t is not None else total_s
    text = "".join(chunks)
    if last_tps is None and tokens > 0:
        gen_s = max(1e-4, time.monotonic() - (first_token_t or t0))
        last_tps = tokens / gen_s

    passed, score, reason = evaluate_case(case, text)
    norm_calls = normalize_tool_calls(None, text) if case.category == CATEGORY_TOOL_CALL else []

    return CaseResult(
        case_id=case.id,
        category=case.category,
        title=case.title,
        passed=passed,
        score=score,
        reason=reason,
        ttft_s=round(ttft_s, 4),
        total_s=round(total_s, 4),
        tokens=int(tokens),
        tps=round(last_tps, 2) if last_tps is not None else None,
        response_text=text,
        tool_calls=norm_calls,
    )


def _aggregate_run(
    target: BenchmarkTarget,
    params: BenchmarkParams,
    load_time_s: float | None,
    case_results: list[CaseResult],
    total_elapsed_s: float,
    note: str = "",
) -> BenchmarkRun:
    ttfts = [c.ttft_s for c in case_results if c.ttft_s is not None and c.passed is not None]
    tpss = [c.tps for c in case_results if c.tps is not None and c.passed is not None]

    cat_scores: dict[str, float | None] = {}
    cat_tps: dict[str, float | None] = {}
    for cat in ALL_CATEGORIES:
        cat_items = [c for c in case_results if c.category == cat and c.passed is not None]
        if cat_items:
            cat_scores[cat] = round(100.0 * sum(c.score for c in cat_items) / len(cat_items), 1)
            cat_t = [c.tps for c in cat_items if c.tps is not None]
            cat_tps[cat] = round(sum(cat_t) / len(cat_t), 2) if cat_t else None
        else:
            cat_scores[cat] = None
            cat_tps[cat] = None

    # Overall accuracy score across evaluated accuracy categories (qa, tool_call, vision),
    # falling back to all evaluated categories if only basic was selected.
    acc_cats = [c for c in (CATEGORY_QA, CATEGORY_TOOL_CALL, CATEGORY_VISION) if cat_scores[c] is not None]
    if acc_cats:
        overall = round(sum(cat_scores[c] for c in acc_cats if cat_scores[c] is not None) / len(acc_cats), 1)
    else:
        any_cats = [c for c in ALL_CATEGORIES if cat_scores[c] is not None]
        overall = (
            round(sum(cat_scores[c] for c in any_cats if cat_scores[c] is not None) / len(any_cats), 1)
            if any_cats
            else None
        )

    return BenchmarkRun(
        run_id=uuid.uuid4().hex[:10],
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        backend=target.backend,
        model_id=target.model_id,
        model_name=target.model_name,
        params=asdict(params),
        load_time_s=round(load_time_s, 3) if load_time_s is not None else None,
        avg_ttft_s=round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        avg_tps=round(sum(tpss) / len(tpss), 2) if tpss else None,
        total_elapsed_s=round(total_elapsed_s, 2),
        category_scores=cat_scores,
        category_tps=cat_tps,
        overall_score=overall,
        case_results=case_results,
        note=note,
    )


def run_benchmark_for_target(
    target: BenchmarkTarget,
    cases: list[BenchmarkCase],
    params: BenchmarkParams,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_case_done: Callable[[CaseResult], None] | None = None,
) -> BenchmarkRun:
    """Runs all selected cases for a single (target, params) pair."""
    t_start = time.monotonic()
    results: list[CaseResult] = []
    load_time_s: float | None = None

    if target.backend == "mlx":
        if not tools.mlx_lm_available():
            raise RuntimeError("mlx-lm が利用できません（Apple Silicon Mac 環境が必要です）")
        session = chat_engine.MlxChatSession()
        if on_progress:
            on_progress(f"[MLX] モデル読み込み中: {target.model_name} ...")
        t_load = time.monotonic()
        session.ensure_loaded(Path(target.model_id))
        load_time_s = time.monotonic() - t_load
        if on_progress:
            on_progress(f"[MLX] 読み込み完了 ({load_time_s:.2f}s)")
        try:
            for idx, case in enumerate(cases, 1):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("キャンセルされました")
                if on_progress:
                    on_progress(f"[{idx}/{len(cases)}] {target.model_name}: {case.title}")
                res = _run_mlx_case(session.model, session.tokenizer, case, params, cancel_event)
                results.append(res)
                if on_case_done:
                    on_case_done(res)
        finally:
            session.unload()
    elif target.backend == "ollama":
        for idx, case in enumerate(cases, 1):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("キャンセルされました")
            if on_progress:
                on_progress(f"[{idx}/{len(cases)}] {target.model_name}: {case.title}")
            res, maybe_load = _run_ollama_case(target.model_id, case, params, cancel_event)
            if load_time_s is None and maybe_load is not None:
                load_time_s = maybe_load
            results.append(res)
            if on_case_done:
                on_case_done(res)
    else:
        raise ValueError(f"未知のバックエンドです: {target.backend}")

    return _aggregate_run(
        target=target,
        params=params,
        load_time_s=load_time_s,
        case_results=results,
        total_elapsed_s=time.monotonic() - t_start,
    )


def run_benchmark_sweep(
    targets: list[BenchmarkTarget],
    categories: list[str],
    param_grid: list[BenchmarkParams],
    custom_cases: list[BenchmarkCase] | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_run_done: Callable[[BenchmarkRun], None] | None = None,
    on_case_done: Callable[[CaseResult], None] | None = None,
) -> list[BenchmarkRun]:
    """Runs benchmarks across all targets and parameter combinations."""
    all_cases = builtin_cases() + list(custom_cases or [])
    selected_cats = set(categories or ALL_CATEGORIES)
    cases = [c for c in all_cases if c.category in selected_cats]
    if not cases:
        raise ValueError("実行するベンチマークケースがありません")

    runs: list[BenchmarkRun] = []
    total_jobs = len(targets) * len(param_grid)
    job_idx = 0

    for target in targets:
        for params in param_grid:
            if cancel_event and cancel_event.is_set():
                break
            job_idx += 1
            if on_progress:
                on_progress(
                    f"=== ジョブ {job_idx}/{total_jobs}: {target.backend}:{target.model_name} "
                    f"(temp={params.temperature}, top_p={params.top_p}, max={params.max_tokens}) ==="
                )
            run = run_benchmark_for_target(
                target=target,
                cases=cases,
                params=params,
                cancel_event=cancel_event,
                on_progress=on_progress,
                on_case_done=on_case_done,
            )
            runs.append(run)
            if on_run_done:
                on_run_done(run)
    return runs


class BenchmarkStore:
    """JSONL persistence, ranking, comparison, and CSV export for benchmark runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("benchmarks/history.jsonl")

    def append(self, run: BenchmarkRun) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> list[BenchmarkRun]:
        if not self.path.is_file():
            return []
        runs: list[BenchmarkRun] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                runs.append(BenchmarkRun.from_dict(json.loads(s)))
            except Exception:
                continue
        return runs

    def delete_run(self, run_id: str) -> bool:
        runs = self.load_all()
        kept = [r for r in runs if r.run_id != run_id]
        if len(kept) == len(runs):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        return True

    def clear(self) -> None:
        if self.path.is_file():
            self.path.unlink()

    def export_csv(self, dest: Path, runs: list[BenchmarkRun] | None = None) -> None:
        items = runs if runs is not None else self.load_all()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "run_id",
                "timestamp",
                "backend",
                "model_name",
                "temperature",
                "top_p",
                "max_tokens",
                "overall_score",
                "load_time_s",
                "avg_ttft_s",
                "avg_tps",
                "qa_score",
                "tool_call_score",
                "vision_score",
            ])
            for r in items:
                w.writerow([
                    r.run_id,
                    r.timestamp,
                    r.backend,
                    r.model_name,
                    r.params.get("temperature"),
                    r.params.get("top_p"),
                    r.params.get("max_tokens"),
                    r.overall_score,
                    r.load_time_s,
                    r.avg_ttft_s,
                    r.avg_tps,
                    r.category_scores.get(CATEGORY_QA),
                    r.category_scores.get(CATEGORY_TOOL_CALL),
                    r.category_scores.get(CATEGORY_VISION),
                ])


def rank_runs(
    runs: list[BenchmarkRun],
    sort_by: str = "overall_score",
    backend_filter: str | None = None,
) -> list[BenchmarkRun]:
    """Ranks benchmark runs by the requested metric."""
    items = [r for r in runs if not backend_filter or r.backend == backend_filter]

    def _key(r: BenchmarkRun) -> tuple[int, float]:
        if sort_by == "load_time_s":
            # Smaller is better
            return (0, r.load_time_s) if r.load_time_s is not None else (1, 1e9)
        if sort_by == "avg_ttft_s":
            return (0, r.avg_ttft_s) if r.avg_ttft_s is not None else (1, 1e9)
        if sort_by == "avg_tps":
            return (0, -(r.avg_tps or 0.0)) if r.avg_tps is not None else (1, 0.0)
        if sort_by in ALL_CATEGORIES:
            v = r.category_scores.get(sort_by)
            return (0, -(v or 0.0)) if v is not None else (1, 0.0)
        # Default: overall_score desc, then avg_tps desc
        if r.overall_score is not None:
            return (0, -r.overall_score - (r.avg_tps or 0.0) * 1e-4)
        return (1, -(r.avg_tps or 0.0))

    return sorted(items, key=_key)


def compare_runs(baseline: BenchmarkRun, target: BenchmarkRun) -> dict[str, Any]:
    """Computes metric deltas and per-case comparison between two runs."""
    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return round(b - a, 3)

    summary = {
        "overall_score": (baseline.overall_score, target.overall_score, _delta(baseline.overall_score, target.overall_score)),
        "load_time_s": (baseline.load_time_s, target.load_time_s, _delta(baseline.load_time_s, target.load_time_s)),
        "avg_ttft_s": (baseline.avg_ttft_s, target.avg_ttft_s, _delta(baseline.avg_ttft_s, target.avg_ttft_s)),
        "avg_tps": (baseline.avg_tps, target.avg_tps, _delta(baseline.avg_tps, target.avg_tps)),
    }
    for cat in ALL_CATEGORIES:
        a_s = baseline.category_scores.get(cat)
        b_s = target.category_scores.get(cat)
        summary[f"score_{cat}"] = (a_s, b_s, _delta(a_s, b_s))

    b_map = {c.case_id: c for c in baseline.case_results}
    t_map = {c.case_id: c for c in target.case_results}
    all_ids = list(dict.fromkeys([c.case_id for c in baseline.case_results] + [c.case_id for c in target.case_results]))
    cases = []
    for cid in all_ids:
        ca = b_map.get(cid)
        cb = t_map.get(cid)
        cases.append({
            "case_id": cid,
            "title": (cb or ca).title if (cb or ca) else cid,
            "category": (cb or ca).category if (cb or ca) else "",
            "a_passed": ca.passed if ca else None,
            "b_passed": cb.passed if cb else None,
            "a_score": ca.score if ca else None,
            "b_score": cb.score if cb else None,
            "a_tps": ca.tps if ca else None,
            "b_tps": cb.tps if cb else None,
        })
    return {"summary": summary, "cases": cases}
