"""Chat inference backends for MLX (mlx-lm) and Ollama."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import tools


@dataclass
class ChatConfig:
    backend: str  # "mlx" | "ollama"
    model_id: str  # local folder path for MLX, or model:tag for Ollama
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class ChatMetrics:
    tokens: int = 0
    tps: float | None = None
    elapsed: float = 0.0


def ollama_base_url() -> str:
    host = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def list_local_mlx_models(models_dir: Path | None = None) -> list[Path]:
    """Finds local model directories under models/ that look usable with mlx-lm."""
    root = models_dir or Path("models")
    if not root.is_dir():
        return []
    found: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "config.json").is_file() and any(p.glob("*.safetensors")):
            found.append(p.resolve())
    # Prefer pre-quantized MLX folders first
    found.sort(key=lambda d: (0 if "mlx" in d.name.lower() else 1, d.name.lower()))
    return found


def format_chat_prompt(tokenizer: object, messages: list[dict[str, str]]) -> str:
    """Formats messages using the tokenizer's chat_template, or a simple fallback."""
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    has_tmpl = bool(getattr(tokenizer, "chat_template", None))
    if callable(apply_fn) and has_tmpl:
        try:
            out = apply_fn(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(out, str):
                return out
        except Exception:
            pass
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


class MlxChatSession:
    """Holds one loaded MLX model in memory for multi-turn streaming chat."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loaded_path: Path | None = None
        self.model: object = None
        self.tokenizer: object = None

    def is_loaded(self, model_path: Path | None = None) -> bool:
        if self.model is None or self.tokenizer is None:
            return False
        if model_path is None:
            return True
        return self.loaded_path == Path(model_path).resolve()

    def ensure_loaded(self, model_path: Path, on_status: Callable[[str], None] | None = None) -> None:
        target = Path(model_path).resolve()
        if not target.is_dir():
            raise RuntimeError(f"MLX モデルフォルダが見つかりません: {target}")
        if not tools.mlx_lm_available():
            raise RuntimeError("mlx-lm が利用できません（Apple Silicon Mac 環境が必要です）")

        with self._lock:
            if self.loaded_path == target and self.model is not None and self.tokenizer is not None:
                return
            self._unload_unlocked()
            if on_status:
                on_status(f"モデルを読み込み中: {target.name} ...")
            import mlx_lm

            model, tokenizer = mlx_lm.load(str(target))
            self.model = model
            self.tokenizer = tokenizer
            self.loaded_path = target

    def unload(self) -> None:
        with self._lock:
            self._unload_unlocked()

    def _unload_unlocked(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loaded_path = None
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            pass

    def stream_chat(
        self,
        model_path: Path,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        cancel_event: threading.Event | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> ChatMetrics:
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        self.ensure_loaded(model_path, on_status=on_status)
        if cancel_event and cancel_event.is_set():
            return ChatMetrics()

        assert self.model is not None and self.tokenizer is not None
        prompt = format_chat_prompt(self.tokenizer, messages)
        sampler = make_sampler(temp=max(0.0, float(temperature)))

        if on_status:
            on_status("応答を生成中...")
        t0 = time.monotonic()
        tokens = 0
        last_tps: float | None = None

        for resp in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=int(max_tokens),
            sampler=sampler,
        ):
            if cancel_event and cancel_event.is_set():
                break
            text = getattr(resp, "text", "")
            if text and on_chunk:
                on_chunk(text)
            gtok = getattr(resp, "generation_tokens", None)
            if isinstance(gtok, int):
                tokens = gtok
            else:
                tokens += 1
            gtps = getattr(resp, "generation_tps", None)
            if isinstance(gtps, (int, float)) and gtps > 0:
                last_tps = float(gtps)

        elapsed = time.monotonic() - t0
        if last_tps is None and tokens > 0 and elapsed > 0:
            last_tps = tokens / elapsed
        return ChatMetrics(tokens=tokens, tps=last_tps, elapsed=elapsed)


def stream_ollama_chat(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    cancel_event: threading.Event | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> ChatMetrics:
    """Streams a chat completion from the local Ollama HTTP API (/api/chat)."""
    url = f"{ollama_base_url()}/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if on_status:
        on_status(f"Ollama ({model_name}) で生成中...")

    t0 = time.monotonic()
    tokens = 0
    tps: float | None = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                msg = data.get("message") or {}
                content = msg.get("content") or ""
                if content and on_chunk:
                    on_chunk(content)
                if data.get("done"):
                    ec = data.get("eval_count")
                    ed = data.get("eval_duration")
                    if isinstance(ec, int):
                        tokens = ec
                    if isinstance(ec, int) and isinstance(ed, int) and ed > 0:
                        tps = ec / (ed / 1e9)
                    break
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama に接続できません（ollama が起動しているか確認してください）: {e.reason}") from e

    elapsed = time.monotonic() - t0
    if tps is None and tokens > 0 and elapsed > 0:
        tps = tokens / elapsed
    return ChatMetrics(tokens=tokens, tps=tps, elapsed=elapsed)
