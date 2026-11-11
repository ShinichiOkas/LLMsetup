"""Locating the external tools: llama.cpp binaries, the HF converter, Ollama."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned llama.cpp release (must match the vendor/llama.cpp submodule commit).
LLAMA_TAG = "b12110"
LLAMA_RELEASE = "v0.9.0"
LLAMA_COMMIT = "cd959bae99ff4bed22bef2604f45241c09bbb8c8"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def llama_cpp_dir() -> Path:
    return Path(os.environ.get("LLMQUANT_LLAMA_CPP", REPO_ROOT / "vendor" / "llama.cpp"))


def llama_bin_dir() -> Path:
    return Path(os.environ.get("LLMQUANT_LLAMA_BIN", REPO_ROOT / "vendor" / "llama-bin"))


def quantize_exe() -> Path | None:
    p = llama_bin_dir() / _exe("llama-quantize")
    return p if p.is_file() else None


def converter_script() -> Path | None:
    p = llama_cpp_dir() / "convert_hf_to_gguf.py"
    return p if p.is_file() else None


def ollama_exe() -> Path | None:
    found = shutil.which("ollama")
    if found:
        return Path(found)
    if os.name == "nt":
        cand = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if cand.is_file():
            return cand
    return None


def python_exe() -> Path:
    """Interpreter for the converter: the venv's python.exe even when the GUI
    itself runs under pythonw.exe (run.bat)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.is_file():
            return console
    return exe


def ollama_models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


_ollama_cache: tuple[float, set[str] | None] = (0.0, None)


def ollama_model_names(max_age: float = 5.0) -> set[str] | None:
    """Names (``name:tag``) registered in Ollama, or None if it cannot be asked."""
    global _ollama_cache
    import time

    now = time.monotonic()
    if _ollama_cache[1] is not None and now - _ollama_cache[0] < max_age:
        return _ollama_cache[1]
    exe = ollama_exe()
    if exe is None:
        return None
    try:
        p = subprocess.run([str(exe), "list"], capture_output=True, timeout=15, creationflags=CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    names = set()
    for line in p.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
        parts = line.split()
        if parts:
            names.add(parts[0].lower())
    _ollama_cache = (now, names)
    return names


def forget_ollama_cache() -> None:
    global _ollama_cache
    _ollama_cache = (0.0, None)


_REGISTER_RE = re.compile(r"@ModelBase\.register\((.*?)\)", re.DOTALL)
_QUOTED_RE = re.compile(r"""["']([A-Za-z0-9_.\-]+)["']""")


@lru_cache(maxsize=4)
def _supported_architectures_cached(conv_dir: str, mtime: float) -> frozenset[str]:
    names: set[str] = set()
    root = Path(conv_dir)
    files = list((root / "conversion").glob("*.py")) + [root / "convert_hf_to_gguf.py"]
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _REGISTER_RE.finditer(text):
            names.update(_QUOTED_RE.findall(m.group(1)))
    return frozenset(names)


def supported_architectures() -> frozenset[str] | None:
    """HF ``architectures`` names the pinned converter can handle.

    Read statically from the ``@ModelBase.register("...")`` decorators so the
    converter (and torch) is never imported just to fill the GUI.
    Returns None when the converter source is not present.
    """
    root = llama_cpp_dir()
    conv = root / "conversion"
    if not conv.is_dir() and not (root / "convert_hf_to_gguf.py").is_file():
        return None
    mtime = conv.stat().st_mtime if conv.is_dir() else 0.0
    return _supported_architectures_cached(str(root), mtime)


@dataclass
class ToolStatus:
    quantize: Path | None
    converter: Path | None
    ollama: Path | None
    llama_version: str | None

    @property
    def missing(self) -> list[str]:
        out = []
        if self.quantize is None:
            out.append(f"llama-quantize が見つかりません（{llama_bin_dir()}）。install.bat を実行してください")
        if self.converter is None:
            out.append(f"convert_hf_to_gguf.py が見つかりません（{llama_cpp_dir()}）。install.bat を実行してください")
        return out


def llama_bin_version() -> str | None:
    f = llama_bin_dir() / "VERSION"
    try:
        return f.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def tool_status() -> ToolStatus:
    return ToolStatus(quantize_exe(), converter_script(), ollama_exe(), llama_bin_version())


@lru_cache(maxsize=1)
def gpu_vram_bytes() -> int | None:
    """Total VRAM of the largest NVIDIA GPU, or None."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=CREATE_NO_WINDOW,
        ).stdout
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return max(vals) * 1024 * 1024 if vals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def free_bytes(path: Path) -> int | None:
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            return None
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None
