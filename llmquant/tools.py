"""Locating the external tools: llama.cpp binaries, the HF converter, Ollama."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class OllamaModelItem:
    name: str
    id: str = ""
    size: str = ""
    modified: str = ""


_ollama_cache: tuple[float, list[OllamaModelItem] | None] = (0.0, None)


def ollama_list_models(max_age: float = 5.0) -> list[OllamaModelItem] | None:
    """Detailed list of models registered in Ollama, or None if it cannot be asked."""
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
    items: list[OllamaModelItem] = []
    lines = p.stdout.decode("utf-8", errors="replace").splitlines()
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        mid = parts[1] if len(parts) > 1 else ""
        size = ""
        modified = ""
        if len(parts) >= 4 and parts[3].upper() in ("B", "KB", "MB", "GB", "TB"):
            size = f"{parts[2]} {parts[3]}"
            modified = " ".join(parts[4:])
        elif len(parts) >= 3:
            size = parts[2]
            modified = " ".join(parts[3:])
        items.append(OllamaModelItem(name=name, id=mid, size=size, modified=modified))
    _ollama_cache = (now, items)
    return items


def ollama_model_names(max_age: float = 5.0) -> set[str] | None:
    """Names (``name:tag``) registered in Ollama, or None if it cannot be asked."""
    items = ollama_list_models(max_age=max_age)
    if items is None:
        return None
    return {m.name.lower() for m in items}


def ollama_show_modelfile(model_name: str) -> str | None:
    """Raw modelfile string for a model from ``ollama show <name> --modelfile``."""
    exe = ollama_exe()
    if exe is None:
        return None
    try:
        p = subprocess.run([str(exe), "show", model_name, "--modelfile"],
                           capture_output=True, timeout=15, creationflags=CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.decode("utf-8", errors="replace")


@dataclass
class OllamaModelDetails:
    architecture: str = ""
    parameters_size: str = ""
    context_length: int = 2048
    embedding_length: int = 0
    quantization: str = ""
    system: str = ""
    model_parameters: dict[str, str] = field(default_factory=dict)
    raw_show: str = ""


def parse_ollama_show_output(raw: str) -> OllamaModelDetails:
    """Parses `ollama show <model>` plain text output into OllamaModelDetails."""
    details = OllamaModelDetails(raw_show=raw)
    current_section = ""
    lines = raw.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Section header check (starts with 2 spaces or 0 spaces)
        if (line.startswith("  ") and not line.startswith("    ")) or not line.startswith(" "):
            current_section = stripped.lower()
            continue

        if current_section == "model":
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                k = parts[0].lower()
                v = parts[1].strip()
                if k == "architecture":
                    details.architecture = v
                elif k == "parameters":
                    details.parameters_size = v
                elif k == "quantization":
                    details.quantization = v
                elif k == "context" and v.lower().startswith("length"):
                    sub = v.split(maxsplit=1)
                    if len(sub) == 2 and sub[1].isdigit():
                        details.context_length = int(sub[1])
                elif k == "embedding" and v.lower().startswith("length"):
                    sub = v.split(maxsplit=1)
                    if len(sub) == 2 and sub[1].isdigit():
                        details.embedding_length = int(sub[1])
        elif current_section == "parameters":
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2:
                details.model_parameters[parts[0].lower()] = parts[1].strip()
        elif current_section == "system":
            if details.system:
                details.system += "\n" + line.strip()
            else:
                details.system = line.strip()

    return details


def ollama_show_details(model_name: str) -> OllamaModelDetails | None:
    """Fetches and parses detailed model info via `ollama show <name>`."""
    exe = ollama_exe()
    if exe is None:
        return None
    try:
        p = subprocess.run([str(exe), "show", model_name],
                           capture_output=True, timeout=15, creationflags=CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return parse_ollama_show_output(p.stdout.decode("utf-8", errors="replace"))


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


def is_apple_silicon() -> bool:
    """True when running on an Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_lm_available() -> bool:
    """True when both mlx and mlx_lm are importable in the current Python."""
    return importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_lm") is not None


def mlx_lm_version() -> str | None:
    if not mlx_lm_available():
        return None
    try:
        return importlib.metadata.version("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=2)
def _mlx_supported_model_types_cached(models_dir_str: str, utils_mtime: float) -> frozenset[str]:
    import ast

    models_dir = Path(models_dir_str)
    types: set[str] = {
        p.stem for p in models_dir.glob("*.py")
        if not p.name.startswith("_")
    }
    utils_py = models_dir.parent / "utils.py"
    if utils_py.is_file():
        try:
            tree = ast.parse(utils_py.read_text(encoding="utf-8", errors="replace"))
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "MODEL_REMAPPING":
                            mapping = ast.literal_eval(node.value)
                            if isinstance(mapping, dict):
                                for k, v in mapping.items():
                                    if v in types:
                                        types.add(str(k))
        except (OSError, SyntaxError, ValueError):
            pass
    return frozenset(types)


def mlx_supported_model_types() -> frozenset[str] | None:
    """HF ``model_type`` names supported by the installed ``mlx_lm`` package.

    Inspected statically from ``mlx_lm/models/*.py`` and ``MODEL_REMAPPING``
    in ``mlx_lm/utils.py`` so neither ``mlx`` nor ``transformers`` is imported.
    Returns None when ``mlx_lm`` is not installed.
    """
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    models_dir = pkg_dir / "models"
    if not models_dir.is_dir():
        return None
    utils_py = pkg_dir / "utils.py"
    mtime = utils_py.stat().st_mtime if utils_py.is_file() else 0.0
    return _mlx_supported_model_types_cached(str(models_dir), mtime)


@dataclass
class ToolStatus:
    quantize: Path | None
    converter: Path | None
    ollama: Path | None
    llama_version: str | None
    apple_silicon: bool = False
    mlx_available: bool = False
    mlx_version: str | None = None

    @property
    def missing(self) -> list[str]:
        out = []
        setup_hint = "install.sh" if sys.platform == "darwin" else "install.bat"
        if self.quantize is None:
            out.append(f"llama-quantize が見つかりません（{llama_bin_dir()}）。{setup_hint} を実行してください")
        if self.converter is None:
            out.append(f"convert_hf_to_gguf.py が見つかりません（{llama_cpp_dir()}）。{setup_hint} を実行してください")
        return out

    @property
    def mlx_missing(self) -> list[str]:
        if not self.apple_silicon:
            return ["MLX は Apple Silicon Mac (arm64) 専用です"]
        if not self.mlx_available:
            return ["mlx-lm がインストールされていません。install.sh を実行してください"]
        return []


def llama_bin_version() -> str | None:
    f = llama_bin_dir() / "VERSION"
    try:
        return f.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def tool_status() -> ToolStatus:
    return ToolStatus(
        quantize_exe(),
        converter_script(),
        ollama_exe(),
        llama_bin_version(),
        apple_silicon=is_apple_silicon(),
        mlx_available=mlx_lm_available(),
        mlx_version=mlx_lm_version(),
    )


@lru_cache(maxsize=1)
def gpu_vram_bytes() -> int | None:
    """Total VRAM of the largest NVIDIA GPU, or Unified Memory on Apple Silicon, or None."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, creationflags=CREATE_NO_WINDOW,
            ).stdout
            vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            if vals:
                return max(vals) * 1024 * 1024
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if is_apple_silicon():
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if out.isdigit():
                return int(out)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
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
