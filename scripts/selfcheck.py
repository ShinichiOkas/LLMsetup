"""Environment self-check run at the end of install.bat.

Fails (exit 1) when the interpreter is not this checkout's venv, when a
dependency resolves to code outside this checkout (a sibling checkout or
another venv), or when a tool the GUI needs is missing or broken.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
errors: list[str] = []


def check(cond: bool, ok: str, bad: str) -> None:
    print(("  [ok]   " + ok) if cond else ("  [FAIL] " + bad))
    if not cond:
        errors.append(bad)


def inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    venv = REPO / ".venv"
    check(inside(Path(sys.prefix), venv), f"python is the venv ({sys.prefix})",
          f"python is not this checkout's venv: {sys.prefix}")

    import llmquant  # noqa: PLC0415

    check(inside(Path(llmquant.__file__), REPO), f"llmquant from this checkout ({llmquant.__file__})",
          f"llmquant resolves outside this checkout: {llmquant.__file__}")

    import gguf  # noqa: PLC0415

    gguf_root = REPO / "vendor" / "llama.cpp" / "gguf-py"
    check(inside(Path(gguf.__file__), gguf_root), f"gguf from the pinned submodule ({gguf.__file__})",
          f"gguf resolves outside vendor/llama.cpp/gguf-py: {gguf.__file__}")

    for mod in ("PySide6", "torch", "transformers", "sentencepiece", "numpy"):
        try:
            __import__(mod)
            check(True, f"import {mod}", "")
        except Exception as e:  # noqa: BLE001
            check(False, "", f"import {mod} failed: {e}")

    from llmquant import tools  # noqa: PLC0415

    conv = tools.converter_script()
    check(conv is not None, f"converter {conv}", "convert_hf_to_gguf.py missing (submodule empty?)")
    archs = tools.supported_architectures() or frozenset()
    check(len(archs) > 50, f"converter knows {len(archs)} architectures", "converter architecture list is empty")
    q = tools.quantize_exe()
    check(q is not None, f"llama-quantize {q}", "llama-quantize.exe missing (fetch_llama_bin.py failed?)")
    check(tools.llama_bin_version() == tools.LLAMA_TAG, f"binaries are {tools.LLAMA_TAG}",
          f"binaries VERSION {tools.llama_bin_version()} != pinned {tools.LLAMA_TAG}")
    if q is not None:
        try:
            p = subprocess.run([str(q), "--help"], capture_output=True, timeout=30, creationflags=tools.CREATE_NO_WINDOW)
            out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
            check("Q4_K_M" in out, "llama-quantize runs and lists Q4_K_M", f"llama-quantize --help output unexpected (rc={p.returncode})")
        except (OSError, subprocess.SubprocessError) as e:
            check(False, "", f"llama-quantize does not run: {e}")
    sub_head = subprocess.run(["git", "-C", str(REPO / "vendor" / "llama.cpp"), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    check(sub_head == tools.LLAMA_COMMIT, f"submodule at {sub_head[:10]}",
          f"submodule HEAD {sub_head[:10]} != pinned {tools.LLAMA_COMMIT[:10]}")
    ol = tools.ollama_exe()
    print(f"  [info] ollama: {ol or 'not found (registration will be unavailable)'}")

    if errors:
        print(f"selfcheck: {len(errors)} problem(s)")
        return 1
    print("selfcheck: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
