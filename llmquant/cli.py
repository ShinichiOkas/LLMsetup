"""Command line entry: ``python -m llmquant [gui|inspect|run|doctor] ...``.

With no sub-command the GUI starts. ``inspect`` and ``run`` drive the same
core as the GUI, so every GUI path can also be exercised from a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import tools
from .basemodel import human_bytes, human_params, inspect_base
from .options import Environment, build_options
from .pipeline import JobError, JobSpec, run_job


def _utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def cmd_inspect(a: argparse.Namespace) -> int:
    base = inspect_base(a.path)
    out_dir = Path(a.out_dir) if a.out_dir else (base.path.parent if base.path.is_file() else base.path.parent)
    opts = build_options(base, Environment.detect(out_dir), Path(a.imatrix) if a.imatrix else None)
    if a.json:
        print(json.dumps({
            "kind": base.kind, "name": base.name, "architecture": base.architecture,
            "arch_supported": base.arch_supported, "source_type": base.source_type,
            "is_quantized": base.is_quantized, "n_params": base.n_params,
            "weight_bytes": base.weight_bytes, "has_chat_template": base.has_chat_template,
            "problems": base.problems, "warnings": base.warnings,
            "options": [{
                "type": o.name, "status": o.status, "est_bytes": o.est_bytes, "fits_vram": o.fits_vram,
                "recommended": o.recommended, "reasons": o.reasons, "notes": o.notes,
            } for o in opts],
        }, ensure_ascii=False, indent=2))
        return 0 if base.ok else 2
    print(f"種類: {base.kind}  名前: {base.name}  構造: {base.architecture}  元の形式: {base.source_type}")
    print(f"パラメータ: {human_params(base.n_params)}  重み: {human_bytes(base.weight_bytes)}  "
          f"チャットテンプレート: {base.has_chat_template}")
    for p in base.problems:
        print(f"  [不可] {p}")
    for w in base.warnings:
        print(f"  [注意] {w}")
    for o in opts:
        mark = "*" if o.recommended else " "
        vram = "" if o.fits_vram is None else ("GPU○" if o.fits_vram else "GPU×")
        why = "; ".join(o.reasons + o.notes)
        print(f" {mark} {o.name:10s} {o.status:8s} {human_bytes(o.est_bytes):>10s} {vram:5s} {why}")
    return 0 if base.ok else 2


def cmd_run(a: argparse.Namespace) -> int:
    base = inspect_base(a.path)
    out_dir = Path(a.out_dir) if a.out_dir else base.path.parent
    names = {}
    if a.ollama_name:
        if len(a.type) != 1:
            print("--ollama-name は --type が 1 つのときだけ使えます", file=sys.stderr)
            return 2
        names[a.type[0].upper()] = a.ollama_name
    spec = JobSpec(
        base=base, targets=[t.upper() for t in a.type], out_dir=out_dir,
        work_dir=Path(a.work_dir) if a.work_dir else None,
        imatrix=Path(a.imatrix) if a.imatrix else None,
        keep_intermediate=a.keep_intermediate, overwrite=a.overwrite,
        register_ollama=a.ollama, ollama_names=names,
    )
    last = [-1]

    def prog(frac: float, idx: int, title: str) -> None:
        pct = int(frac * 100)
        if pct // 10 != last[0] // 10:
            last[0] = pct
            print(f"[progress] {pct}% {title}", flush=True)

    try:
        res = run_job(spec, on_log=lambda s: print(s, flush=True), on_progress=prog)
    except JobError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for t, p in res.outputs.items():
        print(f"OUTPUT {t} {p}")
    for t, n in res.ollama_models.items():
        print(f"OLLAMA {t} {n}")
    for w in res.warnings:
        print(f"WARN {w}")
    print(f"DONE in {res.seconds:.1f}s")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    st = tools.tool_status()
    print(f"python      : {tools.python_exe()}")
    print(f"llama.cpp   : {tools.LLAMA_RELEASE} ({tools.LLAMA_TAG}) bin VERSION={st.llama_version}")
    print(f"quantize    : {st.quantize}")
    print(f"converter   : {st.converter}")
    archs = tools.supported_architectures()
    print(f"archs       : {len(archs) if archs else 0} 個")
    print(f"ollama      : {st.ollama}")
    print(f"GPU VRAM    : {human_bytes(tools.gpu_vram_bytes())}")
    for m in st.missing:
        print(f"MISSING: {m}")
    return 1 if st.missing else 0


def cmd_gui(a: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui(a.path, screenshot=a.screenshot, width=a.width, height=a.height,
                   select=a.select, imatrix=a.imatrix)


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    p = argparse.ArgumentParser(prog="llmquant", description="LLM を GGUF に量子化して Ollama で使えるようにする")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", help="GUI を開く（既定）")
    g.add_argument("path", nargs="?")
    g.add_argument("--screenshot", help="画面を PNG に保存して終了する（確かめ用）")
    g.add_argument("--width", type=int, default=1280)
    g.add_argument("--height", type=int, default=860)
    g.add_argument("--select", action="append", default=[], help="（確かめ用）印を付ける型")
    g.add_argument("--imatrix")
    g.set_defaults(func=cmd_gui)

    i = sub.add_parser("inspect", help="ベースを調べて選択肢を表示")
    i.add_argument("path")
    i.add_argument("--out-dir")
    i.add_argument("--imatrix")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("run", help="量子化を実行")
    r.add_argument("path")
    r.add_argument("--type", "-t", action="append", required=True, help="Q4_K_M など（複数可）")
    r.add_argument("--out-dir")
    r.add_argument("--work-dir")
    r.add_argument("--imatrix")
    r.add_argument("--keep-intermediate", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--ollama", action="store_true", help="Ollama に登録する")
    r.add_argument("--ollama-name")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("doctor", help="ツールの有無を確かめる")
    d.set_defaults(func=cmd_doctor)

    argv = sys.argv[1:] if argv is None else argv
    if not argv or (argv[0] not in ("gui", "inspect", "run", "doctor", "-h", "--help")):
        argv = ["gui", *argv]
    a = p.parse_args(argv)
    return a.func(a)
