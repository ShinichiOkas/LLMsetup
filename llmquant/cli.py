"""Command line entry: ``python -m llmquant [gui|inspect|run|doctor] ...``.

With no sub-command the GUI starts. ``inspect`` and ``run`` drive the same
core as the GUI, so every GUI path can also be exercised from a script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import catalog, tools
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
    fmt = getattr(a, "format", catalog.FMT_GGUF) or catalog.FMT_GGUF
    base = inspect_base(a.path)
    out_dir = Path(a.out_dir) if a.out_dir else (base.path.parent if base.path.is_file() else base.path.parent)
    opts = build_options(base, Environment.detect(out_dir), Path(a.imatrix) if a.imatrix else None, target_format=fmt)
    probs = base.problems_for(fmt)
    ok = base.ok_for(fmt)
    if a.json:
        print(json.dumps({
            "format": fmt, "kind": base.kind, "name": base.name, "architecture": base.architecture,
            "arch_supported": base.arch_supported, "mlx_model_type": base.mlx_model_type,
            "mlx_supported": base.mlx_supported, "mlx_quant": base.mlx_quant,
            "source_type": base.source_type,
            "is_quantized": base.is_quantized, "n_params": base.n_params,
            "weight_bytes": base.weight_bytes, "has_chat_template": base.has_chat_template,
            "problems": probs, "warnings": base.warnings,
            "options": [{
                "type": o.name, "status": o.status, "est_bytes": o.est_bytes, "fits_vram": o.fits_vram,
                "recommended": o.recommended, "reasons": o.reasons, "notes": o.notes,
            } for o in opts],
        }, ensure_ascii=False, indent=2))
        return 0 if ok else 2
    print(f"種類: {base.kind}  名前: {base.name}  構造: {base.architecture}  元の形式: {base.source_type}")
    if base.mlx_model_type:
        mlx_st = "対応" if base.mlx_supported else ("未確認" if base.mlx_supported is None else "非対応")
        print(f"MLX model_type: {base.mlx_model_type} ({mlx_st})")
    print(f"パラメータ: {human_params(base.n_params)}  重み: {human_bytes(base.weight_bytes)}  "
          f"チャットテンプレート: {base.has_chat_template}")
    for p in probs:
        print(f"  [不可] {p}")
    for w in base.warnings:
        print(f"  [注意] {w}")
    for o in opts:
        mark = "*" if o.recommended else " "
        vram = "" if o.fits_vram is None else ("メモリ○" if o.fits_vram else "メモリ×")
        why = "; ".join(o.reasons + o.notes)
        print(f" {mark} {o.name:12s} {o.status:8s} {human_bytes(o.est_bytes):>10s} {vram:7s} {why}")
    return 0 if ok else 2


def cmd_run(a: argparse.Namespace) -> int:
    base = inspect_base(a.path)
    out_dir = Path(a.out_dir) if a.out_dir else base.path.parent
    fmt = getattr(a, "format", None)
    if not fmt:
        fmt = catalog.FMT_MLX if any(t.upper().startswith("MLX-") for t in a.type) else catalog.FMT_GGUF
    names = {}
    if a.ollama_name:
        if len(a.type) != 1:
            print("--ollama-name は --type が 1 つのときだけ使えます", file=sys.stderr)
            return 2
        names[a.type[0].upper()] = a.ollama_name
    targets = [catalog.get(t, fmt).name if (fmt == catalog.FMT_MLX and t.upper() in catalog.MLX_BY_NAME) else t.upper() for t in a.type]
    spec = JobSpec(
        base=base, targets=targets, out_dir=out_dir,
        work_dir=Path(a.work_dir) if a.work_dir else None,
        imatrix=Path(a.imatrix) if a.imatrix else None,
        keep_intermediate=a.keep_intermediate, overwrite=a.overwrite,
        register_ollama=a.ollama, ollama_names=names,
        target_format=fmt,
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
    print(f"apple silicon: {st.apple_silicon}")
    mlx_types = tools.mlx_supported_model_types()
    print(f"mlx-lm      : {st.mlx_version or ('利用可' if st.mlx_available else '未導入')} "
          f"(対応 model_type: {len(mlx_types) if mlx_types else 0} 個)")
    mem_label = "Unified Mem " if st.apple_silicon and not shutil.which("nvidia-smi") else "GPU VRAM    "
    print(f"{mem_label}: {human_bytes(tools.gpu_vram_bytes())}")
    for m in st.missing:
        print(f"MISSING (GGUF): {m}")
    if st.apple_silicon:
        for m in st.mlx_missing:
            print(f"MISSING (MLX) : {m}")
    has_missing = bool(st.missing) and (not st.apple_silicon or bool(st.mlx_missing))
    return 1 if has_missing else 0


def cmd_ollama_list(a: argparse.Namespace) -> int:
    models = tools.ollama_list_models(max_age=0.0)
    if models is None:
        print("エラー: Ollama のモデル一覧を取得できませんでした（ollama が起動しているか確認してください）", file=sys.stderr)
        return 1
    if not models:
        print("ダウンロード済みの Ollama モデルはありません。")
        return 0
    print(f"{'NAME':<45} {'ID':<15} {'SIZE':<12} {'MODIFIED'}")
    for m in models:
        print(f"{m.name:<45} {m.id:<15} {m.size:<12} {m.modified}")
    return 0


def cmd_ollama_create(a: argparse.Namespace) -> int:
    from . import ollama_modelfile as om

    base_model = a.base
    target_name = a.name

    if a.modelfile:
        mf_path = Path(a.modelfile)
        if not mf_path.is_file():
            print(f"エラー: 指定された Modelfile が見つかりません: {mf_path}", file=sys.stderr)
            return 1
        mf_text = mf_path.read_text(encoding="utf-8")
    else:
        show_raw = tools.ollama_show_modelfile(base_model) if not a.no_base_template else None
        if show_raw:
            cleaned = om.clean_show_modelfile(show_raw, base_model)
            sec = om.parse_modelfile_sections(cleaned)
            sys_text = a.system if a.system is not None else sec["system"]
            params = sec["parameters"]
            if a.param:
                for p in a.param:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        params.append((k.strip(), v.strip()))
            mf_text = om.build_modelfile(
                from_model=base_model,
                system=sys_text,
                parameters=params,
                template=sec["template"],
                extra=sec["extra"],
            )
        else:
            params = []
            if a.param:
                for p in a.param:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        params.append((k.strip(), v.strip()))
            mf_text = om.build_modelfile(
                from_model=base_model,
                system=a.system or "",
                parameters=params,
            )

    job = om.OllamaCreateJob(
        name=target_name,
        modelfile_content=mf_text,
        overwrite=a.overwrite,
        on_log=print,
    )
    ok = job.run()
    return 0 if ok else 1


def cmd_hf_inspect(a: argparse.Namespace) -> int:
    from . import hf_downloader as hfd

    repo, fn = hfd.parse_hf_input(a.repo)
    if not repo:
        print("エラー: リポジトリ名または URL を指定してください", file=sys.stderr)
        return 1
    info = hfd.inspect_hf_repo(repo, token=a.token)
    if not info.is_valid:
        print(f"エラー: {info.error_message}", file=sys.stderr)
        return 1
    print(f"リポジトリ   : {info.repo_id}")
    print(f"総ファイル数 : {info.total_files}")
    print(f"HF重み形式   : {'あり (config.json + safetensors/bin)' if info.has_hf_weights else 'なし'}")
    print(f"MLXモデル    : {'はい' if info.is_mlx else 'いいえ'}")
    print(f"GGUFファイル : {len(info.gguf_files)} 個")
    for g in info.gguf_files:
        sz = info.file_sizes.get(g)
        sz_str = f" ({sz:,} bytes)" if sz else ""
        print(f"  - {g}{sz_str}")
    return 0


def cmd_hf_download(a: argparse.Namespace) -> int:
    from . import hf_downloader as hfd

    repo, fn = hfd.parse_hf_input(a.repo)
    if not repo:
        print("エラー: リポジトリ名または URL を指定してください", file=sys.stderr)
        return 1
    filename = a.file or fn
    out_dir = Path(a.out_dir) if a.out_dir else Path("models") / repo.replace("/", "_")
    job = hfd.HfDownloadJob(
        repo_id=repo,
        local_dir=out_dir,
        filename=filename,
        token=a.token,
        on_log=print,
    )
    try:
        path = job.run()
        print(f"ダウンロード完了: {path}")
        return 0
    except Exception as e:
        print(f"ダウンロード失敗: {e}", file=sys.stderr)
        return 1


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
    i.add_argument("--format", "-F", choices=[catalog.FMT_GGUF, catalog.FMT_MLX], default=catalog.FMT_GGUF,
                   help="出力形式 (gguf / mlx)")
    i.add_argument("--out-dir")
    i.add_argument("--imatrix")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("run", help="量子化を実行")
    r.add_argument("path")
    r.add_argument("--format", "-F", choices=[catalog.FMT_GGUF, catalog.FMT_MLX],
                   help="出力形式 (gguf / mlx)。省略時は -t が MLX-* なら mlx、それ以外は gguf")
    r.add_argument("--type", "-t", action="append", required=True, help="Q4_K_M / MLX-4bit など（複数可）")
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

    ol = sub.add_parser("ollama-list", help="Ollama のダウンロード済みモデル一覧を表示")
    ol.set_defaults(func=cmd_ollama_list)

    oc = sub.add_parser("ollama-create", help="Modelfile を作成して Ollama モデルを作成（Create）")
    oc.add_argument("base", help="ベースとなるダウンロード済みモデル名（例: qwen2.5:7b）")
    oc.add_argument("name", help="作成する新規モデル名（例: my-qwen:latest）")
    oc.add_argument("--system", help="システムプロンプト (SYSTEM)")
    oc.add_argument("--param", action="append", default=[], help="パラメータ（例: --param temperature=0.7）")
    oc.add_argument("--modelfile", "-f", help="既存の Modelfile パスを直接指定する場合")
    oc.add_argument("--no-base-template", action="store_true", help="ベースモデルの既存設定を引き継がない")
    oc.add_argument("--overwrite", action="store_true", help="同名のモデルが存在する場合に上書きする")
    oc.set_defaults(func=cmd_ollama_create)

    hi = sub.add_parser("hf-inspect", help="Hugging Face リポジトリの情報を確認")
    hi.add_argument("repo", help="Hugging Face リポジトリ名または URL（例: Qwen/Qwen2.5-Coder-7B-Instruct）")
    hi.add_argument("--token", help="Hugging Face Access Token（任意）")
    hi.set_defaults(func=cmd_hf_inspect)

    hd = sub.add_parser("hf-download", help="Hugging Face からモデルまたは GGUF をダウンロード")
    hd.add_argument("repo", help="Hugging Face リポジトリ名または URL")
    hd.add_argument("--file", "-f", help="単体ファイルを指定してダウンロードする場合（例: model-Q4_K_M.gguf）")
    hd.add_argument("--out-dir", "-o", help="保存先ディレクトリ（既定: models/<repo_name>）")
    hd.add_argument("--token", help="Hugging Face Access Token（任意）")
    hd.set_defaults(func=cmd_hf_download)

    argv = sys.argv[1:] if argv is None else argv
    known_commands = ("gui", "inspect", "run", "doctor", "ollama-list", "ollama-create", "hf-inspect", "hf-download", "-h", "--help")
    if not argv or (argv[0] not in known_commands):
        argv = ["gui", *argv]
    a = p.parse_args(argv)
    return a.func(a)
