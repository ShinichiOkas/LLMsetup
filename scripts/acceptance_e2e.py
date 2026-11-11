"""End-to-end acceptance: real model -> GGUF -> Ollama -> a chat reply.

Run with the venv python after install.bat:

    .venv\\Scripts\\python.exe -X utf8 scripts\\acceptance_e2e.py [--work DIR] [--keep]

Stages (each prints PASS/FAIL; exit 1 if any fails):
  1 download   a tiny ungated chat model (HuggingFaceTB/SmolLM2-135M-Instruct)
               into a folder whose path has Japanese and spaces
  2 inspect    HF folder is recognised, Q4_K_M is offered and recommended/enabled
  3 gui        through the GUI entrance (MainWindow, offscreen): HF -> Q4_K_M,
               registered in Ollama; screenshot of the finished window
  4 api        HF -> Q8_0 + BF16 in one job; outputs read back with the right type
  5 requant    GGUF Q8_0 -> Q4_K_S (requantize path) + exact size via --dry-run
  6 imatrix    IQ2_XXS without imatrix is refused before running
  7 cancel     a running job is cancelled; no .part / intermediate left behind
  8 ollama     every registered model answers a prompt; TEMPLATE is not bare
Registered models are prefixed "llmquant-accept-" and removed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llmquant import tools  # noqa: E402
from llmquant.basemodel import inspect_base  # noqa: E402
from llmquant.gguf_reader import read_header  # noqa: E402
from llmquant.options import Environment, build_options  # noqa: E402
from llmquant.pipeline import (  # noqa: E402
    Cancelled,
    JobSpec,
    Runner,
    check_outputs,
    dry_run_size,
    plan,
    run_job,
    template_is_bare,
)

MODEL_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
results: list[tuple[str, bool, str]] = []
created_models: list[str] = []


def stage(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def ollama(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    exe = tools.ollama_exe()
    return subprocess.run([str(exe), *args], capture_output=True, timeout=timeout,
                          creationflags=tools.CREATE_NO_WINDOW)


def ask(name: str) -> str:
    p = ollama("run", name, "Reply with one short sentence: what is the capital of France?", timeout=300)
    return p.stdout.decode("utf-8", errors="replace").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--keep", action="store_true", help="keep files and Ollama models")
    a = ap.parse_args()
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="llmquant-accept-"))
    tag = uuid.uuid4().hex[:6]
    prefix = f"llmquant-accept-{tag}"
    model_dir = work / "受入 テスト" / "SmolLM2 135M 指示版"
    out_dir = work / "出力 先"
    print(f"work dir: {work}")

    try:
        # 1 download
        if not (model_dir / "config.json").is_file():
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            snap = snapshot_download(MODEL_REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
                                     cache_dir=str(work / "hf-cache"))
            shutil.copytree(snap, model_dir, dirs_exist_ok=True)
        stage("1 download", (model_dir / "config.json").is_file(), str(model_dir))

        # 2 inspect
        base = inspect_base(model_dir)
        opts = {o.name: o for o in build_options(base, Environment.detect(out_dir))}
        stage("2 inspect", base.ok and base.arch_supported is True and opts["Q4_K_M"].status == "ok",
              f"arch={base.architecture} src={base.source_type} params={base.n_params} "
              f"recommended={[n for n, o in opts.items() if o.recommended]}")

        # 3 gui
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication  # noqa: PLC0415

        from llmquant.gui import MainWindow  # noqa: PLC0415

        app = QApplication.instance() or QApplication([])
        w = MainWindow()
        w.load_base(str(model_dir))
        w.out_edit.setText(str(out_dir))
        w._on_out_dir_edited()
        w.set_selected(["Q4_K_M"])
        w.ollama_chk.setChecked(True)
        w.ollama_edit.setText(prefix + "-gui")
        w._on_ollama_edited(prefix + "-gui")
        out_dir.mkdir(parents=True, exist_ok=True)
        enabled = w.run_btn.isEnabled()
        done: list = []
        w.sig.finished.connect(lambda res, err: done.append((res, err)))
        if enabled:
            w.start()
            t0 = time.monotonic()
            while not done and time.monotonic() - t0 < 1800:
                app.processEvents()
                time.sleep(0.05)
        app.processEvents()
        shot = work / "gui_finished.png"
        w.grab().save(str(shot))
        res, err = done[0] if done else (None, "timeout or run button disabled: " + w.status.text())
        if res:
            created_models.extend(res.ollama_models.values())
        q4 = out_dir / f"{model_dir.name}-Q4_K_M.gguf"
        ok = res is not None and q4.is_file() and read_header(q4).file_type_name == "Q4_K_M"
        leftovers = list(out_dir.glob("*.part")) + list(out_dir.glob("*-BF16.gguf"))
        stage("3 gui", ok and not leftovers, f"err={err} out={q4.name if q4.exists() else None} "
              f"leftovers={[p.name for p in leftovers]} shot={shot}")
        w.close()

        # 4 api: two targets in one job (one of them is the converter's own output)
        spec = JobSpec(base, ["Q8_0", "BF16"], out_dir=out_dir, register_ollama=True,
                       ollama_names={"Q8_0": f"{prefix}:q8_0", "BF16": f"{prefix}:bf16"})
        errs = check_outputs(spec)
        r = run_job(spec, on_log=lambda s: None) if not errs else None
        if r:
            created_models.extend(r.ollama_models.values())
        types = {t: read_header(p).file_type_name for t, p in (r.outputs.items() if r else [])}
        sizes = {t: p.stat().st_size for t, p in (r.outputs.items() if r else [])}
        stage("4 api", bool(r) and types == {"Q8_0": "Q8_0", "BF16": "BF16"} and sizes["Q8_0"] < sizes["BF16"],
              f"errs={errs} types={types} sizes={sizes}")

        # 5 requant from a GGUF
        q8 = out_dir / f"{model_dir.name}-Q8_0.gguf"
        gb = inspect_base(q8)
        gopts = {o.name: o for o in build_options(gb, Environment.detect(out_dir))}
        exact, need_im = dry_run_size(q8, "Q4_K_S")
        spec = JobSpec(gb, ["Q4_K_S"], out_dir=out_dir)
        r = run_job(spec)
        got = r.outputs.get("Q4_K_S")
        # dry-run reports tensor bytes only; compare with the header, not the file
        # size (tokenizer metadata is a large share of a ~100 MB file)
        real = read_header(got).tensor_bytes if got else None
        close = bool(exact and real and abs(exact - real) / real < 0.01)
        stage("5 requant", gb.is_quantized and gopts["Q8_0"].status == "disabled" and got is not None and close,
              f"dry-run={exact} real={real} need_imatrix={need_im}")

        # 6 imatrix-only type is refused up front, by the options AND by the job gate
        o = {x.name: x for x in build_options(base, Environment.detect(out_dir))}["IQ2_XXS"]
        gate = check_outputs(JobSpec(base, ["IQ2_XXS"], out_dir=out_dir))
        stage("6 imatrix", o.status == "disabled" and any("IQ2_XXS は作れません" in e for e in gate),
              "; ".join(o.reasons) + " | gate: " + "; ".join(gate))

        # 7 cancel mid-run
        cdir = work / "取り消し 先"
        spec = JobSpec(base, ["Q4_0"], out_dir=cdir, work_dir=cdir)
        cdir.mkdir(parents=True, exist_ok=True)
        runner = Runner(plan(spec))
        errbox: list = []
        th = threading.Thread(target=lambda: _run_catch(runner, errbox))
        th.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120 and not list(cdir.glob("*.part")):
            time.sleep(0.1)
        runner.cancel()
        th.join(60)
        left = [p.name for p in cdir.iterdir()]
        stage("7 cancel", bool(errbox) and isinstance(errbox[0], Cancelled) and not left,
              f"left={left} err={errbox[:1]}")

        # 8 ollama answers
        answers = {}
        bare = []
        for name in created_models:
            answers[name] = ask(name)[:120]
            show = ollama("show", name, "--modelfile").stdout.decode("utf-8", errors="replace")
            if template_is_bare(show):
                bare.append(name)
        ok = bool(answers) and all(len(v) > 0 for v in answers.values()) and not bare
        stage("8 ollama", ok, json.dumps({"answers": answers, "bare_template": bare}, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        stage("exception", False, f"{type(e).__name__}: {e}")
    finally:
        if not a.keep:
            for name in created_models:
                ollama("rm", name, timeout=60)
            print(f"removed Ollama models: {created_models}")

    failed = [n for n, ok, _ in results if not ok]
    print(f"== {len(results) - len(failed)}/{len(results)} PASS" + (f"  FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


def _run_catch(runner: Runner, box: list) -> None:
    try:
        runner.run()
        box.append(None)
    except BaseException as e:  # noqa: BLE001
        box.append(e)


if __name__ == "__main__":
    raise SystemExit(main())
