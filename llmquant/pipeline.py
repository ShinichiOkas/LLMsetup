"""Planning and running a quantization job.

A job is a list of steps (subprocesses or small Python actions). The runner
streams every output line, turns tool output into progress fractions, can be
cancelled at any time, and never leaves a half-written GGUF under the final
name: tools write to ``<final>.part`` and the file is renamed on success.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import catalog, tools
from .basemodel import KIND_GGUF, KIND_HF, BaseInfo, ollama_name, output_filename
from .gguf_reader import read_header
from .basemodel import human_bytes
from .options import Environment, build_options, intermediate_type


class JobError(RuntimeError):
    pass


class Cancelled(JobError):
    pass


@dataclass
class JobSpec:
    base: BaseInfo
    targets: list[str]
    out_dir: Path
    work_dir: Path | None = None
    imatrix: Path | None = None
    keep_intermediate: bool = False
    overwrite: bool = False
    register_ollama: bool = False
    ollama_names: dict[str, str] = field(default_factory=dict)
    threads: int | None = None

    def output_path(self, qtype: str) -> Path:
        return Path(self.out_dir) / output_filename(self.base.stem, qtype)

    def ollama_name_for(self, qtype: str) -> str:
        return self.ollama_names.get(qtype) or ollama_name(self.base.stem, qtype)


@dataclass
class Step:
    title: str
    kind: str  # "convert" | "quantize" | "move" | "verify" | "ollama"
    cmd: list[str] | None = None
    action: Callable[["Runner"], None] | None = None
    partial: Path | None = None  # file written by this step under a temp name
    final: Path | None = None  # name the partial is renamed to on success
    weight: float = 1.0  # relative duration, for the overall progress bar
    temporary: bool = False  # final is an intermediate: delete it if the job fails


@dataclass
class JobResult:
    outputs: dict[str, Path] = field(default_factory=dict)
    ollama_models: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _part(p: Path) -> Path:
    return p.with_name(p.name + ".part")


def check_outputs(spec: JobSpec) -> list[str]:
    """Problems that must be resolved before running (empty = OK)."""
    errs: list[str] = []
    if not spec.base.ok:
        errs.extend(spec.base.problems or ["このベースからは作れません"])
    if not spec.targets:
        errs.append("量子化の種類を 1 つ以上選んでください")
    st = tools.tool_status()
    if st.quantize is None and any(
        not (spec.base.kind == KIND_HF and t in catalog.CONVERTER_OUTTYPES) for t in spec.targets
    ):
        errs.append(st.missing[0] if st.missing else "llama-quantize が見つかりません")
    if spec.base.kind == KIND_HF and st.converter is None:
        errs.append("convert_hf_to_gguf.py が見つかりません。install.bat を実行してください")
    if spec.register_ollama and st.ollama is None:
        errs.append("ollama が見つかりません（Ollama をインストールしてください）")
    if spec.imatrix and not Path(spec.imatrix).is_file():
        errs.append(f"imatrix ファイルがありません: {spec.imatrix}")
    if not spec.overwrite:
        for t in spec.targets:
            p = spec.output_path(t)
            if p.exists():
                errs.append(f"出力先に同名のファイルがあります: {p.name}（上書きするなら「上書き」に印）")
    names = [spec.ollama_name_for(t) for t in spec.targets] if spec.register_ollama else []
    bad = [n for n in names if not re.fullmatch(r"[a-z0-9][a-z0-9._\-/]*(:[a-z0-9._\-]+)?", n)]
    if bad:
        errs.append(f"Ollama のモデル名に使えない文字があります: {', '.join(bad)}（英小文字・数字・. _ - : のみ）")
    for t in spec.targets:
        try:
            if spec.output_path(t).resolve() in [p.resolve() for p in spec.base.gguf_parts]:
                errs.append("出力先が元のファイルと同じです")
        except OSError:
            pass

    # The same rules the GUI shows: a disabled type never reaches the tools,
    # whichever entrance (GUI, CLI, API) asked for it.
    if spec.base.ok and spec.targets:
        env = Environment.detect(Path(spec.out_dir), Path(spec.work_dir) if spec.work_dir else None)
        by = {o.name: o for o in build_options(spec.base, env, spec.imatrix)}
        for t in spec.targets:
            o = by.get(t.upper())
            if o is None:
                errs.append(f"{t} は扱えない種類です")
            elif not o.enabled:
                errs.append(f"{t} は作れません: {' / '.join(o.reasons)}")
        if spec.register_ollama:
            need = sum(by[t.upper()].est_bytes for t in spec.targets if t.upper() in by)
            odir = tools.ollama_models_dir()
            free = tools.free_bytes(odir)
            same = Path(odir).resolve().drive.lower() == Path(spec.out_dir).resolve().drive.lower()
            if free is not None and need * (2 if same else 1) > free:
                errs.append(f"Ollama の保存先（{odir}）の空きが足りません。登録で GGUF がもう 1 つ複製されます"
                            f"（必要 約 {human_bytes(need * (2 if same else 1))} / 空き {human_bytes(free)}）")

    # ollama create silently replaces a model of the same name.
    if names and not spec.overwrite:
        existing = tools.ollama_model_names()
        if existing:
            dup = [n for n in names if n.lower() in existing or f"{n.lower()}:latest" in existing]
            if dup:
                errs.append(f"Ollama に同名のモデルがあります: {', '.join(dup)}（置き換えるなら「上書き」に印）")
    return errs


def template_is_bare(show_modelfile: str) -> bool:
    """True when Ollama did not recognise a chat template.

    Real ``ollama show --modelfile`` output (0.40): a recognised template is
    ``TEMPLATE "<multi-line>"``; an unrecognised or missing one is the bare
    passthrough ``TEMPLATE {{ .Prompt }}``. A RENDERER line means Ollama renders
    the conversation itself.
    """
    if re.search(r"^RENDERER\s+\S", show_modelfile, re.MULTILINE):
        return False
    m = re.search(r'^TEMPLATE\s+(?:"((?:[^"\\]|\\.)*)"|(.*))$', show_modelfile, re.MULTILINE | re.DOTALL)
    if not m:
        return True
    body = (m.group(1) if m.group(1) is not None else (m.group(2) or "").splitlines()[0]).strip()
    return body in ("", "{{ .Prompt }}")


def plan(spec: JobSpec) -> list[Step]:
    base = spec.base
    steps: list[Step] = []
    out_dir = Path(spec.out_dir)
    work_dir = Path(spec.work_dir or spec.out_dir)
    targets = [t.upper() for t in spec.targets]
    py = str(tools.python_exe())
    conv = str(tools.converter_script() or "convert_hf_to_gguf.py")
    qexe = str(tools.quantize_exe() or "llama-quantize")
    gb = max(base.n_params / 1e9, 0.1)

    def quant_cmd(src: Path, dst: Path, t: str, requant: bool) -> list[str]:
        cmd = [qexe]
        if requant:
            cmd.append("--allow-requantize")
        if spec.imatrix:
            cmd += ["--imatrix", str(spec.imatrix)]
        cmd += [str(src), str(dst), t]
        if spec.threads:
            cmd.append(str(spec.threads))
        return cmd

    if base.kind == KIND_HF:
        model_dir = str(base.model_dir)
        inter_t = intermediate_type(base)
        direct = [t for t in targets if t in ("F32", inter_t)]
        via_inter = [t for t in targets if t not in direct]
        # Formats the converter writes directly
        for t in direct:
            final = spec.output_path(t)
            steps.append(Step(
                f"変換: HF → {t}", "convert",
                cmd=[py, conv, model_dir, "--outtype", catalog.CONVERTER_OUTTYPES[t], "--outfile", str(_part(final))],
                partial=_part(final), final=final, weight=2.0 * gb,
            ))
        if via_inter:
            inter_name = output_filename(base.stem, inter_t)
            created_inter = False
            if inter_t in targets:
                inter = spec.output_path(inter_t)  # already produced above
            else:
                inter = work_dir / inter_name
                if inter.is_file():
                    # Never overwrite or delete a file this job did not create:
                    # an existing intermediate (e.g. kept from an earlier run) is reused.
                    steps.append(Step(f"既存の中間ファイルを使う: {inter.name}", "move",
                                      action=_check_intermediate(inter, inter_t), weight=0.01))
                else:
                    created_inter = True
                    steps.append(Step(
                        f"変換: HF → {inter_t}（中間ファイル）", "convert",
                        cmd=[py, conv, model_dir, "--outtype", catalog.CONVERTER_OUTTYPES[inter_t],
                             "--outfile", str(_part(inter))],
                        partial=_part(inter), final=inter, weight=2.0 * gb,
                        temporary=not spec.keep_intermediate,
                    ))
            for t in via_inter:
                final = spec.output_path(t)
                steps.append(Step(f"量子化: {inter_t} → {t}", "quantize", cmd=quant_cmd(inter, _part(final), t, False),
                                  partial=_part(final), final=final, weight=1.0 * gb))
            if created_inter and not spec.keep_intermediate:
                steps.append(Step("中間ファイルを削除", "move", action=_remove_file(inter), weight=0.01))
    elif base.kind == KIND_GGUF:
        src = base.gguf_parts[0] if base.gguf_parts else base.path
        for t in targets:
            final = spec.output_path(t)
            steps.append(Step(f"量子化: {base.source_type} → {t}", "quantize",
                              cmd=quant_cmd(src, _part(final), t, base.is_quantized),
                              partial=_part(final), final=final, weight=1.0 * gb))
    else:
        raise JobError("このベースからは作れません")

    for t in targets:
        steps.append(Step(f"確認: {t} の GGUF を読み戻す", "verify", action=_verify_output(spec.output_path(t), t), weight=0.01))

    if spec.register_ollama:
        for t in targets:
            name = spec.ollama_name_for(t)
            steps.append(Step(f"Ollama に登録: {name}", "ollama", action=_ollama_create(spec.output_path(t), name, t),
                              weight=0.3 * gb))
    return steps


# ---------------------------------------------------------------- actions

def _remove_file(p: Path) -> Callable[["Runner"], None]:
    def run(r: "Runner") -> None:
        try:
            p.unlink()
            r.log(f"削除: {p}")
        except FileNotFoundError:
            pass
    return run


def _check_intermediate(p: Path, t: str) -> Callable[["Runner"], None]:
    def run(r: "Runner") -> None:
        hdr = read_header(p)
        got = hdr.file_type_name
        if got != t:
            raise JobError(f"作業フォルダの既存ファイル {p.name} は {got} です（{t} が必要）。別の作業フォルダを指定してください")
        r.log(f"既存の中間ファイルを使います（消しません）: {p}")
    return run


def _verify_output(p: Path, t: str) -> Callable[["Runner"], None]:
    def run(r: "Runner") -> None:
        if not p.is_file():
            raise JobError(f"出力がありません: {p}")
        hdr = read_header(p)
        got = hdr.file_type_name
        r.log(f"読み戻し: {p.name}  file_type={got}  tensors={len(hdr.tensors)}  size={p.stat().st_size:,} bytes")
        if got and got != t.upper():
            raise JobError(f"出力の形式が違います（期待 {t} / 実際 {got}）")
        r.result.outputs[t] = p
    return run


def _modelfile_text(gguf: Path) -> str:
    return f'FROM "{gguf.resolve()}"\n'


def _ollama_create(gguf: Path, name: str, t: str) -> Callable[["Runner"], None]:
    def run(r: "Runner") -> None:
        exe = tools.ollama_exe()
        if exe is None:
            raise JobError("ollama が見つかりません")
        mf = gguf.with_name(gguf.stem + ".Modelfile")
        mf.write_text(_modelfile_text(gguf), encoding="utf-8")
        r.log(f"Modelfile: {mf}")
        r.run_cmd([str(exe), "create", name, "-f", str(mf)], cwd=gguf.parent)
        tools.forget_ollama_cache()
        show = r.capture([str(exe), "show", name, "--modelfile"])
        if template_is_bare(show):
            w = (f"{name}: Ollama がチャットテンプレートを認識しませんでした（TEMPLATE が素通し）。"
                 "会話で使うなら Modelfile に TEMPLATE を書き足してください")
            r.log("WARN: " + w)
            r.result.warnings.append(w)
        r.result.ollama_models[t] = name
        r.log(f"Ollama に登録しました: {name}  （使い方: ollama run {name}）")
    return run


# ---------------------------------------------------------------- runner

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_QUANT_PROGRESS = re.compile(r"^\[\s*(\d+)\s*/\s*(\d+)\]")
_PERCENT = re.compile(r"(\d{1,3})%\|")


def parse_progress(kind: str, line: str) -> float | None:
    if kind == "quantize":
        m = _QUANT_PROGRESS.match(line.strip())
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            return i / n if n else None
    if kind == "convert":
        m = _PERCENT.search(line)
        if m:
            return min(int(m.group(1)), 100) / 100
    return None


class Runner:
    """Runs the steps of a plan. Callbacks are called from the runner thread."""

    def __init__(
        self,
        steps: list[Step],
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[float, int, str], None] | None = None,
    ):
        self.steps = steps
        self.on_log = on_log or (lambda s: None)
        self.on_progress = on_progress or (lambda f, i, t: None)
        self.result = JobResult()
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._total = sum(s.weight for s in steps) or 1.0
        self._done = 0.0
        self._idx = 0

    # -- public
    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            p = self._proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def log(self, line: str) -> None:
        # ollama prints terminal control sequences (ESC[K, ESC[2K ...)
        line = _ANSI_RE.sub("", line)
        if line.strip():
            self.on_log(line)

    def run(self) -> JobResult:
        t0 = time.monotonic()
        try:
            for i, step in enumerate(self.steps):
                if self.cancelled:
                    raise Cancelled("取り消しました")
                self._idx = i
                self.log(f"=== [{i + 1}/{len(self.steps)}] {step.title}")
                self._report(0.0, step)
                if step.partial and step.partial.exists():
                    step.partial.unlink()
                if step.cmd:
                    self.run_cmd(step.cmd, kind=step.kind, step=step)
                if step.action:
                    step.action(self)
                if step.partial and step.final:
                    if not step.partial.is_file():
                        raise JobError(f"出力ができていません: {step.partial}")
                    os.replace(step.partial, step.final)
                    self.log(f"出力: {step.final}")
                self._done += step.weight
                self._report(0.0, step)
            self.on_progress(1.0, len(self.steps), "完了")
        except BaseException:
            self._cleanup()
            raise
        finally:
            self.result.seconds = time.monotonic() - t0
        return self.result

    # -- helpers
    def _report(self, frac: float, step: Step) -> None:
        overall = (self._done + step.weight * frac) / self._total
        self.on_progress(min(overall, 1.0), self._idx, step.title)

    def _cleanup(self) -> None:
        for s in self.steps:
            doomed = [s.partial] if s.partial else []
            if s.temporary and s.final:
                doomed.append(s.final)
            for f in doomed:
                if f and f.exists():
                    try:
                        f.unlink()
                        self.log(f"途中のファイルを削除: {f}")
                    except OSError:
                        pass

    @staticmethod
    def _env() -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    def capture(self, cmd: list[str]) -> str:
        p = subprocess.run(cmd, capture_output=True, env=self._env(), creationflags=tools.CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
        return p.stdout.decode("utf-8", errors="replace")

    def run_cmd(self, cmd: list[str], kind: str = "", step: Step | None = None, cwd: Path | None = None) -> None:
        self.log("> " + subprocess.list2cmdline(cmd))
        with self._lock:
            if self.cancelled:
                raise Cancelled("取り消しました")
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=self._env(), cwd=str(cwd) if cwd else None, creationflags=tools.CREATE_NO_WINDOW,
            )
        proc = self._proc
        assert proc.stdout is not None
        buf = b""
        tail: list[str] = []
        last_emit = 0.0
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                tail = (tail + [line])[-30:]
                frac = parse_progress(kind, line) if step else None
                if frac is not None and step:
                    self._report(frac, step)
                    # tqdm/quantize lines are many; keep the log readable
                    now = time.monotonic()
                    if now - last_emit < 0.5 and kind == "convert":
                        continue
                    last_emit = now
                self.log(line)
        if buf.strip():
            self.log(buf.decode("utf-8", errors="replace").rstrip())
        rc = proc.wait()
        with self._lock:
            self._proc = None
        if self.cancelled:
            raise Cancelled("取り消しました")
        if rc != 0:
            hint = _diagnose(tail)
            raise JobError(f"失敗しました（終了コード {rc}）" + (f": {hint}" if hint else ""))


def _diagnose(tail: list[str]) -> str:
    text = "\n".join(tail)
    if "requires an imatrix" in text or "Missing importance matrix" in text:
        return "この形式には imatrix ファイルが必要です"
    if "requantizing from type" in text:
        return "量子化済みの GGUF からの再量子化が拒否されました"
    if "No module named" in text:
        m = re.search(r"No module named '([^']+)'", text)
        return f"Python のモジュール {m.group(1) if m else ''} がありません。install.bat を実行してください"
    if "Model " in text and "is not supported" in text:
        return "このモデル構造は変換器が対応していません"
    if "out of memory" in text.lower() or "MemoryError" in text:
        return "メモリが足りません"
    for line in reversed(tail):
        if "error" in line.lower():
            return line.strip()[:300]
    return tail[-1][:300] if tail else ""


def run_job(spec: JobSpec, on_log=None, on_progress=None) -> JobResult:
    errs = check_outputs(spec)
    if errs:
        raise JobError("\n".join(errs))
    Path(spec.out_dir).mkdir(parents=True, exist_ok=True)
    if spec.work_dir:
        Path(spec.work_dir).mkdir(parents=True, exist_ok=True)
    return Runner(plan(spec), on_log, on_progress).run()


def dry_run_size(src: Path, qtype: str, imatrix: Path | None = None, timeout: float = 120) -> tuple[int | None, bool]:
    """Exact output size from ``llama-quantize --dry-run`` (GGUF bases only).

    Returns (bytes or None, needs_imatrix).
    """
    exe = tools.quantize_exe()
    if exe is None:
        return None, False
    cmd = [str(exe), "--dry-run", "--allow-requantize"]
    if imatrix:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(src), qtype]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=tools.CREATE_NO_WINDOW,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None, False
    out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
    m = re.search(r"quant size\s*=\s*([\d.]+)\s*MiB", out)
    need = "will require an imatrix" in out
    if p.returncode != 0 or not m:
        return None, need
    return int(float(m.group(1)) * 1024 * 1024), need
