import sys
import threading
import time
from pathlib import Path

import pytest
from fixtures import write_gguf, write_hf_dir

from llmquant.basemodel import inspect_base
from llmquant.pipeline import Cancelled, JobError, JobSpec, Runner, Step, check_outputs, parse_progress, plan


def kinds(steps):
    return [s.kind for s in steps]


def test_plan_hf_multi(jp_dir: Path):
    b = inspect_base(write_hf_dir(jp_dir / "Tiny"))
    spec = JobSpec(b, ["Q4_K_M", "Q8_0"], out_dir=jp_dir / "出力 先", register_ollama=True)
    steps = plan(spec)
    assert kinds(steps) == ["convert", "quantize", "quantize", "move", "verify", "verify", "ollama", "ollama"]
    conv = steps[0]
    assert "--outtype" in conv.cmd and conv.cmd[conv.cmd.index("--outtype") + 1] == "bf16"
    assert conv.temporary and conv.final.name == "Tiny-BF16.gguf"
    q = steps[1].cmd
    assert q[-3].endswith("Tiny-BF16.gguf") and q[-2].endswith("Tiny-Q4_K_M.gguf.part") and q[-1] == "Q4_K_M"
    assert "--allow-requantize" not in q


def test_plan_hf_target_is_intermediate(jp_dir: Path):
    b = inspect_base(write_hf_dir(jp_dir / "Tiny"))
    steps = plan(JobSpec(b, ["BF16", "Q4_K_M"], out_dir=jp_dir))
    assert kinds(steps) == ["convert", "quantize", "verify", "verify"]  # BF16 doubles as intermediate, not deleted
    assert not steps[0].temporary


def test_plan_reuses_existing_intermediate(jp_dir: Path):
    b = inspect_base(write_hf_dir(jp_dir / "Tiny"))
    write_gguf(jp_dir / "Tiny-BF16.gguf", ftype="BF16")
    steps = plan(JobSpec(b, ["Q4_K_M"], out_dir=jp_dir))
    assert kinds(steps) == ["move", "quantize", "verify"]  # no convert, no delete of a file we did not make


def test_plan_gguf_requant(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-Q8.gguf", ftype="Q8_0", main_type="Q8_0"))
    im = jp_dir / "im.dat"
    im.write_bytes(b"x")
    steps = plan(JobSpec(b, ["Q4_K_M"], out_dir=jp_dir, imatrix=im))
    cmd = steps[0].cmd
    assert "--allow-requantize" in cmd and cmd[cmd.index("--imatrix") + 1] == str(im)


def test_check_outputs(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-F16.gguf"))
    existing = jp_dir / "m-Q4_K_M.gguf"
    existing.write_bytes(b"x")
    errs = check_outputs(JobSpec(b, ["Q4_K_M"], out_dir=jp_dir))
    assert any("同名" in e for e in errs)
    errs = check_outputs(JobSpec(b, ["Q4_K_M"], out_dir=jp_dir, overwrite=True))
    assert not any("同名" in e for e in errs)
    errs = check_outputs(JobSpec(b, ["Q4_K_M"], out_dir=jp_dir, overwrite=True, register_ollama=True,
                                 ollama_names={"Q4_K_M": "Bad Name!"}))
    assert any("Ollama" in e for e in errs)
    assert any("1 つ以上" in e for e in check_outputs(JobSpec(b, [], out_dir=jp_dir)))


def test_log_strips_ansi():
    got = []
    r = Runner([], on_log=got.append)
    r.log("parsing GGUF \x1b[K")
    r.log("\x1b[1G\x1b[K\x1b[2K\x1b[1G")
    assert got == ["parsing GGUF "]


def test_parse_progress():
    assert parse_progress("quantize", "[  12/ 291]   blk.0.attn_q.weight - [ 2048, 2048, 1, 1]") == pytest.approx(12 / 291)
    assert parse_progress("convert", "Writing:  45%|####5     | 1.23G/2.71G [00:05<00:06]") == 0.45
    assert parse_progress("convert", "INFO:hf-to-gguf:Loading model") is None


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_runner_success_renames_partial(jp_dir: Path):
    final = jp_dir / "out.gguf"
    part = jp_dir / "out.gguf.part"
    code = (
        "import sys,pathlib\n"
        "for i in range(1,4): print(f'[ {i}/ 3] tensor', flush=True)\n"
        f"pathlib.Path(r'{part}').write_bytes(b'data')\n"
        "print('日本語の出力')\n"
    )
    logs, progress = [], []
    r = Runner([Step("t", "quantize", cmd=_py(code), partial=part, final=final)],
               on_log=logs.append, on_progress=lambda f, i, t: progress.append(f))
    r.run()
    assert final.read_bytes() == b"data" and not part.exists()
    assert "日本語の出力" in logs
    assert progress[-1] == 1.0 and any(0 < p < 1 for p in progress)


def test_runner_failure_cleans_up(jp_dir: Path):
    part = jp_dir / "x.gguf.part"
    inter = jp_dir / "inter.gguf"
    inter_part = jp_dir / "inter.gguf.part"
    mk_inter = f"import pathlib; pathlib.Path(r'{inter_part}').write_bytes(b'i')"
    boom = (f"import pathlib,sys; pathlib.Path(r'{part}').write_bytes(b'half'); "
            "print('ERROR: this quantization requires an imatrix!'); sys.exit(3)")
    steps = [Step("conv", "convert", cmd=_py(mk_inter), partial=inter_part, final=inter, temporary=True),
             Step("q", "quantize", cmd=_py(boom), partial=part, final=jp_dir / "x.gguf")]
    with pytest.raises(JobError) as ei:
        Runner(steps).run()
    assert "imatrix" in str(ei.value)
    assert not part.exists() and not inter.exists()


def test_runner_cancel(jp_dir: Path):
    part = jp_dir / "slow.gguf.part"
    code = f"import time,pathlib; pathlib.Path(r'{part}').write_bytes(b'x'); time.sleep(30)"
    r = Runner([Step("slow", "quantize", cmd=_py(code), partial=part, final=jp_dir / "slow.gguf")])
    err = []

    def go():
        try:
            r.run()
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=go)
    th.start()
    for _ in range(100):
        if part.exists():
            break
        time.sleep(0.05)
    t0 = time.monotonic()
    r.cancel()
    th.join(10)
    assert not th.is_alive() and time.monotonic() - t0 < 5
    assert isinstance(err[0], Cancelled)
    assert not part.exists()
