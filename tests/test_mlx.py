import json
from pathlib import Path

from PySide6.QtWidgets import QApplication
from fixtures import write_gguf, write_hf_dir

from llmquant import catalog, cli, tools
from llmquant.basemodel import KIND_HF, KIND_MLX, inspect_base, output_mlx_dirname
from llmquant.gui import MainWindow
from llmquant.options import STATUS_DISABLED, Environment, build_options, recommended
from llmquant.pipeline import JobSpec, check_outputs, plan


def _write_hf_mlx_dir(d: Path, model_type: str = "llama", arch: str = "LlamaForCausalLM") -> Path:
    write_hf_dir(d, arch=arch)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = model_type
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _write_mlx_quantized_dir(d: Path, bits: int = 4, group_size: int = 64, mode: str = "affine") -> Path:
    _write_hf_mlx_dir(d, model_type="llama")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg["quantization"] = {"group_size": group_size, "bits": bits, "mode": mode}
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def test_mlx_supported_model_types():
    if not tools.mlx_lm_available():
        return
    mtypes = tools.mlx_supported_model_types()
    assert mtypes is not None
    for t in ("llama", "qwen2", "mistral"):
        assert t in mtypes


def test_inspect_hf_for_mlx(jp_dir: Path, monkeypatch):
    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_supported_model_types", lambda: frozenset({"llama", "qwen2"}))

    d = _write_hf_mlx_dir(jp_dir / "Tiny-Llama")
    b = inspect_base(d)
    assert b.kind == KIND_HF
    assert b.mlx_model_type == "llama"
    assert b.mlx_supported is True
    assert b.ok_for(catalog.FMT_MLX)

    # Unknown model_type is rejected for MLX
    bad_dir = write_hf_dir(jp_dir / "Weird-Model")
    cfg = json.loads((bad_dir / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "totally_unknown_mlx_arch"
    (bad_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    b2 = inspect_base(bad_dir)
    assert not b2.ok_for(catalog.FMT_MLX)
    assert any("mlx-lm" in p for p in b2.problems_for(catalog.FMT_MLX))


def test_inspect_prequantized_mlx_dir(jp_dir: Path, monkeypatch):
    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_supported_model_types", lambda: frozenset({"llama"}))

    d = _write_mlx_quantized_dir(jp_dir / "Tiny-Llama-4bit", bits=4, group_size=64)
    b = inspect_base(d)
    assert b.kind == KIND_MLX
    assert b.is_quantized is True
    assert b.source_type == "MLX-4bit"
    assert "4bit (group_size=64" in b.details["MLX 量子化"]
    assert not b.ok_for(catalog.FMT_MLX)
    assert any("すでに MLX 形式" in p for p in b.problems_for(catalog.FMT_MLX))


def test_inspect_gguf_rejected_for_mlx(jp_dir: Path):
    g = write_gguf(jp_dir / "tiny-F16.gguf")
    b = inspect_base(g)
    assert not b.ok_for(catalog.FMT_MLX)
    assert any("GGUF" in p for p in b.problems_for(catalog.FMT_MLX))


def test_mlx_options_and_recommendations(jp_dir: Path, monkeypatch):
    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_supported_model_types", lambda: frozenset({"llama"}))

    b = inspect_base(_write_hf_mlx_dir(jp_dir / "Tiny-Llama"))
    opts = build_options(b, Environment(vram_bytes=None, out_free_bytes=10 * 1024**3), target_format=catalog.FMT_MLX)
    assert len(opts) == len(catalog.MLX_CATALOG)
    rec = recommended(opts)
    assert rec is not None and rec.name == "MLX-4bit"

    # Low disk disables options
    opts_low = build_options(b, Environment(vram_bytes=None, out_free_bytes=10), target_format=catalog.FMT_MLX)
    assert all(o.status == STATUS_DISABLED for o in opts_low)


def test_mlx_pipeline_plan_and_checks(jp_dir: Path, monkeypatch):
    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_lm_available", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_supported_model_types", lambda: frozenset({"llama"}))

    b = inspect_base(_write_hf_mlx_dir(jp_dir / "Tiny-Llama"))
    out_dir = jp_dir / "out"
    out_dir.mkdir()
    spec = JobSpec(base=b, targets=["MLX-4bit", "MLX-BF16"], out_dir=out_dir, target_format=catalog.FMT_MLX)
    assert spec.output_path("MLX-4bit") == out_dir / "Tiny-Llama-MLX-4bit-g64"
    assert spec.output_path("MLX-mxfp4") == out_dir / "Tiny-Llama-MLX-mxfp4-g32"
    assert spec.output_path("MLX-BF16") == out_dir / "Tiny-Llama-MLX-BF16"
    assert check_outputs(spec) == []

    steps = plan(spec)
    assert len(steps) == 4
    assert steps[0].kind == "mlx_convert"
    assert "-m" in steps[0].cmd and "mlx_lm" in steps[0].cmd and "convert" in steps[0].cmd
    assert "--q-bits" in steps[0].cmd and "4" in steps[0].cmd
    assert "--dtype" in steps[2].cmd and "bfloat16" in steps[2].cmd

    # Existing output dir blocks without overwrite
    (out_dir / output_mlx_dirname(b.stem, "MLX-4bit")).mkdir()
    errs = check_outputs(spec)
    assert any("同名のファイルまたはフォルダ" in e for e in errs)
    spec.overwrite = True
    assert check_outputs(spec) == []


def test_cli_inspect_mlx_json(jp_dir: Path, capsys, monkeypatch):
    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_supported_model_types", lambda: frozenset({"llama"}))

    d = _write_hf_mlx_dir(jp_dir / "Tiny-Llama")
    rc = cli.main(["inspect", str(d), "--format", "mlx", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["format"] == "mlx"
    assert data["mlx_model_type"] == "llama"
    assert any(o["recommended"] for o in data["options"])


def test_gui_switch_to_mlx(jp_dir: Path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    assert app is not None
    monkeypatch.setattr("llmquant.tools.is_apple_silicon", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_lm_available", lambda: True)
    monkeypatch.setattr("llmquant.tools.mlx_supported_model_types", lambda: frozenset({"llama"}))

    d = _write_hf_mlx_dir(jp_dir / "Tiny-Llama")
    w = MainWindow()
    w.load_base(str(d))
    assert w.fmt_mlx_radio.isEnabled()

    w.fmt_mlx_radio.setChecked(True)
    assert w.target_format() == catalog.FMT_MLX
    assert not w.ollama_group.isVisible()
    assert not w.imatrix_edit.isVisible()
    assert len(w.selected_types()) == 1
    assert w.selected_types()[0].startswith("MLX-")
    assert not w._spec().register_ollama
    assert w._spec().target_format == catalog.FMT_MLX
