from pathlib import Path

from fixtures import write_gguf, write_hf_dir

from llmquant.basemodel import inspect_base
from llmquant.options import STATUS_DISABLED, STATUS_OK, STATUS_WARN, Environment, build_options, recommended

GB = 1024**3


def by(opts):
    return {o.name: o for o in opts}


def test_float_gguf_options(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-F16.gguf"))
    o = by(build_options(b, Environment()))
    assert o["Q4_K_M"].status == STATUS_OK
    assert o["F16"].status == STATUS_DISABLED  # same as source
    assert o["F32"].status == STATUS_DISABLED  # only bigger
    assert o["BF16"].status == STATUS_WARN
    assert o["IQ2_XXS"].status == STATUS_DISABLED and "imatrix" in o["IQ2_XXS"].reasons[0]
    assert o["IQ3_XS"].status == STATUS_DISABLED
    assert o["TQ1_0"].status == STATUS_DISABLED  # not a ternary model
    assert o["MXFP4_MOE"].status == STATUS_DISABLED  # not MoE
    assert o["Q1_0"].status == STATUS_WARN  # new type, Ollama unverified
    assert recommended(list(o.values())).name == "Q4_K_M"  # no GPU info -> default


def test_imatrix_unlocks(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-F16.gguf"))
    im = jp_dir / "imatrix.gguf"
    im.write_bytes(b"x")
    o = by(build_options(b, Environment(), imatrix=im))
    assert o["IQ2_XXS"].status == STATUS_OK
    assert o["IQ4_XS"].notes == []


def test_requantize_rules(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-Q5.gguf", ftype="Q5_K_M", main_type="Q5_K"))
    o = by(build_options(b, Environment()))
    assert o["Q5_K_M"].status == STATUS_DISABLED
    assert o["Q8_0"].status == STATUS_DISABLED
    assert o["Q6_K"].status == STATUS_DISABLED
    assert o["Q4_K_M"].status == STATUS_WARN and "再量子化" in o["Q4_K_M"].notes[0]


def test_moe_and_ternary(jp_dir: Path):
    moe = inspect_base(write_gguf(jp_dir / "moe.gguf", extra={"llama.expert_count": 8}))
    assert by(build_options(moe))["MXFP4_MOE"].enabled
    tern = inspect_base(write_gguf(jp_dir / "bn.gguf", arch="bitnet"))
    assert by(build_options(tern))["TQ1_0"].enabled


def test_disk_space(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-F16.gguf"))
    o = by(build_options(b, Environment(out_free_bytes=10)))
    assert all(not x.enabled for x in o.values())
    assert any("空き容量" in r for r in o["Q4_K_M"].reasons)


def test_vram_recommendation(jp_dir: Path):
    b = inspect_base(write_hf_dir(jp_dir / "big", dim=64))
    # pretend it is an 8B BF16 model
    b.n_params, b.n_params_1d = 8_000_000_000, 1_000_000
    small = by(build_options(b, Environment(vram_bytes=8 * GB)))
    # 8 GB card keeps 2 GB for KV cache/desktop: Q8_0 (7.9 GiB) and Q6_K (6.1 GiB) do not fit, Q5_K_M does
    assert recommended(list(small.values())).name == "Q5_K_M"
    assert small["Q8_0"].fits_vram is False and small["Q6_K"].fits_vram is False
    tiny = by(build_options(b, Environment(vram_bytes=6 * GB)))
    assert recommended(list(tiny.values())).name == "Q4_K_M"
    huge = by(build_options(b, Environment(vram_bytes=32 * GB)))
    assert recommended(list(huge.values())).name == "Q8_0"


def test_hf_float_targets(jp_dir: Path):
    b = inspect_base(write_hf_dir(jp_dir / "m", dtype="BF16"))
    o = by(build_options(b))
    assert o["BF16"].status == STATUS_OK  # HF -> BF16 GGUF is a real product
    assert o["F16"].status == STATUS_WARN  # range loss
    assert o["F32"].status == STATUS_DISABLED


def test_estimates_are_monotonic(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "m-F16.gguf", dim=512, n_layers=4))
    o = by(build_options(b))
    assert o["Q8_0"].est_bytes > o["Q6_K"].est_bytes > o["Q4_K_M"].est_bytes > o["Q2_K"].est_bytes


def test_unsupported_base_has_no_options(jp_dir: Path):
    assert build_options(inspect_base(jp_dir / "missing")) == []
