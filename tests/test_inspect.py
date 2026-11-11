import json
from pathlib import Path

from fixtures import write_gguf, write_hf_dir, write_safetensors

from llmquant import tools
from llmquant.basemodel import KIND_GGUF, KIND_HF, KIND_UNSUPPORTED, inspect_base, ollama_name, output_filename
from llmquant.gguf_reader import read_header, split_parts


def test_gguf_header_roundtrip(jp_dir: Path):
    p = write_gguf(jp_dir / "tiny-F16.gguf", ftype="F16")
    h = read_header(p)
    assert h.architecture == "llama"
    assert h.file_type_name == "F16"
    assert h.get("tokenizer.chat_template")
    toks = h.get("tokenizer.ggml.tokens")
    assert toks.length == 512 and toks.head[:2] == ["t0", "t1"]
    assert h.n_params == 256 * 512 + 2 * (256 + 256 * 256 + 256 * 512) + 256


def test_inspect_float_gguf(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "tiny-F16.gguf"))
    assert b.kind == KIND_GGUF and b.ok
    assert b.source_type == "F16" and not b.is_quantized
    assert b.has_chat_template is True
    assert b.n_params_1d == 2 * 256 + 256
    assert b.name == "Tiny Test Model"


def test_inspect_quantized_gguf_warns(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "tiny-Q4.gguf", ftype="Q4_K_M", main_type="Q4_K"))
    assert b.ok and b.is_quantized and b.source_type == "Q4_K_M"
    assert any("量子化済み" in w for w in b.warnings)


def test_inspect_lora_and_mmproj_gguf(jp_dir: Path):
    lora = inspect_base(write_gguf(jp_dir / "lora.gguf", extra={"general.type": "adapter"}))
    assert not lora.ok and "LoRA" in lora.problems[0]
    mm = inspect_base(write_gguf(jp_dir / "mmproj.gguf", arch="clip"))
    assert not mm.ok and "mmproj" in mm.problems[0]


def test_inspect_hf_dir(jp_dir: Path):
    d = write_hf_dir(jp_dir / "Tiny-Llama")
    b = inspect_base(d)
    assert b.kind == KIND_HF
    assert b.source_type == "BF16"
    assert b.n_params == 1000 * 64 + 2 * (64 + 64 * 64 + 256 * 64)
    assert b.n_params_1d == 2 * 64
    if tools.supported_architectures() is not None:
        assert b.arch_supported is True and b.ok


def test_inspect_hf_via_file_inside(jp_dir: Path):
    d = write_hf_dir(jp_dir / "Tiny-Llama")
    b = inspect_base(d / "model.safetensors")
    assert b.kind == KIND_HF and b.model_dir == d


def test_unsupported_arch(jp_dir: Path):
    if tools.supported_architectures() is None:
        return
    b = inspect_base(write_hf_dir(jp_dir / "weird", arch="TotallyUnknownForCausalLM"))
    assert not b.ok and "対応していません" in b.problems[0]


def test_missing_tokenizer(jp_dir: Path):
    b = inspect_base(write_hf_dir(jp_dir / "notok", tokenizer=False))
    assert not b.ok and any("トークナイザ" in p for p in b.problems)


def test_single_safetensors_and_lora(jp_dir: Path):
    single = write_safetensors(jp_dir / "image_model.safetensors", {"unet.w": ("F16", (8, 8))})
    b = inspect_base(single)
    assert b.kind == KIND_UNSUPPORTED and "単体の safetensors" in b.problems[0]
    lora = write_safetensors(jp_dir / "my_lora.safetensors", {"x.lora_A.weight": ("F16", (8, 8))})
    assert "LoRA" in inspect_base(lora).problems[0]


def test_diffusers_and_adapter_dirs(jp_dir: Path):
    d = jp_dir / "sdxl"
    d.mkdir()
    (d / "model_index.json").write_text("{}", encoding="utf-8")
    assert "diffusers" in inspect_base(d).problems[0]
    a = jp_dir / "adapter"
    a.mkdir()
    (a / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    assert "LoRA" in inspect_base(a).problems[0]


def test_missing_path_and_garbage(jp_dir: Path):
    assert not inspect_base(jp_dir / "nope").ok
    g = jp_dir / "x.gguf"
    g.write_bytes(b"GGUF\x03\x00\x00\x00")  # truncated
    b = inspect_base(g)
    assert not b.ok and "GGUF" in b.problems[0]


def test_split_parts(tmp_path: Path):
    p = tmp_path / "m-00001-of-00003.gguf"
    assert [x.name for x in split_parts(p)] == [f"m-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    assert split_parts(tmp_path / "plain.gguf") == [tmp_path / "plain.gguf"]


def test_names():
    assert ollama_name("Qwen2.5-0.5B-Instruct", "Q4_K_M") == "qwen2.5-0.5b-instruct:q4_k_m"
    assert ollama_name("Tiny Test Model", "Q8_0") == "tiny-test-model:q8_0"
    assert ollama_name("model-F16", "Q4_K_M") == "model:q4_k_m"
    assert ollama_name("Qwen3-8B-Instruct-日本語版-2026年11月版", "Q6_K") == "qwen3-8b-instruct-2026-11:q6_k"
    assert output_filename("Qwen3-8B", "Q4_K_M") == "Qwen3-8B-Q4_K_M.gguf"
    assert output_filename("Qwen3-8B-F16", "Q4_K_M") == "Qwen3-8B-Q4_K_M.gguf"
    assert output_filename("a/b:c", "Q8_0") == "a-b-c-Q8_0.gguf"


def test_supported_arch_scan_has_common_models():
    archs = tools.supported_architectures()
    if archs is None:
        return
    for a in ("LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM", "Gemma2ForCausalLM"):
        assert a in archs


def test_gguf_stem_drops_type_and_split(jp_dir: Path):
    b = inspect_base(write_gguf(jp_dir / "gemma-4-26B-A4B-it-Q5_K_M.gguf", ftype="Q5_K_M", main_type="Q5_K"))
    assert b.name == "Tiny Test Model" and b.stem == "gemma-4-26B-A4B-it"
    assert output_filename(b.stem, "Q4_K_M") == "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    assert ollama_name(b.stem, "Q4_K_M") == "gemma-4-26b-a4b-it:q4_k_m"
    s = inspect_base(write_gguf(jp_dir / "big-model-f16-00001-of-00001.gguf"))
    assert s.stem == "big-model"
