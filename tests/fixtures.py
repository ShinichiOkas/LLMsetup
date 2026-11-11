"""Tiny synthetic model files (headers + zero data) for tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from llmquant.ggml_types import FTYPE_IDS, GGML_TYPES, tensor_nbytes

_TYPE_IDS = {v[0]: k for k, v in GGML_TYPES.items()}


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    out = _s(key)
    if isinstance(value, bool):
        return out + struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return out + struct.pack("<I", 4) + struct.pack("<I", value)
    if isinstance(value, float):
        return out + struct.pack("<I", 6) + struct.pack("<f", value)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            body = b"".join(_s(v) for v in value)
            return out + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + body
        body = b"".join(struct.pack("<i", v) for v in value)
        return out + struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", len(value)) + body
    raise TypeError(value)


def write_gguf(path: Path, arch: str = "llama", ftype: str = "F16", main_type: str | None = None,
               n_layers: int = 2, dim: int = 256, vocab: int = 512, extra: dict | None = None,
               chat_template: bool = True) -> Path:
    main_type = main_type or ftype
    meta: dict = {
        "general.architecture": arch,
        "general.name": "Tiny Test Model",
        "general.file_type": FTYPE_IDS[ftype],
        f"{arch}.context_length": 4096,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(vocab)],
    }
    if chat_template:
        meta["tokenizer.chat_template"] = "{% for m in messages %}{{ m.content }}{% endfor %}"
    meta.update(extra or {})
    tensors = [("token_embd.weight", (dim, vocab), main_type)]
    for i in range(n_layers):
        tensors.append((f"blk.{i}.attn_norm.weight", (dim,), "F32"))
        tensors.append((f"blk.{i}.attn_q.weight", (dim, dim), main_type))
        tensors.append((f"blk.{i}.ffn_up.weight", (dim, dim * 2), main_type))
    tensors.append(("output_norm.weight", (dim,), "F32"))
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta))
    head += b"".join(_kv(k, v) for k, v in meta.items())
    offset = 0
    infos = b""
    sizes = []
    for name, shape, t in tensors:
        tid = _TYPE_IDS[t]
        n = 1
        for d in shape:
            n *= d
        size = tensor_nbytes(tid, n)
        infos += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        infos += struct.pack("<I", tid) + struct.pack("<Q", offset)
        sizes.append(size)
        offset += (size + 31) // 32 * 32
    data = head + infos
    data += b"\0" * ((32 - len(data) % 32) % 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.write(b"\0" * offset)
    return path


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> Path:
    sizes = {"BF16": 2, "F16": 2, "F32": 4}
    header: dict = {"__metadata__": {"format": "pt"}}
    off = 0
    for name, (dt, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n * sizes[dt]]}
        off += n * sizes[dt]
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * off)
    return path


def write_hf_dir(d: Path, arch: str = "LlamaForCausalLM", dtype: str = "BF16", chat_template: bool = True,
                 tokenizer: bool = True, extra_cfg: dict | None = None, dim: int = 64, layers: int = 2) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"architectures": [arch], "torch_dtype": "bfloat16" if dtype == "BF16" else "float16",
           "max_position_embeddings": 8192, "hidden_size": dim}
    cfg.update(extra_cfg or {})
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if tokenizer:
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    tc = {"chat_template": "{{ messages }}"} if chat_template else {}
    (d / "tokenizer_config.json").write_text(json.dumps(tc), encoding="utf-8")
    t: dict = {"model.embed_tokens.weight": (dtype, (1000, dim))}
    for i in range(layers):
        t[f"model.layers.{i}.input_layernorm.weight"] = (dtype, (dim,))
        t[f"model.layers.{i}.self_attn.q_proj.weight"] = (dtype, (dim, dim))
        t[f"model.layers.{i}.mlp.up_proj.weight"] = (dtype, (dim * 4, dim))
    write_safetensors(d / "model.safetensors", t)
    return d
