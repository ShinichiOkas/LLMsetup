"""Inspecting the base the user picked: an HF model folder or a GGUF file.

Everything here reads headers only (GGUF header, safetensors JSON header,
config.json) so it answers in milliseconds even for 100 GB models.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import tools
from .ggml_types import FLOAT_TYPES, LLAMA_FTYPES
from .gguf_reader import GGUFError, is_gguf, read_all_parts

KIND_HF = "hf"
KIND_GGUF = "gguf"
KIND_UNSUPPORTED = "unsupported"

_ST_DTYPES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "U32": 4, "U64": 8,
    "F4": 1, "F6_E2M3": 1, "F6_E3M2": 1,
}


@dataclass
class BaseInfo:
    path: Path
    kind: str
    name: str = ""
    architecture: str | None = None
    arch_supported: bool | None = None
    source_type: str | None = None  # "BF16"/"F16"/"F32" or a quant ftype name
    is_quantized: bool = False
    n_params: int = 0
    n_params_1d: int = 0  # norms/biases, kept as F32 by the quantizer
    weight_bytes: int = 0  # bytes of weights on disk
    is_moe: bool = False
    is_ternary: bool = False
    has_chat_template: bool | None = None
    context_length: int | None = None
    model_dir: Path | None = None  # HF folder
    gguf_parts: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # fatal: no options possible
    warnings: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)
    file_stem: str = ""  # base for output file / Ollama names (file or folder name)

    @property
    def stem(self) -> str:
        return self.file_stem or self.name

    @property
    def ok(self) -> bool:
        return self.kind != KIND_UNSUPPORTED and not self.problems

    @property
    def n_params_2d(self) -> int:
        return max(self.n_params - self.n_params_1d, 0)


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def human_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


# ---------------------------------------------------------------- safetensors

def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError("not a safetensors file (bad header length)")
        return json.loads(f.read(n).decode("utf-8"))


def _looks_like_lora(keys: list[str]) -> bool:
    return any(("lora_A" in k or "lora_B" in k or "lora_down" in k or "lora_up" in k) for k in keys)


def _weight_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            wm = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            files = sorted({model_dir / v for v in wm.values()})
            if files and all(f.is_file() for f in files):
                return files
        except (OSError, ValueError):
            pass
    files = sorted(model_dir.glob("model*.safetensors"))
    if not files:
        files = sorted(p for p in model_dir.glob("*.safetensors") if not p.name.startswith("consolidated"))
    if not files:
        files = sorted(model_dir.glob("*.safetensors"))
    return files


def _count_safetensors(files: list[Path]) -> tuple[int, int, int, dict[str, int]]:
    n = n1 = nbytes = 0
    dtypes: dict[str, int] = {}
    for f in files:
        hdr = read_safetensors_header(f)
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            shape = v.get("shape", [])
            cnt = 1
            for d in shape:
                cnt *= int(d)
            n += cnt
            if len(shape) <= 1:
                n1 += cnt
            dt = str(v.get("dtype", "?")).upper()
            dtypes[dt] = dtypes.get(dt, 0) + cnt
            off = v.get("data_offsets")
            if off and len(off) == 2:
                nbytes += int(off[1]) - int(off[0])
            else:
                nbytes += cnt * _ST_DTYPES.get(dt, 2)
    return n, n1, nbytes, dtypes


# ---------------------------------------------------------------- HF folder

def _sub(cfg: dict) -> dict:
    for key in ("text_config", "llm_config", "language_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict):
            return sub
    return cfg


def _inspect_hf(model_dir: Path, picked: Path) -> BaseInfo:
    info = BaseInfo(path=picked, kind=KIND_HF, model_dir=model_dir, name=model_dir.name, file_stem=model_dir.name)
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        info.kind = KIND_UNSUPPORTED
        info.problems.append(f"config.json を読めません: {e}")
        return info

    archs = cfg.get("architectures") or []
    info.architecture = archs[0] if archs else None
    text = _sub(cfg)
    info.context_length = text.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    for key in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
        if (text.get(key) or cfg.get(key) or 0) > 1:
            info.is_moe = True
    if info.architecture and "bitnet" in info.architecture.lower():
        info.is_ternary = True

    supported = tools.supported_architectures()
    if not archs:
        info.arch_supported = False
        info.problems.append("config.json に architectures がありません（LLM の HF 形式ではないようです）")
    elif supported is None:
        info.arch_supported = None
        info.warnings.append("変換器（llama.cpp）が未取得のため、対応構造を確かめられません。install.bat を実行してください")
    else:
        hit = [a for a in archs if a in supported]
        info.arch_supported = bool(hit)
        if hit:
            info.architecture = hit[0]
        else:
            info.problems.append(
                f"構造 {', '.join(archs)} は llama.cpp {tools.LLAMA_RELEASE} の変換器が対応していません"
            )

    files = _weight_files(model_dir)
    if files:
        try:
            n, n1, nbytes, dtypes = _count_safetensors(files)
            info.n_params, info.n_params_1d, info.weight_bytes = n, n1, nbytes
            if dtypes:
                main = max(dtypes, key=lambda k: dtypes[k])
                info.source_type = {"F16": "F16", "BF16": "BF16", "F32": "F32"}.get(main, main)
                if main.startswith("F8") or main in ("I8", "U8", "I32", "F4"):
                    info.warnings.append(f"重みが {main} で保存されています（事前量子化済みの HF モデル）。変換器が展開できない場合があります")
            info.details["重みファイル"] = f"safetensors {len(files)} 個"
        except (OSError, ValueError) as e:
            info.problems.append(f"safetensors のヘッダを読めません: {e}")
    else:
        bins = sorted(model_dir.glob("pytorch_model*.bin"))
        if bins:
            info.weight_bytes = sum(p.stat().st_size for p in bins)
            dt = str(cfg.get("torch_dtype") or text.get("torch_dtype") or cfg.get("dtype") or "float16")
            bytes_per = 4 if "32" in dt else 2
            info.n_params = info.weight_bytes // bytes_per
            info.source_type = "BF16" if "bfloat" in dt else ("F32" if "32" in dt else "F16")
            info.details["重みファイル"] = f"pytorch .bin {len(bins)} 個（パラメータ数は推定）"
        else:
            info.problems.append("重みファイル（*.safetensors / pytorch_model*.bin）が見つかりません")

    cfg_dtype = str(cfg.get("torch_dtype") or text.get("torch_dtype") or cfg.get("dtype") or "")
    if info.source_type is None and cfg_dtype:
        info.source_type = "BF16" if "bfloat" in cfg_dtype else ("F32" if "32" in cfg_dtype else "F16")

    tok_files = ["tokenizer.json", "tokenizer.model", "vocab.json", "tekken.json"]
    if not any((model_dir / t).is_file() for t in tok_files):
        info.problems.append("トークナイザ（tokenizer.json / tokenizer.model など）がありません")

    tmpl = (model_dir / "chat_template.jinja").is_file() or (model_dir / "chat_template.json").is_file()
    if not tmpl:
        try:
            tc = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
            tmpl = bool(tc.get("chat_template"))
        except (OSError, ValueError):
            tmpl = False
    info.has_chat_template = tmpl
    if not tmpl:
        info.warnings.append("チャットテンプレートがありません。Ollama では素の補完モデルとして扱われます")

    name = cfg.get("_name_or_path")
    if isinstance(name, str) and name and not Path(name).is_absolute():
        info.name = name.rstrip("/").split("/")[-1]
    return info


# ---------------------------------------------------------------- GGUF

def _inspect_gguf(path: Path) -> BaseInfo:
    stem = SPLIT_STEM_RE.sub("", path.stem)
    info = BaseInfo(path=path, kind=KIND_GGUF, name=stem, file_stem=strip_type_suffix(stem))
    try:
        hdr, parts = read_all_parts(path)
    except (OSError, GGUFError) as e:
        info.kind = KIND_UNSUPPORTED
        info.problems.append(f"GGUF を読めません: {e}")
        return info
    missing = [p for p in parts if not p.exists()]
    if missing:
        info.problems.append(f"分割 GGUF の一部が見つかりません: {', '.join(p.name for p in missing)}")
    info.gguf_parts = parts
    arch = hdr.architecture
    info.architecture = arch
    gtype = hdr.get("general.type")
    if gtype == "adapter":
        info.problems.append("LoRA アダプタの GGUF です。単体では量子化できません（ベースモデルにマージが必要）")
    if arch == "clip" or gtype == "mmproj":
        info.problems.append("画像用プロジェクタ（mmproj）の GGUF です。LLM 本体を指定してください")
    name = hdr.get("general.name")
    if isinstance(name, str) and name.strip():
        info.name = name.strip()
    info.n_params = hdr.n_params
    info.n_params_1d = sum(t.n_elements for t in hdr.tensors if len(t.shape) <= 1 or sum(1 for d in t.shape if d > 1) <= 1)
    info.weight_bytes = sum(p.stat().st_size for p in parts if p.exists())

    hist = hdr.type_histogram()
    big = {k: v for k, v in hist.items() if k not in ("F32",)} or hist
    main_tensor_type = max(big, key=lambda k: big[k]) if big else None
    ft = hdr.file_type_name
    if ft and ft != "GUESSED":
        info.source_type = ft
    else:
        info.source_type = main_tensor_type
    info.is_quantized = bool(main_tensor_type and main_tensor_type not in FLOAT_TYPES)
    if info.is_quantized and info.source_type in FLOAT_TYPES:
        info.source_type = main_tensor_type
    if arch:
        info.context_length = hdr.get(f"{arch}.context_length")
        if (hdr.get(f"{arch}.expert_count") or 0) > 1:
            info.is_moe = True
        if "bitnet" in arch.lower():
            info.is_ternary = True
    info.has_chat_template = bool(hdr.get("tokenizer.chat_template"))
    if not info.has_chat_template:
        info.warnings.append("GGUF にチャットテンプレートがありません。Ollama では素の補完モデルとして扱われます")
    if info.is_quantized:
        info.warnings.append(
            f"この GGUF は量子化済み（{info.source_type}）です。ここから更に量子化すると品質が二重に落ちます。"
            "可能なら元の F16/BF16 から作ってください"
        )
    if len(parts) > 1:
        info.details["分割"] = f"{len(parts)} ファイル（出力は 1 ファイルにまとめます）"
    tensor_types = ", ".join(f"{k} {v / max(info.n_params, 1):.0%}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:4])
    info.details["テンソルの型"] = tensor_types
    return info


# ---------------------------------------------------------------- entry

def inspect_base(path: str | Path) -> BaseInfo:
    """Classify what the user picked and read what the options depend on."""
    p = Path(str(path).strip().strip('"'))
    if not p.exists():
        return BaseInfo(path=p, kind=KIND_UNSUPPORTED, name=p.name, problems=["指定したパスがありません"])

    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".gguf" or is_gguf(p):
            return _inspect_gguf(p)
        if suffix in (".safetensors", ".bin", ".json", ".model") and (p.parent / "config.json").is_file():
            return _inspect_hf(p.parent, p)
        if suffix == ".safetensors":
            info = BaseInfo(path=p, kind=KIND_UNSUPPORTED, name=p.stem)
            try:
                hdr = read_safetensors_header(p)
                keys = [k for k in hdr if k != "__metadata__"]
                if _looks_like_lora(keys):
                    info.problems.append("LoRA アダプタの safetensors です。単体では量子化できません（ベースモデルにマージが必要）")
                else:
                    info.problems.append(
                        "単体の safetensors です。LLM の変換には config.json とトークナイザの入ったフォルダが必要です"
                        "（画像生成モデル等の単体ファイルは対象外）"
                    )
            except (OSError, ValueError) as e:
                info.problems.append(f"safetensors を読めません: {e}")
            return info
        return BaseInfo(path=p, kind=KIND_UNSUPPORTED, name=p.name,
                        problems=["対応していない形式です（HF モデルのフォルダか .gguf を指定してください）"])

    # directory
    if (p / "config.json").is_file():
        return _inspect_hf(p, p)
    info = BaseInfo(path=p, kind=KIND_UNSUPPORTED, name=p.name)
    if (p / "model_index.json").is_file():
        info.problems.append("画像生成（diffusers）モデルのフォルダです。LLM ではないため対象外です")
    elif (p / "adapter_config.json").is_file():
        info.problems.append("LoRA アダプタのフォルダです。単体では量子化できません（ベースモデルにマージが必要）")
    else:
        ggufs = sorted(p.glob("*.gguf"))
        if ggufs:
            info.problems.append(f"フォルダに GGUF が {len(ggufs)} 個あります。量子化したい .gguf ファイルを指定してください")
        else:
            info.problems.append("config.json がありません。HF 形式の LLM フォルダか .gguf を指定してください")
    return info


_SANITIZE_RE = re.compile(r"[^a-z0-9._-]+")
SPLIT_STEM_RE = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_TYPE_SUFFIX_RE = re.compile(r"[-_.](f32|f16|bf16|q\d\w*|iq\d\w*|tq\d\w*|mxfp4\w*)$", re.IGNORECASE)


def strip_type_suffix(stem: str) -> str:
    """``gemma-4-it-Q5_K_M`` -> ``gemma-4-it`` (the type is re-added for outputs)."""
    return _TYPE_SUFFIX_RE.sub("", stem) or stem


def ollama_name(base_name: str, qtype: str) -> str:
    """Ollama model name ``<name>:<tag>`` built from the base name and type."""
    stem = _SANITIZE_RE.sub("-", strip_type_suffix(base_name).lower())
    stem = re.sub(r"-{2,}", "-", stem).strip("-.") or "model"
    return f"{stem[:60].rstrip('-.')}:{qtype.lower()}"


def output_filename(base_name: str, qtype: str) -> str:
    stem = re.sub(r'[\\/:*?"<>|]+', "-", base_name).strip() or "model"
    return f"{strip_type_suffix(stem)}-{qtype}.gguf"


__all__ = [
    "BaseInfo", "inspect_base", "human_bytes", "human_params", "ollama_name", "output_filename",
    "KIND_HF", "KIND_GGUF", "KIND_UNSUPPORTED", "LLAMA_FTYPES",
]
