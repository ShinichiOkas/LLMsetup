"""Which quantization types are possible for a given base, and why not."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import catalog, tools
from .basemodel import KIND_HF, BaseInfo, human_bytes
from .catalog import (
    IMATRIX_RECOMMENDED,
    IMATRIX_REQUIRED,
    OLLAMA_UNVERIFIED,
    QuantType,
)
from .ggml_types import FLOAT_TYPES

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_DISABLED = "disabled"

def vram_headroom(vram: int) -> int:
    """Room left on the GPU for the KV cache, runtime and desktop when judging "fits"."""
    return max(2 * 1024**3, int(vram * 0.2))


def fits_vram(size: int, vram: int | None) -> bool | None:
    if not vram:
        return None
    return size + vram_headroom(vram) <= vram

# Order in which the recommendation is picked: best quality that fits the GPU.
# Q4_K_M is the floor: below it quality drops sharply, and Ollama can offload
# part of a Q4_K_M model to the CPU instead.
RECOMMEND_ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
MLX_RECOMMEND_ORDER = ["MLX-8bit", "MLX-6bit", "MLX-4bit"]

_FLOAT_BITS = {"F32": 32, "F16": 16, "BF16": 16}


@dataclass
class Option:
    qtype: QuantType
    est_bytes: int
    status: str = STATUS_OK
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    fits_vram: bool | None = None
    recommended: bool = False
    exact_bytes: int | None = None  # filled by a llama-quantize --dry-run

    @property
    def name(self) -> str:
        return self.qtype.name

    @property
    def enabled(self) -> bool:
        return self.status != STATUS_DISABLED

    @property
    def size_bytes(self) -> int:
        return self.exact_bytes if self.exact_bytes is not None else self.est_bytes


@dataclass
class Environment:
    vram_bytes: int | None = None
    out_free_bytes: int | None = None
    work_free_bytes: int | None = None
    same_drive: bool = True

    @classmethod
    def detect(cls, out_dir: Path | None, work_dir: Path | None = None) -> "Environment":
        out_free = tools.free_bytes(out_dir) if out_dir else None
        work_free = tools.free_bytes(work_dir) if work_dir else out_free
        same = True
        if out_dir and work_dir:
            same = Path(out_dir).resolve().drive.lower() == Path(work_dir).resolve().drive.lower()
        return cls(tools.gpu_vram_bytes(), out_free, work_free, same)


def estimate_bytes(base: BaseInfo, q: QuantType) -> int:
    """Size estimate: 1-D tensors stay F32, the rest at the mix's effective bpw."""
    if base.n_params <= 0:
        return 0
    return int(base.n_params_1d * 4 + base.n_params_2d * q.bpw / 8) + 8 * 1024 * 1024


def intermediate_type(base: BaseInfo) -> str:
    """16-bit float type the HF converter writes before llama-quantize runs."""
    return "BF16" if base.source_type == "BF16" else "F16"


def build_options(
    base: BaseInfo,
    env: Environment | None = None,
    imatrix: Path | None = None,
    target_format: str = catalog.FMT_GGUF,
) -> list[Option]:
    if target_format == catalog.FMT_MLX:
        return _build_mlx_options(base, env)
    if not base.ok_for(catalog.FMT_GGUF):
        return []
    env = env or Environment()
    have_imatrix = bool(imatrix and Path(imatrix).is_file())
    src = (base.source_type or "").upper()
    src_q = catalog.BY_NAME.get(src)
    inter_bytes = 0
    if base.kind == KIND_HF:
        inter_bytes = estimate_bytes(base, catalog.get(intermediate_type(base)))

    opts: list[Option] = []
    for q in catalog.CATALOG:
        o = Option(q, estimate_bytes(base, q))

        # --- what the source allows
        if base.is_quantized:
            if q.name == src:
                o.status, o.reasons = STATUS_DISABLED, ["元と同じ形式です"]
            elif src_q and q.bpw >= src_q.bpw:
                o.status = STATUS_DISABLED
                o.reasons.append(f"元（{src}）より大きくなるだけで、失った品質は戻りません")
            else:
                o.status = STATUS_WARN
                o.notes.append(f"量子化済み（{src}）からの再量子化。品質が二重に落ちます")
        elif q.name in _FLOAT_BITS and src in _FLOAT_BITS:
            if _FLOAT_BITS[q.name] > _FLOAT_BITS[src]:
                o.status = STATUS_DISABLED
                o.reasons.append(f"元が {src} なので、大きくなるだけで精度は上がりません")
            elif src == "BF16" and q.name == "F16":
                o.status = STATUS_WARN
                o.notes.append("BF16 → F16 は値の範囲が狭まり、まれに溢れます。BF16 を推奨")
            elif src == "F16" and q.name == "BF16":
                o.status = STATUS_WARN
                o.notes.append("F16 → BF16 は仮数部が減るだけで利点がありません。F16 を推奨")
            elif q.name == src and base.kind != KIND_HF:
                o.status, o.reasons = STATUS_DISABLED, ["元と同じ形式です"]

        # --- arch-restricted types
        if q.special == "ternary" and not base.is_ternary:
            o.status = STATUS_DISABLED
            o.reasons.append("三値で学習したモデル（BitNet 等）専用。通常のモデルは壊れます")
        if q.special == "moe" and not base.is_moe:
            o.status = STATUS_DISABLED
            o.reasons.append("MoE（専門家混合）モデル専用です")

        # --- imatrix
        if q.imatrix == IMATRIX_REQUIRED and not have_imatrix:
            o.status = STATUS_DISABLED
            o.reasons.append("imatrix ファイルが必要です（下の欄で指定）")
        elif q.imatrix == IMATRIX_RECOMMENDED and not have_imatrix and o.status != STATUS_DISABLED:
            o.notes.append("imatrix があると品質が上がります")

        # --- Ollama
        if q.ollama == OLLAMA_UNVERIFIED and o.status != STATUS_DISABLED:
            o.status = STATUS_WARN
            o.notes.append(f"llama.cpp {tools.LLAMA_RELEASE} の新しい型。Ollama で読めない恐れがあります")

        # --- disk
        need_out = o.est_bytes
        need_work = inter_bytes if (base.kind == KIND_HF and q.name != intermediate_type(base)) else 0
        if env.out_free_bytes is not None:
            need = need_out + (need_work if env.same_drive else 0)
            if need > env.out_free_bytes:
                o.status = STATUS_DISABLED
                o.reasons.append(f"空き容量が足りません（必要 {human_bytes(need)} / 空き {human_bytes(env.out_free_bytes)}）")
        if not env.same_drive and env.work_free_bytes is not None and need_work > env.work_free_bytes:
            o.status = STATUS_DISABLED
            o.reasons.append(f"作業フォルダの空きが足りません（必要 {human_bytes(need_work)}）")

        o.fits_vram = fits_vram(o.est_bytes, env.vram_bytes)
        opts.append(o)

    by = {o.name: o for o in opts}
    rec = None
    for name in RECOMMEND_ORDER:
        o = by.get(name)
        if o and o.status == STATUS_OK and (o.fits_vram is None or o.fits_vram):
            if o.fits_vram is None and name != catalog.DEFAULT_RECOMMENDED:
                continue  # no GPU info: stay with the default
            rec = o
            break
    if rec is None:
        o = by.get(catalog.DEFAULT_RECOMMENDED)
        if o and o.enabled:
            rec = o
    if rec is None:
        rec = next((o for o in opts if o.status == STATUS_OK and o.qtype.group != "float"), None)
    if rec:
        rec.recommended = True
    return opts


def _build_mlx_options(base: BaseInfo, env: Environment | None = None) -> list[Option]:
    if not base.ok_for(catalog.FMT_MLX):
        return []
    env = env or Environment()
    src = (base.source_type or "").upper()
    opts: list[Option] = []
    for q in catalog.MLX_CATALOG:
        o = Option(q, estimate_bytes(base, q))
        if q.mlx_dtype == "bfloat16" and src == "F16":
            o.status = STATUS_WARN
            o.notes.append("F16 → BF16 は仮数部が減るだけで利点がありません。MLX-FP16 を推奨")
        elif q.mlx_dtype == "float16" and src == "BF16":
            o.status = STATUS_WARN
            o.notes.append("BF16 → F16 は値の範囲が狭まり、まれに溢れます。MLX-BF16 を推奨")
        if env.out_free_bytes is not None and o.est_bytes > env.out_free_bytes:
            o.status = STATUS_DISABLED
            o.reasons.append(f"空き容量が足りません（必要 {human_bytes(o.est_bytes)} / 空き {human_bytes(env.out_free_bytes)}）")
        o.fits_vram = fits_vram(o.est_bytes, env.vram_bytes)
        opts.append(o)

    by = {o.name: o for o in opts}
    rec = None
    for name in MLX_RECOMMEND_ORDER:
        o = by.get(name)
        if o and o.status == STATUS_OK and (o.fits_vram is None or o.fits_vram):
            if o.fits_vram is None and name != catalog.DEFAULT_MLX_RECOMMENDED:
                continue
            rec = o
            break
    if rec is None:
        o = by.get(catalog.DEFAULT_MLX_RECOMMENDED)
        if o and o.enabled:
            rec = o
    if rec is None:
        rec = next((o for o in opts if o.status == STATUS_OK and o.qtype.group != "float"), None)
    if rec:
        rec.recommended = True
    return opts


def recommended(opts: list[Option]) -> Option | None:
    return next((o for o in opts if o.recommended), None)


def is_float(name: str) -> bool:
    return name.upper() in FLOAT_TYPES
