"""Catalog of llama-quantize target types with Japanese guidance.

The list mirrors QUANT_OPTIONS in vendor/llama.cpp/tools/quantize/quantize.cpp
at the pinned commit (v0.9.0 = b12110). Aliases (Q3_K, Q4_K, Q5_K) and COPY are
left out on purpose. ``bpw`` is the *effective* bits per weight of the whole
mix as measured on Llama-3-8B (upstream size table) or the base block bpw when
upstream gives no measurement; sizes derived from it are estimates ("目安").
Exact sizes come from ``llama-quantize --dry-run`` (see pipeline.dry_run_size).
"""

from __future__ import annotations

from dataclasses import dataclass

# Ollama compatibility levels
OLLAMA_OK = "ok"  # long-established type, loads in Ollama
OLLAMA_UNVERIFIED = "unverified"  # new in recent llama.cpp; may not load in Ollama

# imatrix needs
IMATRIX_NONE = "none"
IMATRIX_RECOMMENDED = "recommended"
IMATRIX_REQUIRED = "required"


@dataclass(frozen=True)
class QuantType:
    name: str
    bpw: float
    quality: int  # 1 (worst) .. 6 (lossless)
    group: str
    note: str
    imatrix: str = IMATRIX_NONE
    ollama: str = OLLAMA_OK
    special: str = ""  # "ternary" / "moe" for arch-restricted types


CATALOG: list[QuantType] = [
    QuantType("F32", 32.0, 6, "float", "無劣化・元の精度のまま。最大のサイズ"),
    QuantType("BF16", 16.0, 6, "float", "ほぼ無劣化。元が BF16 のモデルならこれ"),
    QuantType("F16", 16.0, 6, "float", "ほぼ無劣化。元が FP16 のモデル向け"),
    QuantType("Q8_0", 8.51, 5, "legacy", "ほぼ無劣化（+0.003 ppl）。大きいが安全"),
    QuantType("Q6_K", 6.57, 5, "k", "非常に高品質（+0.022 ppl）"),
    QuantType("Q5_K_M", 5.70, 4, "k", "高品質（+0.057 ppl）。VRAM に余裕があれば"),
    QuantType("Q5_K_S", 5.57, 4, "k", "高品質（+0.105 ppl）"),
    QuantType("Q4_K_M", 4.90, 4, "k", "品質とサイズの釣り合いが最良（+0.175 ppl）。定番"),
    QuantType("Q4_K_S", 4.67, 3, "k", "Q4_K_M より少し小さい（+0.269 ppl）"),
    QuantType("IQ4_XS", 4.46, 3, "iq", "Q4_K_S 並の品質でより小さい", IMATRIX_RECOMMENDED),
    QuantType("IQ4_NL", 4.68, 3, "iq", "4.5 bpw 非線形。imatrix があると良い", IMATRIX_RECOMMENDED),
    QuantType("Q5_1", 6.04, 4, "legacy", "旧形式（+0.106 ppl）。通常は Q5_K_M が良い"),
    QuantType("Q5_0", 5.57, 4, "legacy", "旧形式（+0.132 ppl）。通常は Q5_K_S が良い"),
    QuantType("Q4_1", 5.11, 3, "legacy", "旧形式（+0.451 ppl）"),
    QuantType("Q4_0", 4.64, 3, "legacy", "旧形式（+0.469 ppl）。古い環境との互換用"),
    QuantType("Q3_K_L", 4.31, 3, "k", "劣化が目立ち始める（+0.556 ppl）"),
    QuantType("Q3_K_M", 4.00, 2, "k", "劣化あり（+0.657 ppl）。VRAM が厳しいとき"),
    QuantType("IQ3_M", 3.76, 2, "iq", "3.66 bpw。3 bit 台では良好", IMATRIX_RECOMMENDED),
    QuantType("Q3_K_S", 3.65, 2, "k", "劣化大（+1.63 ppl）", IMATRIX_RECOMMENDED),
    QuantType("IQ3_S", 3.66, 2, "iq", "3.44 bpw", IMATRIX_RECOMMENDED),
    QuantType("IQ3_XS", 3.50, 2, "iq", "3.3 bpw。imatrix が必須", IMATRIX_REQUIRED),
    QuantType("IQ3_XXS", 3.21, 2, "iq", "3.06 bpw。imatrix が必須", IMATRIX_REQUIRED),
    QuantType("Q2_K", 3.17, 1, "k", "劣化が非常に大きい（+3.52 ppl）", IMATRIX_RECOMMENDED),
    QuantType("Q2_K_S", 3.00, 1, "k", "劣化が非常に大きい。imatrix が必須", IMATRIX_REQUIRED),
    QuantType("IQ2_M", 2.93, 1, "iq", "2.7 bpw。大きいモデルを無理に載せる用", IMATRIX_REQUIRED),
    QuantType("IQ2_S", 2.74, 1, "iq", "2.5 bpw", IMATRIX_REQUIRED),
    QuantType("IQ2_XS", 2.59, 1, "iq", "2.31 bpw", IMATRIX_REQUIRED),
    QuantType("IQ2_XXS", 2.38, 1, "iq", "2.06 bpw", IMATRIX_REQUIRED),
    QuantType("IQ1_M", 2.01, 1, "iq", "1.75 bpw。巨大モデル専用・大幅劣化", IMATRIX_REQUIRED),
    QuantType("IQ1_S", 1.88, 1, "iq", "1.56 bpw。巨大モデル専用・大幅劣化", IMATRIX_REQUIRED),
    QuantType("MXFP4_MOE", 4.60, 3, "new", "MoE の専門家層を MXFP4 に（gpt-oss 形式）", ollama=OLLAMA_UNVERIFIED, special="moe"),
    QuantType("TQ2_0", 2.06, 1, "new", "三値（BitNet 等の三値学習モデル専用）", ollama=OLLAMA_UNVERIFIED, special="ternary"),
    QuantType("TQ1_0", 1.69, 1, "new", "三値（BitNet 等の三値学習モデル専用）", ollama=OLLAMA_UNVERIFIED, special="ternary"),
    QuantType("Q2_0", 2.25, 1, "new", "新形式 2 bit（実験的）", ollama=OLLAMA_UNVERIFIED),
    QuantType("Q1_0", 1.125, 1, "new", "新形式 1 bit（実験的）", ollama=OLLAMA_UNVERIFIED),
]

BY_NAME: dict[str, QuantType] = {q.name: q for q in CATALOG}

# Types the HF converter can write directly (skips llama-quantize)
CONVERTER_OUTTYPES = {"F32": "f32", "F16": "f16", "BF16": "bf16", "Q8_0": "q8_0"}

DEFAULT_RECOMMENDED = "Q4_K_M"


def get(name: str) -> QuantType:
    return BY_NAME[name.upper()]
