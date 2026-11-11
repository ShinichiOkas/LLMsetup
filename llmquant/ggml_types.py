"""GGML tensor types and llama file types.

Copied from vendor/llama.cpp/gguf-py/gguf/constants.py at the pinned commit
(v0.9.0 = b12110 = cd959bae). Kept as a plain table so that reading a GGUF
header does not need to import the converter's package.
"""

from __future__ import annotations

QK_K = 256

# ggml_type id -> (name, block_size, type_size_bytes)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 2 + 16),
    3: ("Q4_1", 32, 2 + 2 + 16),
    6: ("Q5_0", 32, 2 + 4 + 16),
    7: ("Q5_1", 32, 2 + 2 + 4 + 16),
    8: ("Q8_0", 32, 2 + 32),
    9: ("Q8_1", 32, 2 + 2 + 32),
    10: ("Q2_K", 256, 2 + 2 + QK_K // 16 + QK_K // 4),
    11: ("Q3_K", 256, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: ("Q4_K", 256, 2 + 2 + QK_K // 2 + 12),
    13: ("Q5_K", 256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: ("Q6_K", 256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    15: ("Q8_K", 256, 4 + QK_K + QK_K // 8),
    16: ("IQ2_XXS", 256, 2 + QK_K // 4),
    17: ("IQ2_XS", 256, 2 + QK_K // 4 + QK_K // 32),
    18: ("IQ3_XXS", 256, 2 + QK_K // 4 + QK_K // 8),
    19: ("IQ1_S", 256, 2 + QK_K // 8 + QK_K // 16),
    20: ("IQ4_NL", 32, 2 + 16),
    21: ("IQ3_S", 256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: ("IQ2_S", 256, 2 + QK_K // 4 + QK_K // 16),
    23: ("IQ4_XS", 256, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 2 + 4 * 13),
    35: ("TQ2_0", 256, 2 + 64),
    39: ("MXFP4", 32, 1 + 16),
    40: ("NVFP4", 64, 4 + 32),
    41: ("Q1_0", 128, 2 + 16),
    42: ("Q2_0", 64, 2 + 16),
    43: ("F8_E4M3", 1, 1),
}

FLOAT_TYPES = {"F32", "F16", "BF16", "F64"}

# llama_ftype id -> name used by llama-quantize (None = not a quantize target)
LLAMA_FTYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
    41: "Q2_0",
    42: "F8_E4M3",
    1024: "GUESSED",
}

FTYPE_IDS: dict[str, int] = {v: k for k, v in LLAMA_FTYPES.items()}


def type_name(type_id: int) -> str:
    entry = GGML_TYPES.get(type_id)
    return entry[0] if entry else f"type{type_id}"


def tensor_nbytes(type_id: int, n_elements: int) -> int:
    entry = GGML_TYPES.get(type_id)
    if entry is None:
        return 0
    _, block, size = entry
    return (n_elements // block) * size
