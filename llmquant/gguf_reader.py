"""Minimal GGUF header reader (metadata + tensor infos, no tensor data).

Pure Python and dependency-free so the GUI can inspect a 30 GB file in
milliseconds without importing the converter's package.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from .ggml_types import LLAMA_FTYPES, tensor_nbytes, type_name

GGUF_MAGIC = b"GGUF"

# value type id -> struct format (scalars)
_SCALARS = {
    0: "<B",  # UINT8
    1: "<b",  # INT8
    2: "<H",  # UINT16
    3: "<h",  # INT16
    4: "<I",  # UINT32
    5: "<i",  # INT32
    6: "<f",  # FLOAT32
    7: "<?",  # BOOL
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}
_STRING = 8
_ARRAY = 9

# Arrays longer than this are not materialised (tokenizer vocabularies are
# 100k+ strings); only their length is kept.
MAX_ARRAY_ITEMS = 64

SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


class GGUFError(ValueError):
    pass


@dataclass
class ArraySummary:
    item_type: int
    length: int
    head: list[Any]


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    type_id: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def type_name(self) -> str:
        return type_name(self.type_id)

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.type_id, self.n_elements)


@dataclass
class GGUFHeader:
    path: Path
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    @property
    def file_type_id(self) -> int | None:
        v = self.metadata.get("general.file_type")
        return int(v) if v is not None else None

    @property
    def file_type_name(self) -> str | None:
        ft = self.file_type_id
        if ft is None:
            return None
        return LLAMA_FTYPES.get(ft, f"ftype{ft}")

    @property
    def n_params(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    @property
    def tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def type_histogram(self) -> dict[str, int]:
        """Number of elements per tensor type."""
        hist: dict[str, int] = {}
        for t in self.tensors:
            hist[t.type_name] = hist.get(t.type_name, 0) + t.n_elements
        return hist

    @property
    def split_count(self) -> int:
        v = self.metadata.get("split.count")
        return int(v) if v else 1


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("unexpected end of file (truncated GGUF header)")
        return b

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > 1 << 30:
            raise GGUFError(f"implausible string length {n}")
        return self.read(n).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        n = self.unpack("<Q")
        self.f.seek(n, 1)

    def value(self, vtype: int) -> Any:
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            item_type = self.unpack("<I")
            length = self.unpack("<Q")
            head: list[Any] = []
            if item_type in _SCALARS:
                size = struct.calcsize(_SCALARS[item_type])
                keep = min(length, MAX_ARRAY_ITEMS)
                for _ in range(keep):
                    head.append(self.unpack(_SCALARS[item_type]))
                self.f.seek((length - keep) * size, 1)
            elif item_type == _STRING:
                for i in range(length):
                    if i < MAX_ARRAY_ITEMS:
                        head.append(self.string())
                    else:
                        self.skip_string()
            elif item_type == _ARRAY:
                for _ in range(length):
                    self.value(_ARRAY)
            else:
                raise GGUFError(f"unknown array item type {item_type}")
            if length <= MAX_ARRAY_ITEMS:
                return head
            return ArraySummary(item_type, length, head)
        raise GGUFError(f"unknown metadata value type {vtype}")


def is_gguf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def read_header(path: str | Path) -> GGUFHeader:
    path = Path(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.read(4) != GGUF_MAGIC:
            raise GGUFError("not a GGUF file (magic mismatch)")
        version = r.unpack("<I")
        if version < 2 or version > 3:
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        hdr = GGUFHeader(path=path, version=version)
        for _ in range(n_kv):
            key = r.string()
            vtype = r.unpack("<I")
            hdr.metadata[key] = r.value(vtype)
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            if n_dims > 8:
                raise GGUFError(f"implausible tensor rank {n_dims} for {name}")
            shape = tuple(r.unpack("<Q") for _ in range(n_dims))
            type_id = r.unpack("<I")
            r.unpack("<Q")  # offset
            hdr.tensors.append(TensorInfo(name, shape, type_id))
    return hdr


def split_parts(path: Path) -> list[Path]:
    """All shards of a split GGUF (``name-00001-of-00003.gguf``), or [path]."""
    m = SPLIT_RE.match(path.name)
    if not m:
        return [path]
    count = int(m.group("count"))
    stem = m.group("stem")
    return [path.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


def read_all_parts(path: str | Path) -> tuple[GGUFHeader, list[Path]]:
    """Header of the first shard with tensors of every shard merged in."""
    path = Path(path)
    parts = split_parts(path)
    first = parts[0]
    hdr = read_header(first)
    for p in parts[1:]:
        if p.exists():
            hdr.tensors.extend(read_header(p).tensors)
    return hdr, parts
