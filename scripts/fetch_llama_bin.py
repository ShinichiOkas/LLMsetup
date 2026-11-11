"""Download the pinned llama.cpp Windows binaries into vendor/llama-bin.

The release (tag b12110 = v0.9.0) must match the vendor/llama.cpp submodule
commit. The archive is checked against the sha256 GitHub publishes for it;
a mismatch is fatal. Re-running is a no-op when VERSION already matches.

Exit codes: 0 ok, 1 failure.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llmquant.tools import LLAMA_TAG, llama_bin_dir  # noqa: E402

ASSET = f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{ASSET}"
SHA256 = "98901e327c252875accaf2d1bf364e10e868843b0b63468baad1aa945cdc5da2"
NEEDED = ["llama-quantize.exe"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if os.name != "nt":
        print("[fetch] not Windows: build llama.cpp yourself and set LLMQUANT_LLAMA_BIN")
        return 1
    dest = llama_bin_dir()
    ver = dest / "VERSION"
    if ver.is_file() and ver.read_text(encoding="utf-8").strip() == LLAMA_TAG and all((dest / n).is_file() for n in NEEDED):
        print(f"[fetch] llama.cpp {LLAMA_TAG} binaries already present in {dest}")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="llmquant-fetch-"))
    try:
        zpath = tmp / ASSET
        print(f"[fetch] downloading {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "llmquant-installer"})
        with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        got = _sha256(zpath)
        if got != SHA256:
            print(f"[fetch] ERROR sha256 mismatch: expected {SHA256}, got {got}")
            return 1
        print(f"[fetch] sha256 ok ({got[:16]}...)")
        staging = tmp / "x"
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                target = (staging / m).resolve()
                if not str(target).startswith(str(staging.resolve())):
                    print(f"[fetch] ERROR unsafe path in archive: {m}")
                    return 1
            z.extractall(staging)
        # some archives nest everything in a folder
        root = staging
        if not (root / NEEDED[0]).is_file():
            hits = list(staging.rglob(NEEDED[0]))
            if not hits:
                print(f"[fetch] ERROR {NEEDED[0]} not found in archive")
                return 1
            root = hits[0].parent
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, dest)
        (dest / "VERSION").write_text(LLAMA_TAG + "\n", encoding="utf-8")
        print(f"[fetch] installed llama.cpp {LLAMA_TAG} binaries into {dest}")
        return 0
    except OSError as e:
        print(f"[fetch] ERROR {e}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
