#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== LLMQuant install (macOS) ==="

echo "[1/5] Checking Python 3.10+..."
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "  [ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

"$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo "  [ERROR] Python 3.10 or newer is required."
    exit 1
}

echo "[2/5] Preparing virtual environment .venv..."
if [ ! -x ".venv/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv --python "$PY"
    else
        "$PY" -m venv .venv
    fi
fi
VPY="$(pwd)/.venv/bin/python"

echo "[3/5] Updating submodules (optional for MLX-only, required for GGUF)..."
if [ -d ".git" ]; then
    git submodule sync --recursive || true
    git submodule update --init --recursive --depth 1 || echo "  [WARN] submodule update skipped or failed"
fi

echo "[4/5] Installing Python dependencies..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VPY" -e ".[dev]"
    if [ -f "vendor/llama.cpp/gguf-py/pyproject.toml" ]; then
        uv pip install --python "$VPY" -e "vendor/llama.cpp/gguf-py"
    fi
else
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -e ".[dev]"
    if [ -f "vendor/llama.cpp/gguf-py/pyproject.toml" ]; then
        "$VPY" -m pip install -e "vendor/llama.cpp/gguf-py"
    fi
fi

echo "[5/5] Self-check..."
"$VPY" scripts/selfcheck.py

echo ""
echo "=== Done. Start the GUI with ./run.sh ==="

