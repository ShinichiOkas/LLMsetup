"""Hugging Face model downloader for LLMQuant."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from tqdm.auto import tqdm


@dataclass
class HfRepoInfo:
    repo_id: str
    is_valid: bool = False
    error_message: str = ""
    gguf_files: list[str] = field(default_factory=list)
    has_hf_weights: bool = False
    pipeline_tag: str = ""
    total_files: int = 0
    file_sizes: dict[str, int] = field(default_factory=dict)


def parse_hf_input(text: str) -> tuple[str, str | None]:
    """Parses user input (Repo ID or Hugging Face URL) into (repo_id, filename_or_none).

    Examples:
        - "Qwen/Qwen2.5-Coder-7B" -> ("Qwen/Qwen2.5-Coder-7B", None)
        - "https://huggingface.co/Qwen/Qwen2.5-Coder-7B" -> ("Qwen/Qwen2.5-Coder-7B", None)
        - "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/model-Q8_0.gguf"
          -> ("bartowski/Llama-3.2-1B-Instruct-GGUF", "model-Q8_0.gguf")
    """
    clean = text.strip()
    if not clean:
        return "", None

    # Check if URL
    if clean.startswith("http://") or clean.startswith("https://"):
        u = urlparse(clean)
        if "huggingface.co" in u.netloc:
            parts = [p for p in u.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = None
                # Check for /blob/branch/... or /resolve/branch/...
                if len(parts) >= 5 and parts[2] in ("blob", "resolve"):
                    filename = "/".join(parts[4:])
                return repo_id, filename

    # Plain repo or repo/file
    parts = clean.split("/")
    if len(parts) == 2:
        return clean, None
    elif len(parts) > 2 and parts[-1].endswith(".gguf"):
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
        return repo_id, filename

    return clean, None


def inspect_hf_repo(repo_id: str, token: str | None = None) -> HfRepoInfo:
    """Fetches repository information from Hugging Face."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    tok = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    info = HfRepoInfo(repo_id=repo_id)

    try:
        m_info = api.model_info(repo_id, files_metadata=True)
        info.is_valid = True
        info.pipeline_tag = getattr(m_info, "pipeline_tag", "") or ""

        files = []
        if getattr(m_info, "siblings", None):
            for s in m_info.siblings:
                rpath = s.rfilename
                files.append(rpath)
                if getattr(s, "size", None) is not None:
                    info.file_sizes[rpath] = s.size
        else:
            files = api.list_repo_files(repo_id, token=tok)

        info.total_files = len(files)
        info.gguf_files = sorted([f for f in files if f.lower().endswith(".gguf")])

        # Check for HF weights (config.json + safetensors/bin)
        has_config = "config.json" in files
        has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
        info.has_hf_weights = has_config and has_weights

    except RepositoryNotFoundError:
        info.is_valid = False
        info.error_message = f"リポジトリ '{repo_id}' が見つかりません（名前を確認してください）"
    except HfHubHTTPError as e:
        info.is_valid = False
        info.error_message = f"Hugging Face API エラー: {e}"
    except Exception as e:
        info.is_valid = False
        info.error_message = f"エラー: {e}"

    return info


class HfDownloadJob:
    """Downloads a HF repository or specific file with progress callbacks and cancellation."""

    def __init__(
        self,
        repo_id: str,
        local_dir: Path,
        filename: str | None = None,
        token: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.filename = filename
        self.token = token or os.environ.get("HF_TOKEN")
        self.allow_patterns = allow_patterns
        self.ignore_patterns = ignore_patterns
        self.on_log = on_log or (lambda msg: None)
        self.on_progress = on_progress or (lambda p, m: None)
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    def run(self) -> Path:
        """Executes the download. Returns the Path to the downloaded model folder or GGUF file."""
        from huggingface_hub import hf_hub_download, snapshot_download

        self.local_dir.mkdir(parents=True, exist_ok=True)
        job = self

        class TqdmCallback(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                job._update_progress(self)

            def update(self, n=1):
                super().update(n)
                job._update_progress(self)

            def close(self):
                super().close()
                job._update_progress(self)

        if self.filename:
            self.on_log(f"単体ファイルをダウンロード中: {self.repo_id} / {self.filename}")
            self.on_log(f"保存先: {self.local_dir}")
            self.on_progress(0.1, f"{self.filename} をダウンロード中...")

            dest = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=str(self.local_dir),
                token=self.token,
            )
            if self._canceled:
                raise InterruptedError("ダウンロードを取り消しました")
            self.on_progress(1.0, "ダウンロード完了")
            self.on_log(f"完了: {dest}")
            return Path(dest)

        else:
            self.on_log(f"モデルリポジトリをダウンロード中: {self.repo_id}")
            self.on_log(f"保存先: {self.local_dir}")
            self.on_progress(0.1, "ダウンロード準備中...")

            # Smart defaults: ignore other formats if not specified
            ignore = list(self.ignore_patterns or [])
            if not self.allow_patterns and not self.ignore_patterns:
                # Default HF model download: exclude unrelated heavy artifacts
                ignore = ["*.onnx", "*.ot", "*.msgpack", "*.h5"]

            dest = snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(self.local_dir),
                token=self.token,
                allow_patterns=self.allow_patterns,
                ignore_patterns=ignore,
                tqdm_class=TqdmCallback,
            )
            if self._canceled:
                raise InterruptedError("ダウンロードを取り消しました")
            self.on_progress(1.0, "ダウンロード完了")
            self.on_log(f"完了: {dest}")
            return Path(dest)

    def _update_progress(self, t: tqdm) -> None:
        if self._canceled:
            return
        desc = t.desc or "ダウンロード中"
        if t.total and t.total > 0:
            frac = min(1.0, max(0.0, float(t.n) / float(t.total)))
            self.on_progress(frac, f"{desc}: {t.n}/{t.total}")
        else:
            self.on_progress(0.5, desc)
