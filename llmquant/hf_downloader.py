"""Hugging Face model downloader for LLMQuant with byte-level progress and persistent logging."""

from __future__ import annotations

import errno
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from . import tools
from .basemodel import human_bytes


def human_speed(bytes_per_sec: float) -> str:
    """Formats bytes per second to human readable string (e.g. '12.50 MB/s')."""
    return f"{human_bytes(int(bytes_per_sec))}/s"


@dataclass
class HfRepoInfo:
    repo_id: str
    is_valid: bool = False
    error_message: str = ""
    gguf_files: list[str] = field(default_factory=list)
    has_hf_weights: bool = False
    is_mlx: bool = False
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str = ""
    total_files: int = 0
    file_sizes: dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0


def parse_hf_input(text: str) -> tuple[str, str | None]:
    """Parses user input (Repo ID or Hugging Face URL) into (repo_id, filename_or_none)."""
    clean = text.strip()
    if not clean:
        return "", None

    if clean.startswith("http://") or clean.startswith("https://"):
        u = urlparse(clean)
        if "huggingface.co" in u.netloc:
            parts = [p for p in u.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = None
                if len(parts) >= 5 and parts[2] in ("blob", "resolve"):
                    filename = "/".join(parts[4:])
                return repo_id, filename

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
        info.tags = list(getattr(m_info, "tags", None) or [])

        files = []
        if getattr(m_info, "siblings", None):
            for s in m_info.siblings:
                rpath = s.rfilename
                files.append(rpath)
                if getattr(s, "size", None) is not None:
                    info.file_sizes[rpath] = s.size
                    info.total_size_bytes += s.size
        else:
            files = api.list_repo_files(repo_id, token=tok)

        info.total_files = len(files)
        info.gguf_files = sorted([f for f in files if f.lower().endswith(".gguf")])

        has_config = "config.json" in files
        has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
        info.has_hf_weights = has_config and has_weights
        info.is_mlx = info.has_hf_weights and (
            "mlx" in [t.lower() for t in info.tags]
            or repo_id.lower().startswith("mlx-community/")
            or "mlx" in repo_id.lower().split("/")[-1].split("-")
        )

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


class DownloadProgressTracker:
    """Tracks byte-level progress, download speed, and writes persistent action logs."""

    def __init__(
        self,
        total_expected_bytes: int = 0,
        resumed_bytes: int = 0,
        on_progress: Callable[[float, str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        log_file: Path | None = None,
    ) -> None:
        self.total_expected_bytes = total_expected_bytes
        self.resumed_bytes = resumed_bytes
        self.session_bytes = 0
        self.downloaded_bytes = resumed_bytes
        self.current_file_name = ""
        self.current_file_bytes = 0
        self.current_file_total = 0
        self.start_time = time.monotonic()
        self.last_emit_time = 0.0
        self.on_progress = on_progress or (lambda f, t: None)
        self.on_log = on_log or (lambda t: None)
        self.log_file = log_file
        self.canceled = False
        self._lock = threading.Lock()

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def reset_speed_clock(self) -> None:
        with self._lock:
            self.start_time = time.monotonic()
            self.session_bytes = 0

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.on_log(line)
        if self.log_file:
            try:
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def on_file_start(self, filename: str, total_bytes: int, initial_bytes: int = 0) -> None:
        with self._lock:
            self.current_file_name = filename
            self.current_file_total = total_bytes
            self.current_file_bytes = initial_bytes
        sz_str = human_bytes(total_bytes) if total_bytes > 0 else "不明"
        if initial_bytes > 0:
            self.log(f"ダウンロード再開: {filename} ({human_bytes(initial_bytes)} / {sz_str})")
        else:
            self.log(f"ダウンロード開始: {filename} ({sz_str})")

    def on_bytes(self, n: int) -> None:
        with self._lock:
            if self.canceled:
                raise InterruptedError("ダウンロードが取り消されました")
            if n <= 0:
                return
            self.current_file_bytes += n
            self.session_bytes += n
            self.downloaded_bytes += n

            now = time.monotonic()
            if now - self.last_emit_time >= 0.2:
                self.last_emit_time = now
                self._emit_progress(now)

    def on_file_done(self) -> None:
        with self._lock:
            fn = self.current_file_name
            sz = self.current_file_bytes
        self.log(f"ダウンロード完了: {fn} ({human_bytes(sz)})")

    def _emit_progress(self, now: float) -> None:
        elapsed = max(now - self.start_time, 0.001)
        speed = self.session_bytes / elapsed
        speed_str = human_speed(speed)

        if self.total_expected_bytes > 0:
            effective_total = max(self.total_expected_bytes, self.downloaded_bytes)
            frac = min(1.0, max(0.0, self.downloaded_bytes / effective_total))
            rem_bytes = max(0, effective_total - self.downloaded_bytes)
            eta_sec = rem_bytes / speed if speed > 0 else 0
            eta_str = f"残り約 {int(eta_sec // 60)}分{int(eta_sec % 60):02d}秒" if eta_sec > 60 else f"残り約 {int(eta_sec)}秒"

            cur_str = f"{human_bytes(self.downloaded_bytes)} / {human_bytes(effective_total)}"
            text = f"{self.current_file_name}: {cur_str} ({frac:.1%})  |  {speed_str}  |  {eta_str}"
        elif self.current_file_total > 0:
            frac = min(1.0, max(0.0, self.current_file_bytes / self.current_file_total))
            cur_str = f"{human_bytes(self.current_file_bytes)} / {human_bytes(self.current_file_total)}"
            text = f"{self.current_file_name}: {cur_str} ({frac:.1%})  |  {speed_str}"
        else:
            frac = 0.5
            text = f"{self.current_file_name}: {human_bytes(self.downloaded_bytes)}  |  {speed_str}"

        self.on_progress(frac, text)


def _is_xet_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(k in msg for k in ("xet", "internal writer error", "background writer", "file reconstruction error", "channel closed"))


def _is_fatal_error(exc: Exception) -> bool:
    if isinstance(exc, InterruptedError):
        return True
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    name = type(exc).__name__
    if name in ("RepositoryNotFoundError", "GatedRepoError", "EntryNotFoundError"):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403, 404):
        return True
    return False


def _estimate_existing_bytes(local_dir: Path, filename: str | None = None) -> int:
    """Sums sizes of already downloaded files and .incomplete cache files in local_dir."""
    if not local_dir.exists():
        return 0
    total = 0
    if filename:
        target = local_dir / filename
        if target.is_file():
            return target.stat().st_size
    else:
        for p in local_dir.rglob("*"):
            if not p.is_file():
                continue
            if ".cache" in p.parts:
                if p.name.endswith(".incomplete"):
                    total += p.stat().st_size
            else:
                total += p.stat().st_size
    if filename:
        dl_cache = local_dir / ".cache" / "huggingface" / "download"
        if dl_cache.is_dir():
            for p in dl_cache.rglob("*.incomplete"):
                total += p.stat().st_size
    return total


def has_incomplete_cache(local_dir: Path) -> bool:
    dl_cache = Path(local_dir) / ".cache" / "huggingface" / "download"
    return dl_cache.is_dir() and any(dl_cache.rglob("*.incomplete"))


class HfDownloadJob:
    """Downloads a HF repository or specific file with byte-level tracking, retry/XET fallback, cancellation, and logs."""

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
        log_file: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.filename = filename
        self.token = token or os.environ.get("HF_TOKEN")
        self.allow_patterns = allow_patterns
        self.ignore_patterns = ignore_patterns
        self.on_log = on_log or (lambda msg: None)
        self.on_progress = on_progress or (lambda p, m: None)
        self.log_file = log_file or (Path("logs") / "hf_download.log")
        self.max_retries = max_retries
        self.tracker: DownloadProgressTracker | None = None
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True
        if self.tracker:
            self.tracker.canceled = True

    def _sleep_cancelable(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._canceled:
                raise InterruptedError("ダウンロードを取り消しました")
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))

    def run(self) -> Path:
        """Executes the download with automatic retry and XET->HTTP fallback."""
        import huggingface_hub.constants as hf_const
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import tqdm as hf_tqdm

        self.local_dir.mkdir(parents=True, exist_ok=True)
        info = inspect_hf_repo(self.repo_id, token=self.token)

        # Estimate expected total bytes
        if self.filename:
            total_bytes = info.file_sizes.get(self.filename, 0)
        else:
            total_bytes = info.total_size_bytes

        existing_bytes = _estimate_existing_bytes(self.local_dir, self.filename)
        if total_bytes > 0:
            resumed_bytes = min(existing_bytes, total_bytes)
            remaining_bytes = max(0, total_bytes - resumed_bytes)
            free = tools.free_bytes(self.local_dir)
            if free is not None and remaining_bytes > free:
                raise RuntimeError(
                    f"保存先の空き容量が足りません（残り必要 約 {human_bytes(remaining_bytes)} / 空き {human_bytes(free)}）"
                )
        else:
            resumed_bytes = existing_bytes

        tracker = DownloadProgressTracker(
            total_expected_bytes=total_bytes,
            resumed_bytes=resumed_bytes,
            on_progress=self.on_progress,
            on_log=self.on_log,
            log_file=self.log_file,
        )
        self.tracker = tracker

        tracker.log("==================================================")
        tracker.log(f"ダウンロードジョブ開始: リポジトリ '{self.repo_id}'")
        if self.filename:
            tracker.log(f"対象ファイル: {self.filename}")
        tracker.log(f"保存先: {self.local_dir.resolve()}")
        if total_bytes > 0:
            tracker.log(f"予定ダウンロード総量: {human_bytes(total_bytes)}")
        if resumed_bytes > 0:
            tracker.log(f"既存ファイル / 途中キャッシュを検出: {human_bytes(resumed_bytes)}（続きから再開します）")
        tracker.log("--------------------------------------------------")

        class GuiTqdm(hf_tqdm):
            def __init__(self, *args, **kwargs):
                self._bar_name = kwargs.get("name") or ""
                super().__init__(*args, **kwargs)
                desc = str(getattr(self, "desc", "") or "")
                unit = str(getattr(self, "unit", "") or "")
                self._is_byte_transfer = (
                    unit == "B" and "reconstruct" not in desc.lower()
                )
                if self._is_byte_transfer:
                    tracker.on_file_start(desc or (self._bar_name or "download"), int(self.total or 0), int(self.n or 0))

            def update(self, n=1):
                super().update(n)
                if self._is_byte_transfer:
                    tracker.on_bytes(int(n))
                elif tracker.canceled:
                    raise InterruptedError("ダウンロードが取り消されました")

            def close(self):
                if self._is_byte_transfer:
                    tracker.on_file_done()
                super().close()

        try:
            attempts = self.max_retries + 1
            for attempt in range(1, attempts + 1):
                if self._canceled:
                    raise InterruptedError("ダウンロードを取り消しました")
                try:
                    if self.filename:
                        dest = hf_hub_download(
                            repo_id=self.repo_id,
                            filename=self.filename,
                            local_dir=str(self.local_dir),
                            token=self.token,
                            tqdm_class=GuiTqdm,
                        )
                        res_path = Path(dest)
                    else:
                        ignore = list(self.ignore_patterns or [])
                        if not self.allow_patterns and not self.ignore_patterns:
                            ignore = ["*.onnx", "*.ot", "*.msgpack", "*.h5"]

                        dest = snapshot_download(
                            repo_id=self.repo_id,
                            local_dir=str(self.local_dir),
                            token=self.token,
                            allow_patterns=self.allow_patterns,
                            ignore_patterns=ignore,
                            tqdm_class=GuiTqdm,
                        )
                        res_path = Path(dest)

                    if self._canceled:
                        tracker.log("[終了] ユーザーによって取り消されました")
                        raise InterruptedError("ダウンロードを取り消しました")

                    self.on_progress(1.0, f"ダウンロード完了 ({human_bytes(tracker.downloaded_bytes)})")
                    tracker.log(f"[終了] 正常完了: {res_path}")
                    return res_path

                except InterruptedError:
                    raise
                except Exception as e:
                    if _is_fatal_error(e) or attempt >= attempts:
                        tracker.log(f"[エラー終了] 失敗: {type(e).__name__}: {e}")
                        raise
                    if _is_xet_error(e) and not hf_const.HF_HUB_DISABLE_XET:
                        os.environ["HF_HUB_DISABLE_XET"] = "1"
                        hf_const.HF_HUB_DISABLE_XET = True
                        tracker.log(
                            f"[警告] hf-xet エラーを検出したため通常 HTTP ダウンロードに切り替えて再開します ({attempt}/{self.max_retries}): {e}"
                        )
                    else:
                        wait_s = min(10, 2 * attempt)
                        tracker.log(
                            f"[警告] 一時エラーのため {wait_s} 秒後に続きから自動再試行します ({attempt}/{self.max_retries}): {type(e).__name__}: {e}"
                        )
                        self._sleep_cancelable(wait_s)
                    # Recalculate resumed bytes from disk before retrying
                    new_resumed = _estimate_existing_bytes(self.local_dir, self.filename)
                    tracker.resumed_bytes = min(new_resumed, total_bytes) if total_bytes > 0 else new_resumed
                    tracker.downloaded_bytes = tracker.resumed_bytes
                    tracker.reset_speed_clock()

        except InterruptedError:
            tracker.log("[終了] ダウンロードを取り消しました")
            raise
        finally:
            tracker.log("==================================================\n")
