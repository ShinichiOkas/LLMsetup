"""Tests for hf_downloader module."""

import pytest
from llmquant import hf_downloader as hfd


def test_parse_hf_input():
    # Simple repo ID
    repo, fn = hfd.parse_hf_input("Qwen/Qwen2.5-Coder-7B")
    assert repo == "Qwen/Qwen2.5-Coder-7B"
    assert fn is None

    # URL to model
    repo, fn = hfd.parse_hf_input("https://huggingface.co/Qwen/Qwen2.5-Coder-7B")
    assert repo == "Qwen/Qwen2.5-Coder-7B"
    assert fn is None

    # URL with blob (single file)
    repo, fn = hfd.parse_hf_input(
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf"
    )
    assert repo == "bartowski/Llama-3.2-1B-Instruct-GGUF"
    assert fn == "Llama-3.2-1B-Instruct-Q8_0.gguf"

    # URL with resolve (single file)
    repo, fn = hfd.parse_hf_input(
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/subfolder/model.gguf"
    )
    assert repo == "bartowski/Llama-3.2-1B-Instruct-GGUF"
    assert fn == "subfolder/model.gguf"

    # Empty
    repo, fn = hfd.parse_hf_input("")
    assert repo == ""
    assert fn is None


def test_inspect_hf_repo_mock(monkeypatch):
    class MockSibling:
        def __init__(self, rfilename, size=100):
            self.rfilename = rfilename
            self.size = size

    class MockModelInfo:
        id = "test/mock-model"
        pipeline_tag = "text-generation"
        siblings = [
            MockSibling("config.json", 500),
            MockSibling("model.safetensors", 10000),
            MockSibling("tokenizer.json", 200),
            MockSibling("model-Q4_K_M.gguf", 5000),
        ]

    class MockHfApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, files_metadata=True):
            return MockModelInfo()

    monkeypatch.setattr("huggingface_hub.HfApi", MockHfApi)

    info = hfd.inspect_hf_repo("test/mock-model")
    assert info.is_valid
    assert info.pipeline_tag == "text-generation"
    assert info.has_hf_weights
    assert "model-Q4_K_M.gguf" in info.gguf_files
    assert info.total_files == 4
    assert info.file_sizes["config.json"] == 500


def test_download_progress_tracker(tmp_path):
    log_file = tmp_path / "logs" / "test_download.log"
    logs_received = []
    progress_updates = []

    tracker = hfd.DownloadProgressTracker(
        total_expected_bytes=1000,
        on_progress=lambda f, t: progress_updates.append((f, t)),
        on_log=lambda t: logs_received.append(t),
        log_file=log_file,
    )

    tracker.log("初期化完了")
    assert len(logs_received) == 1
    assert "初期化完了" in logs_received[0]
    assert log_file.exists()
    assert "初期化完了" in log_file.read_text(encoding="utf-8")

    tracker.on_file_start("model.safetensors", 1000)
    assert len(logs_received) == 2
    assert "ダウンロード開始: model.safetensors" in logs_received[1]

    # バイト進捗
    tracker.on_bytes(500)
    assert tracker.downloaded_bytes == 500
    assert tracker.current_file_bytes == 500

    # キャンセル時の例外送出
    tracker.canceled = True
    with pytest.raises(InterruptedError):
        tracker.on_bytes(100)


def test_human_speed():
    assert hfd.human_speed(0) == "0 B/s"
    assert hfd.human_speed(1024) == "1 KB/s"
    assert hfd.human_speed(1024 * 1024 * 15.5) == "15.50 MB/s"

