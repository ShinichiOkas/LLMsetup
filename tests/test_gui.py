from pathlib import Path

import pytest
from fixtures import write_gguf, write_hf_dir

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from llmquant.gui import COL_CHECK, COL_NAME, MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_gui_hf_flow(app, jp_dir: Path):
    d = write_hf_dir(jp_dir / "Tiny-Llama-長い名前のモデル")
    w = MainWindow()
    w.load_base(str(d))
    assert w.table.rowCount() == len(w.options) > 30
    sel = w.selected_types()
    assert len(sel) == 1 and next(o for o in w.options if o.recommended).name == sel[0]
    rec_row = next(r for r, o in enumerate(w.options) if o.recommended)
    assert "おすすめ" in w.table.item(rec_row, COL_NAME).text()
    assert w.out_edit.text() == str(jp_dir)
    assert "Tiny-Llama" in w.files_label.text()
    # disabled rows cannot be checked by clicking
    dis = next(r for r, o in enumerate(w.options) if not o.enabled)
    w._on_cell_clicked(dis, COL_NAME)
    assert w.options[dis].name not in w.selected_types()
    # clicking an enabled row toggles it
    en = next(r for r, o in enumerate(w.options) if o.enabled and not o.recommended)
    w._on_cell_clicked(en, COL_NAME)
    assert w.options[en].name in w.selected_types()
    w.close()


def test_gui_unsupported_shows_reason(app, jp_dir: Path):
    w = MainWindow()
    w.load_base(str(jp_dir / "does-not-exist"))
    assert w.table.rowCount() == 0
    assert "作れません" in w.msg_label.text()
    assert not w.run_btn.isEnabled()
    w.close()


def test_gui_existing_output_blocks_run(app, jp_dir: Path):
    g = write_gguf(jp_dir / "m-F16.gguf")
    w = MainWindow()
    w.ollama_chk.setChecked(False)
    w.load_base(str(g))
    w.set_selected(["Q4_K_M"])
    out = jp_dir / "m-Q4_K_M.gguf"
    out.write_bytes(b"x")
    w._update_plan()
    assert not w.run_btn.isEnabled() and "同名" in w.status.text()
    w.overwrite_chk.setChecked(True)
    # still blocked only if llama-quantize is missing; the duplicate-name error must be gone
    assert "同名" not in w.status.text()
    w.close()


def test_gui_ollama_names(app, jp_dir: Path):
    g = write_gguf(jp_dir / "m-F16.gguf")
    w = MainWindow()
    w.ollama_chk.setChecked(True)
    w.load_base(str(g))
    w.set_selected(["Q4_K_M", "Q8_0"])
    assert sorted(w.ollama_preview.text().splitlines()) == ["m:q4_k_m", "m:q8_0"]
    w.ollama_edit.setText("mymodel")
    w._on_ollama_edited("mymodel")
    assert sorted(w.ollama_preview.text().splitlines()) == ["mymodel:q4_k_m", "mymodel:q8_0"]
    w.close()


def test_checkbox_flags(app, jp_dir: Path):
    w = MainWindow()
    w.load_base(str(write_gguf(jp_dir / "m-F16.gguf")))
    for r, o in enumerate(w.options):
        flags = w.table.item(r, COL_CHECK).flags()
        assert bool(flags & Qt.ItemFlag.ItemIsEnabled) == o.enabled
    w.close()


def test_gui_ollama_tab(app, monkeypatch):
    from llmquant.tools import OllamaModelItem

    mock_models = [
        OllamaModelItem(name="mock-qwen:7b", id="abc123", size="4.5 GB", modified="yesterday"),
        OllamaModelItem(name="mock-llama:latest", id="def456", size="2.0 GB", modified="today"),
    ]
    monkeypatch.setattr("llmquant.tools.ollama_list_models", lambda max_age=0.0: mock_models)
    monkeypatch.setattr("llmquant.tools.ollama_model_names", lambda max_age=0.0: {"mock-qwen:7b", "mock-llama:latest"})
    monkeypatch.setattr("llmquant.tools.ollama_show_modelfile", lambda name: f'FROM blob123\nSYSTEM You are an assistant.\nPARAMETER temperature 0.8\n')

    w = MainWindow()
    assert w.tabs.count() == 3
    assert w.tabs.tabText(0) == "モデル量子化 (GGUF / MLX)"
    assert w.tabs.tabText(1) == "Ollama モデル作成"
    assert w.tabs.tabText(2) == "HF ダウンロード"

    # Switch to Ollama tab
    w.tabs.setCurrentIndex(1)
    assert w._om_initialized
    assert w.om_base_combo.count() == 2
    assert w.om_base_combo.currentText() == "mock-qwen:7b"
    assert "abc123" in w.om_info_label.text()
    assert "mock-qwen-custom:latest" in w.om_name_edit.text()

    # Check parameter guides show defaults and ranges
    assert "0.0 〜 2.0" in w.om_param_guides["temperature"].text()
    assert "デフォルト:" in w.om_param_guides["temperature"].text()
    assert "512 〜" in w.om_param_guides["num_ctx"].text()
    assert "0.0 〜 1.0" in w.om_param_guides["top_p"].text()
    assert "1 〜 100" in w.om_param_guides["top_k"].text()

    # Load base settings
    w._load_om_base_settings()
    assert "FROM mock-qwen:7b" in w.om_modelfile_edit.toPlainText()
    assert w.om_system_edit.toPlainText() == "You are an assistant."
    assert w.om_temp_edit.text() == "0.8"

    # Edit form and apply
    w.om_system_edit.setPlainText("Custom system prompt.")
    w.om_temp_edit.setText("0.5")
    w.om_ctx_combo.setEditText("8192")
    w._apply_form_to_modelfile()

    mf = w.om_modelfile_edit.toPlainText()
    assert 'SYSTEM """Custom system prompt."""' in mf
    assert "PARAMETER temperature 0.5" in mf
    assert "PARAMETER num_ctx 8192" in mf

    # Reset
    w._reset_modelfile()
    assert w.om_modelfile_edit.toPlainText().strip() == "FROM mock-qwen:7b"
    assert w.om_system_edit.toPlainText() == ""

    # Name validation
    w.om_name_edit.setText("INVALID NAME!")
    w._start_om_create()
    assert "使えない文字" in w.om_status.text()
    assert w.om_run_btn.isEnabled()

    w.close()


def test_gui_hf_tab(app, monkeypatch, tmp_path):
    from llmquant import hf_downloader as hfd

    mock_info = hfd.HfRepoInfo(
        repo_id="mock-org/mock-model",
        is_valid=True,
        has_hf_weights=True,
        total_files=5,
        gguf_files=["model-Q4_K_M.gguf", "model-Q8_0.gguf"],
        file_sizes={"model-Q4_K_M.gguf": 4000000},
    )
    monkeypatch.setattr("llmquant.hf_downloader.inspect_hf_repo", lambda repo, token=None: mock_info)

    w = MainWindow()
    w.tabs.setCurrentIndex(2)

    # Inspect repo
    w.hf_repo_edit.setText("mock-org/mock-model")
    w._inspect_hf_repo()

    assert "総ファイル数: 5" in w.hf_info_label.text()
    assert "GGUFファイル: 2 個" in w.hf_info_label.text()
    assert w.hf_file_combo.count() == 2

    # Toggle radio mode
    w.hf_mode_file_radio.setChecked(True)
    assert w.hf_file_combo.isEnabled()
    w.hf_mode_repo_radio.setChecked(True)
    assert not w.hf_file_combo.isEnabled()

    # Simulate download finished and set as base
    target_file = tmp_path / "mock.gguf"
    target_file.write_bytes(b"GGUF")
    w._on_hf_finished(target_file, None)
    assert w.hf_set_base_btn.isEnabled()
    assert "完了" in w.hf_status.text()

    # Click set base
    w._set_downloaded_as_base()
    assert w.tabs.currentIndex() == 0
    assert w.path_edit.text() == str(target_file)

    w.close()
