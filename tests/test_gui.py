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
