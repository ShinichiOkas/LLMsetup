"""PySide6 GUI: pick a base -> see what is possible -> quantize -> (Ollama)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import __version__, tools
from .basemodel import KIND_GGUF, KIND_HF, BaseInfo, human_bytes, human_params, inspect_base, ollama_name, output_filename
from .options import STATUS_DISABLED, STATUS_WARN, Environment, Option, build_options, fits_vram
from .pipeline import Cancelled, JobError, JobSpec, Runner, check_outputs, dry_run_size, plan

COL_CHECK, COL_NAME, COL_SIZE, COL_GPU, COL_QUALITY, COL_NOTE = range(6)


def _is_dark() -> bool:
    pal = QGuiApplication.palette()
    return pal.color(QPalette.ColorRole.Window).lightness() < 128


class _Signals(QObject):
    log = Signal(str)
    progress = Signal(float, str)
    finished = Signal(object, object)  # result, error
    sized = Signal(str, object, bool)  # qtype, bytes|None, needs_imatrix


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"LLMQuant {__version__} - LLM を GGUF に量子化して Ollama へ")
        self.base: BaseInfo | None = None
        self.options: list[Option] = []
        self.runner: Runner | None = None
        self.sig = _Signals()
        self.sig.log.connect(self._append_log)
        self.sig.progress.connect(self._on_progress)
        self.sig.finished.connect(self._on_finished)
        self.sig.sized.connect(self._on_sized)
        self._size_gen = 0
        self._out_dir_auto = True
        self._ollama_stem_auto = True
        self._build()
        self.setAcceptDrops(True)
        self._refresh_tools()

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        dark = _is_dark()
        self.c_red = QColor("#ff8a80" if dark else "#c62828")
        self.c_amber = QColor("#ffcc80" if dark else "#9a6700")
        self.c_grey = QColor("#8a8a8a")
        self.c_rec_bg = QColor("#1f3a24" if dark else "#e7f5e9")

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # 1. base
        g1 = QGroupBox("1. ベースモデル（HF 形式のフォルダ、または .gguf）")
        l1 = QVBoxLayout(g1)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("ここにフォルダ / .gguf をドロップ、またはパスを貼り付けて Enter")
        self.path_edit.returnPressed.connect(lambda: self.load_base(self.path_edit.text()))
        b_dir = QPushButton("フォルダを選ぶ…")
        b_dir.clicked.connect(self._pick_dir)
        b_file = QPushButton("GGUF を選ぶ…")
        b_file.clicked.connect(self._pick_file)
        row.addWidget(self.path_edit, 1)
        row.addWidget(b_dir)
        row.addWidget(b_file)
        l1.addLayout(row)
        self.info_grid = QGridLayout()
        self.info_grid.setHorizontalSpacing(18)
        self.info_labels: dict[str, QLabel] = {}
        keys = ["形式", "名前", "構造", "元の形式", "パラメータ", "重み", "チャットテンプレート", "GPU"]
        for i, k in enumerate(keys):
            kl = QLabel(k + ":")
            kl.setStyleSheet("color: palette(placeholder-text);")
            v = QLabel("-")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self.info_labels[k] = v
            r, c = divmod(i, 4)
            self.info_grid.addWidget(kl, r, c * 2)
            self.info_grid.addWidget(v, r, c * 2 + 1)
        for c in range(4):
            self.info_grid.setColumnStretch(c * 2 + 1, 1)
        l1.addLayout(self.info_grid)
        self.msg_label = QLabel()
        self.msg_label.setWordWrap(True)
        self.msg_label.setTextFormat(Qt.TextFormat.RichText)
        self.msg_label.hide()
        l1.addWidget(self.msg_label)
        root.addWidget(g1)

        # 2. options
        g2 = QGroupBox("2. 量子化の種類（作りたいものに印。複数可）")
        l2 = QVBoxLayout(g2)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["", "種類", "サイズ", "GPU", "品質", "説明・注意"])
        self.table.verticalHeader().hide()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_CHECK, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_GPU, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_QUALITY, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(COL_NOTE, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(COL_CHECK, 30)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        l2.addWidget(self.table, 1)
        opt_row = QHBoxLayout()
        self.show_disabled = QCheckBox("作れない種類も表示")
        self.show_disabled.setChecked(True)
        self.show_disabled.toggled.connect(self._apply_filter)
        opt_row.addWidget(self.show_disabled)
        opt_row.addSpacing(16)
        opt_row.addWidget(QLabel("imatrix:"))
        self.imatrix_edit = QLineEdit()
        self.imatrix_edit.setPlaceholderText("（任意）IQ1/IQ2/IQ3_XS 等に必要な重要度行列 .gguf / .dat")
        self.imatrix_edit.editingFinished.connect(self._rebuild_options)
        b_im = QPushButton("選ぶ…")
        b_im.clicked.connect(self._pick_imatrix)
        opt_row.addWidget(self.imatrix_edit, 1)
        opt_row.addWidget(b_im)
        l2.addLayout(opt_row)

        # 3. output + 4. ollama
        g3 = QGroupBox("3. 出力")
        f3 = QFormLayout(g3)
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.editingFinished.connect(self._on_out_dir_edited)
        b_out = QPushButton("選ぶ…")
        b_out.clicked.connect(self._pick_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(b_out)
        f3.addRow("出力フォルダ:", out_row)
        self.files_label = QLabel("-")
        self.files_label.setWordWrap(True)
        self.files_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f3.addRow("できるファイル:", self.files_label)
        chk_row = QHBoxLayout()
        self.overwrite_chk = QCheckBox("同名のファイルを上書きする")
        self.overwrite_chk.toggled.connect(self._update_plan)
        self.keep_inter_chk = QCheckBox("中間ファイル（F16/BF16）を残す")
        self.keep_inter_chk.toggled.connect(self._update_plan)
        chk_row.addWidget(self.overwrite_chk)
        chk_row.addWidget(self.keep_inter_chk)
        chk_row.addStretch(1)
        f3.addRow("", chk_row)

        g4 = QGroupBox("4. Ollama")
        f4 = QFormLayout(g4)
        self.ollama_chk = QCheckBox("できた GGUF を Ollama に登録する（ollama create）")
        self.ollama_chk.setChecked(tools.ollama_exe() is not None)
        self.ollama_chk.toggled.connect(self._update_plan)
        f4.addRow(self.ollama_chk)
        self.ollama_edit = QLineEdit()
        self.ollama_edit.textEdited.connect(self._on_ollama_edited)
        f4.addRow("モデル名:", self.ollama_edit)
        self.ollama_preview = QLabel("-")
        self.ollama_preview.setWordWrap(True)
        self.ollama_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f4.addRow("登録名:", self.ollama_preview)

        bottom = QHBoxLayout()
        bottom.addWidget(g3, 3)
        bottom.addWidget(g4, 2)

        # run
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("量子化を実行")
        self.run_btn.setMinimumHeight(34)
        f = self.run_btn.font()
        f.setBold(True)
        self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("取り消し")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.open_btn = QPushButton("出力フォルダを開く")
        self.open_btn.clicked.connect(self._open_out)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addWidget(self.progress, 1)
        run_row.addWidget(self.open_btn)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.log.setFont(mono)
        self.log.setPlaceholderText("実行ログ")

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(g2, 1)
        tl.addLayout(bottom)
        tl.addLayout(run_row)
        tl.addWidget(self.status)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.log)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        self.setCentralWidget(central)

        vram = tools.gpu_vram_bytes()
        self.info_labels["GPU"].setText(f"VRAM {human_bytes(vram)}" if vram else "（検出なし）")

    # ------------------------------------------------------------ tools
    def _refresh_tools(self) -> None:
        st = tools.tool_status()
        if st.missing:
            self._set_status("<br>".join(st.missing), self.c_red)

    # ------------------------------------------------------------ pickers / dnd
    def _start_dir(self) -> str:
        t = self.path_edit.text().strip()
        if t:
            p = Path(t)
            return str(p.parent if p.is_file() else p)
        return str(Path.home())

    def _pick_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "HF 形式のモデルフォルダ", self._start_dir())
        if d:
            self.load_base(d)

    def _pick_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "GGUF ファイル", self._start_dir(),
                                           "GGUF / safetensors (*.gguf *.safetensors);;すべて (*)")
        if f:
            self.load_base(f)

    def _pick_imatrix(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "imatrix ファイル", self._start_dir(),
                                           "imatrix (*.gguf *.dat *.imatrix);;すべて (*)")
        if f:
            self.imatrix_edit.setText(f)
            self._rebuild_options()

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "出力フォルダ", self.out_edit.text() or self._start_dir())
        if d:
            self.out_edit.setText(d)
            self._on_out_dir_edited()

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        urls = e.mimeData().urls()
        if urls:
            self.load_base(urls[0].toLocalFile())

    # ------------------------------------------------------------ base
    def load_base(self, path: str) -> None:
        path = path.strip().strip('"')
        if not path:
            return
        self.path_edit.setText(path)
        base = inspect_base(path)
        self.base = base
        L = self.info_labels
        kind = {KIND_HF: "HF 形式（変換 → 量子化）", KIND_GGUF: "GGUF（量子化のみ）"}.get(base.kind, "対象外")
        L["形式"].setText(kind)
        L["名前"].setText(base.name or "-")
        arch = base.architecture or "-"
        if base.kind == KIND_HF and base.arch_supported:
            arch += "（変換可）"
        if base.is_moe:
            arch += " / MoE"
        L["構造"].setText(arch)
        src = base.source_type or "-"
        if base.is_quantized:
            src += "（量子化済み）"
        L["元の形式"].setText(src)
        L["パラメータ"].setText(human_params(base.n_params) if base.n_params else "-")
        L["重み"].setText(human_bytes(base.weight_bytes) if base.weight_bytes else "-")
        L["チャットテンプレート"].setText({True: "あり", False: "なし", None: "-"}[base.has_chat_template])
        for k in ("名前", "構造"):
            L[k].setToolTip(L[k].text())

        msgs = []
        red, amber = self.c_red.name(), self.c_amber.name()
        for p in base.problems:
            msgs.append(f'<span style="color:{red}">作れません: {_esc(p)}</span>')
        for w in base.warnings:
            msgs.append(f'<span style="color:{amber}">注意: {_esc(w)}</span>')
        self.msg_label.setText("<br>".join(msgs))
        self.msg_label.setVisible(bool(msgs))

        if self._out_dir_auto or not self.out_edit.text().strip():
            p = Path(path)
            self.out_edit.setText(str(p.parent if p.is_file() else p.parent))
            self._out_dir_auto = True
        if self._ollama_stem_auto or not self.ollama_edit.text().strip():
            self.ollama_edit.setText(ollama_name(base.stem, "x").split(":")[0])
            self._ollama_stem_auto = True
        self._rebuild_options()

    def _imatrix(self) -> Path | None:
        t = self.imatrix_edit.text().strip().strip('"')
        return Path(t) if t else None

    def _rebuild_options(self) -> None:
        if self.base is None:
            return
        checked = set(self.selected_types())
        env = Environment.detect(Path(self.out_edit.text().strip() or "."))
        self.options = build_options(self.base, env, self._imatrix())
        self._fill_table(checked)
        self._start_exact_sizes()

    def _fill_table(self, previously_checked: set[str]) -> None:
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self.options))
        any_checked = False
        for r, o in enumerate(self.options):
            chk = QTableWidgetItem()
            flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            if not o.enabled:
                flags = Qt.ItemFlag.ItemIsUserCheckable
            chk.setFlags(flags)
            on = o.enabled and (o.name in previously_checked or (not previously_checked and o.recommended))
            any_checked |= on
            chk.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            chk.setData(Qt.ItemDataRole.UserRole, o.name)
            t.setItem(r, COL_CHECK, chk)

            name = QTableWidgetItem(o.name + ("  おすすめ" if o.recommended else ""))
            fnt = name.font()
            fnt.setBold(o.recommended)
            name.setFont(fnt)
            t.setItem(r, COL_NAME, name)
            t.setItem(r, COL_SIZE, QTableWidgetItem(self._size_text(o)))
            gpu = "-" if o.fits_vram is None else ("載る" if o.fits_vram else "載らない")
            t.setItem(r, COL_GPU, QTableWidgetItem(gpu))
            t.setItem(r, COL_QUALITY, QTableWidgetItem("★" * o.qtype.quality + "☆" * (6 - o.qtype.quality)))
            note_parts = [o.qtype.note]
            if o.status == STATUS_DISABLED:
                note_parts = ["作れません: " + " / ".join(o.reasons)]
            elif o.notes:
                note_parts.append("注意: " + " / ".join(o.notes))
            note = QTableWidgetItem("  ".join(note_parts))
            t.setItem(r, COL_NOTE, note)
            tip = "\n".join([o.qtype.note, *o.reasons, *o.notes])
            row_items = [t.item(r, c) for c in range(6)]
            for c, it in enumerate(row_items):
                assert it is not None
                it.setToolTip(tip)
                if c != COL_CHECK:
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if o.recommended:
                    it.setBackground(QBrush(self.c_rec_bg))
                if o.status == STATUS_DISABLED and c != COL_CHECK:
                    it.setForeground(QBrush(self.c_grey))
            if o.status == STATUS_WARN:
                note.setForeground(QBrush(self.c_amber))
            if o.fits_vram is False and o.enabled:
                gpu_item = row_items[COL_GPU]
                assert gpu_item is not None
                gpu_item.setForeground(QBrush(self.c_amber))
        t.blockSignals(False)
        self._apply_filter()
        self._update_plan()
        rec = next((r for r, o in enumerate(self.options) if o.recommended), None)
        if rec is not None:
            t.scrollToItem(t.item(rec, COL_NAME), QAbstractItemView.ScrollHint.PositionAtCenter)

    def _size_text(self, o: Option) -> str:
        if o.exact_bytes is not None:
            return human_bytes(o.exact_bytes)
        return "約 " + human_bytes(o.est_bytes)

    def _apply_filter(self) -> None:
        show = self.show_disabled.isChecked()
        for r, o in enumerate(self.options):
            self.table.setRowHidden(r, (not show) and not o.enabled)

    # exact sizes via llama-quantize --dry-run (GGUF bases only)
    def _start_exact_sizes(self) -> None:
        self._size_gen += 1
        gen = self._size_gen
        base = self.base
        if base is None or base.kind != KIND_GGUF or not base.ok or tools.quantize_exe() is None:
            return
        src = base.gguf_parts[0] if base.gguf_parts else base.path
        names = [o.name for o in self.options if o.enabled]
        imatrix = self._imatrix()

        def work() -> None:
            for n in names:
                if gen != self._size_gen:
                    return
                size, need_im = dry_run_size(src, n, imatrix)
                if gen != self._size_gen:
                    return
                self.sig.sized.emit(n, size, need_im)

        threading.Thread(target=work, daemon=True).start()

    def _on_sized(self, qtype: str, size, need_imatrix: bool) -> None:
        for r, o in enumerate(self.options):
            if o.name != qtype:
                continue
            if size is not None:
                o.exact_bytes = int(size)
                self.table.item(r, COL_SIZE).setText(self._size_text(o))
                o.fits_vram = fits_vram(o.exact_bytes, tools.gpu_vram_bytes())
                if o.fits_vram is not None:
                    self.table.item(r, COL_GPU).setText("載る" if o.fits_vram else "載らない")
            if need_imatrix and self._imatrix() is None and o.enabled:
                o.status = STATUS_DISABLED
                o.reasons.append("imatrix ファイルが必要です（下の欄で指定）")
                self._fill_table(set(self.selected_types()) - {qtype})
            break

    # ------------------------------------------------------------ selection / plan
    def selected_types(self) -> list[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, COL_CHECK)
            if it and it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def set_selected(self, names: list[str]) -> None:
        want = {n.upper() for n in names}
        self.table.blockSignals(True)
        for r, o in enumerate(self.options):
            it = self.table.item(r, COL_CHECK)
            it.setCheckState(Qt.CheckState.Checked if (o.name in want and o.enabled) else Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
        self._update_plan()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == COL_CHECK:
            self._update_plan()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col == COL_CHECK or row >= len(self.options) or not self.options[row].enabled:
            return
        it = self.table.item(row, COL_CHECK)
        it.setCheckState(Qt.CheckState.Unchecked if it.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)

    def _on_out_dir_edited(self) -> None:
        self._out_dir_auto = False
        self._rebuild_options()

    def _on_ollama_edited(self, _text: str) -> None:
        self._ollama_stem_auto = False
        self._update_plan()

    def _spec(self) -> JobSpec | None:
        if self.base is None:
            return None
        sel = self.selected_types()
        stem = self.ollama_edit.text().strip().lower() or ollama_name(self.base.stem, "x").split(":")[0]
        return JobSpec(
            base=self.base, targets=sel, out_dir=Path(self.out_edit.text().strip() or "."),
            imatrix=self._imatrix(), keep_intermediate=self.keep_inter_chk.isChecked(),
            overwrite=self.overwrite_chk.isChecked(), register_ollama=self.ollama_chk.isChecked(),
            ollama_names={t: f"{stem}:{t.lower()}" for t in sel},
        )

    def _update_plan(self) -> None:
        spec = self._spec()
        running = self.runner is not None
        if spec is None:
            self.run_btn.setEnabled(False)
            return
        sel = spec.targets
        self.files_label.setText("\n".join(output_filename(spec.base.stem, t) for t in sel) or "-")
        self.ollama_edit.setEnabled(spec.register_ollama)
        self.ollama_preview.setText(
            ("\n".join(spec.ollama_name_for(t) for t in sel) or "-") if spec.register_ollama else "（登録しない）")
        errs = check_outputs(spec)
        self.run_btn.setEnabled(not errs and not running)
        if running:
            return
        if errs:
            self._set_status("<br>".join(_esc(e) for e in errs), self.c_red if spec.base.ok and sel else self.c_grey)
        else:
            total = sum(o.size_bytes for o in self.options if o.name in sel)
            steps = plan(spec)
            self._set_status(f"準備よし: {len(sel)} 個の GGUF（合計 約 {human_bytes(total)}）を {len(steps)} 手順で作ります",
                             None)

    def _set_status(self, html: str, color: QColor | None) -> None:
        if color is not None:
            html = f'<span style="color:{color.name()}">{html}</span>'
        self.status.setText(html)

    # ------------------------------------------------------------ run
    def start(self) -> None:
        spec = self._spec()
        if spec is None or self.runner is not None:
            return
        errs = check_outputs(spec)
        if errs:
            self._set_status("<br>".join(_esc(e) for e in errs), self.c_red)
            return
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        steps = plan(spec)
        self.log.clear()
        self.runner = Runner(steps, on_log=self.sig.log.emit, on_progress=lambda f, i, t: self.sig.progress.emit(f, t))
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self._set_status("実行中…", None)
        runner = self.runner

        def work() -> None:
            try:
                res = runner.run()
                self.sig.finished.emit(res, None)
            except BaseException as e:  # noqa: BLE001 - reported to the UI
                self.sig.finished.emit(None, e)

        threading.Thread(target=work, daemon=True).start()

    def cancel(self) -> None:
        if self.runner:
            self.runner.cancel()
            self._set_status("取り消し中…", self.c_amber)

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_progress(self, frac: float, title: str) -> None:
        self.progress.setValue(int(frac * 1000))
        if self.runner is not None:
            self._set_status(f"実行中 {frac:.0%}: {_esc(title)}", None)

    def _on_finished(self, res, err) -> None:
        self.runner = None
        self.cancel_btn.setEnabled(False)
        if err is None:
            self.progress.setValue(1000)
            lines = [f"完了（{res.seconds:.0f} 秒）: " + ", ".join(p.name for p in res.outputs.values())]
            if res.ollama_models:
                lines.append("Ollama: " + " / ".join(f"ollama run {n}" for n in res.ollama_models.values()))
            msg = "<br>".join(_esc(x) for x in lines)
            if res.warnings:
                msg += "<br>" + "<br>".join(
                    f'<span style="color:{self.c_amber.name()}">注意: {_esc(w)}</span>' for w in res.warnings)
            self._rebuild_options()
            self._set_status(msg, None)
        else:
            self.progress.setValue(0)
            text = "取り消しました" if isinstance(err, Cancelled) else f"失敗: {err}"
            if not isinstance(err, (Cancelled, JobError)):
                text = f"失敗（予期しないエラー）: {type(err).__name__}: {err}"
            self._append_log(text)
            self._update_plan()
            self._set_status(_esc(text), self.c_amber if isinstance(err, Cancelled) else self.c_red)
        self.run_btn.setEnabled(True)

    def _open_out(self) -> None:
        d = self.out_edit.text().strip()
        if d and Path(d).is_dir() and os.name == "nt":
            os.startfile(d)  # noqa: S606

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.runner is not None:
            ans = QMessageBox.question(self, "実行中", "量子化を実行中です。取り消して閉じますか？")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.runner.cancel()
        e.accept()


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def run_gui(path: str | None = None, screenshot: str | None = None, width: int = 1280, height: int = 860,
            select: list[str] | None = None, imatrix: str | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("LLMQuant")
    w = MainWindow()
    w.resize(width, height)
    if imatrix:
        w.imatrix_edit.setText(imatrix)
    if path:
        w.load_base(path)
    if select:
        w.set_selected(select)
    if screenshot:
        # Render without putting a window on the user's screen.
        w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        w.show()
        ok = [False]

        def shoot() -> None:
            ok[0] = w.grab().save(screenshot)
            w.close()
            app.quit()

        QTimer.singleShot(400, shoot)
        app.exec()
        rc = 0 if ok[0] else 1
    else:
        w.show()
        rc = app.exec()
    # Destroy the window before QApplication goes away; letting the
    # interpreter tear both down in arbitrary order crashes (0xC0000005).
    w._size_gen += 1  # stop background dry-run threads from emitting
    w.deleteLater()
    QApplication.processEvents()
    del w
    return rc
