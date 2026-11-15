"""PySide6 GUI: pick a base -> see what is possible -> quantize -> (Ollama)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import __version__, catalog, tools
from . import chat_engine as ce
from . import hf_downloader as hfd
from . import ollama_modelfile as om
from .basemodel import (
    KIND_GGUF,
    KIND_HF,
    KIND_MLX,
    BaseInfo,
    human_bytes,
    human_params,
    inspect_base,
    ollama_name,
    output_filename,
)
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


class _OllamaSignals(QObject):
    log = Signal(str)
    progress = Signal(float, str)
    finished = Signal(bool)


class _HfSignals(QObject):
    log = Signal(str)
    progress = Signal(float, str)
    finished = Signal(object, object)  # dest_path, error


class _ChatSignals(QObject):
    chunk = Signal(str)
    status = Signal(str)
    finished = Signal(object, object)  # ChatMetrics | None, error


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"LLMQuant {__version__} - LLM 量子化 (GGUF / MLX)")
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

        # Ollama Modelfile tab state
        self.om_models: list[tools.OllamaModelItem] = []
        self.om_details: tools.OllamaModelDetails | None = None
        self.om_param_guides: dict[str, QLabel] = {}
        self.om_job: om.OllamaCreateJob | None = None
        self._om_name_auto = True
        self._om_initialized = False
        self.om_sig = _OllamaSignals()
        self.om_sig.log.connect(self._append_om_log)
        self.om_sig.progress.connect(self._on_om_progress)
        self.om_sig.finished.connect(self._on_om_finished)

        # Hugging Face downloader state
        self.hf_job: hfd.HfDownloadJob | None = None
        self.hf_downloaded_path: Path | None = None
        self.hf_info: hfd.HfRepoInfo | None = None
        self.hf_sig = _HfSignals()
        self.hf_sig.log.connect(self._append_hf_log)
        self.hf_sig.progress.connect(self._on_hf_progress)
        self.hf_sig.finished.connect(self._on_hf_finished)

        # Chat tab state
        self.chat_session = ce.MlxChatSession()
        self.chat_history: list[dict[str, str]] = []
        self.chat_cancel: threading.Event | None = None
        self.chat_running = False
        self._chat_initialized = False
        self._chat_reply_buf: list[str] = []
        self.chat_sig = _ChatSignals()
        self.chat_sig.chunk.connect(self._on_chat_chunk)
        self.chat_sig.status.connect(self._on_chat_status)
        self.chat_sig.finished.connect(self._on_chat_finished)

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
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        quant_widget = self._build_quant_tab()
        self.tabs.addTab(quant_widget, "モデル量子化 (GGUF / MLX)")

        ollama_widget = self._build_ollama_tab()
        self.tabs.addTab(ollama_widget, "Ollama モデル作成")

        hf_widget = self._build_hf_tab()
        self.tabs.addTab(hf_widget, "HF ダウンロード")

        chat_widget = self._build_chat_tab()
        self.tabs.addTab(chat_widget, "チャット")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(central)

        vram = tools.gpu_vram_bytes()
        if tools.is_apple_silicon() and vram:
            self.info_labels["GPU"].setText(f"Unified {human_bytes(vram)}")
        else:
            self.info_labels["GPU"].setText(f"VRAM {human_bytes(vram)}" if vram else "（検出なし）")

    def _build_quant_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 0. target format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("出力形式:"))
        self.fmt_gguf_radio = QRadioButton("GGUF (llama.cpp / Ollama)")
        self.fmt_mlx_radio = QRadioButton("MLX (Apple Silicon)")
        self.fmt_gguf_radio.setChecked(True)
        if not tools.is_apple_silicon():
            self.fmt_mlx_radio.setEnabled(False)
            self.fmt_mlx_radio.setToolTip("MLX は Apple Silicon Mac (arm64) 専用です")
        self.fmt_gguf_radio.toggled.connect(self._on_format_changed)
        fmt_row.addWidget(self.fmt_gguf_radio)
        fmt_row.addWidget(self.fmt_mlx_radio)
        fmt_row.addStretch(1)
        layout.addLayout(fmt_row)

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
        b_hf = QPushButton("HF からダウンロード…")
        b_hf.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        row.addWidget(self.path_edit, 1)
        row.addWidget(b_dir)
        row.addWidget(b_file)
        row.addWidget(b_hf)
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
        layout.addWidget(g1)

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
        self.imatrix_label = QLabel("imatrix:")
        opt_row.addWidget(self.imatrix_label)
        self.imatrix_edit = QLineEdit()
        self.imatrix_edit.setPlaceholderText("（任意）IQ1/IQ2/IQ3_XS 等に必要な重要度行列 .gguf / .dat")
        self.imatrix_edit.editingFinished.connect(self._rebuild_options)
        self.imatrix_btn = QPushButton("選ぶ…")
        self.imatrix_btn.clicked.connect(self._pick_imatrix)
        opt_row.addWidget(self.imatrix_edit, 1)
        opt_row.addWidget(self.imatrix_btn)
        l2.addLayout(opt_row)

        # 3. output + 4. ollama
        g3 = QGroupBox("3. 出力")
        f3 = QFormLayout(g3)
        f3.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.editingFinished.connect(self._on_out_dir_edited)
        b_out = QPushButton("選ぶ…")
        b_out.clicked.connect(self._pick_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(b_out)
        f3.addRow("出力フォルダ:", out_row)
        self.files_title_label = QLabel("できるファイル:")
        self.files_label = QLabel("-")
        self.files_label.setWordWrap(True)
        self.files_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.files_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f3.addRow(self.files_title_label, self.files_label)
        chk_row = QHBoxLayout()
        self.overwrite_chk = QCheckBox("同名のファイル・Ollama モデルを上書きする")
        self.overwrite_chk.toggled.connect(self._update_plan)
        self.keep_inter_chk = QCheckBox("中間ファイル（F16/BF16）を残す")
        self.keep_inter_chk.toggled.connect(self._update_plan)
        chk_row.addWidget(self.overwrite_chk)
        chk_row.addWidget(self.keep_inter_chk)
        chk_row.addStretch(1)
        f3.addRow("", chk_row)

        self.ollama_group = QGroupBox("4. Ollama")
        g4 = self.ollama_group
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
        layout.addWidget(split, 1)
        return w

    def _build_ollama_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. base model
        g1 = QGroupBox("1. ベースモデル（Ollama ダウンロード済みモデル）")
        l1 = QVBoxLayout(g1)
        r1 = QHBoxLayout()
        self.om_base_combo = QComboBox()
        self.om_base_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.om_base_combo.currentIndexChanged.connect(self._on_om_model_selected)
        self.om_refresh_btn = QPushButton("一覧を更新")
        self.om_refresh_btn.clicked.connect(self._refresh_ollama_models)
        self.om_load_btn = QPushButton("ベースの設定を読み込む")
        self.om_load_btn.clicked.connect(self._load_om_base_settings)
        r1.addWidget(QLabel("ベースモデル:"), 0)
        r1.addWidget(self.om_base_combo, 1)
        r1.addWidget(self.om_refresh_btn)
        r1.addWidget(self.om_load_btn)
        l1.addLayout(r1)

        self.om_info_label = QLabel("-")
        self.om_info_label.setStyleSheet("color: palette(placeholder-text);")
        self.om_info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        l1.addWidget(self.om_info_label)
        layout.addWidget(g1)

        # 2. model definition (Modelfile)
        g2 = QGroupBox("2. モデル定義（Modelfile）")
        l2 = QVBoxLayout(g2)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: scrollable form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(QLabel("システムプロンプト (SYSTEM):"))
        self.om_system_edit = QPlainTextEdit()
        self.om_system_edit.setPlaceholderText("モデルへのシステム指示（例: あなたは優秀なプログラミングアシスタントです。）")
        self.om_system_edit.setMaximumHeight(90)
        left_layout.addWidget(self.om_system_edit)

        f_params = QFormLayout()
        f_params.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        f_params.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.om_param_guides: dict[str, QLabel] = {}

        def add_param_row(key: str, label_text: str, edit_widget: QWidget, initial_guide: str = "") -> None:
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(10)
            edit_widget.setFixedWidth(120)
            h.addWidget(edit_widget, 0)
            guide = QLabel(initial_guide)
            guide.setStyleSheet("color: palette(placeholder-text);")
            h.addWidget(guide, 1)
            f_params.addRow(label_text, container)
            self.om_param_guides[key] = guide

        # 1. temperature
        self.om_temp_edit = QLineEdit()
        add_param_row("temperature", "temperature:", self.om_temp_edit, "デフォルト: 0.8  |  範囲: 0.0 〜 2.0")

        # 2. num_ctx
        self.om_ctx_combo = QComboBox()
        self.om_ctx_combo.setEditable(True)
        add_param_row("num_ctx", "num_ctx:", self.om_ctx_combo, "デフォルト: 2048  |  範囲: 512 〜 2,048 tokens")

        # 3. top_p
        self.om_topp_edit = QLineEdit()
        add_param_row("top_p", "top_p:", self.om_topp_edit, "デフォルト: 0.9  |  範囲: 0.0 〜 1.0")

        # 4. top_k
        self.om_topk_edit = QLineEdit()
        add_param_row("top_k", "top_k:", self.om_topk_edit, "デフォルト: 40  |  範囲: 1 〜 100")

        # 5. repeat_penalty
        self.om_rep_edit = QLineEdit()
        add_param_row("repeat_penalty", "repeat_penalty:", self.om_rep_edit, "デフォルト: 1.1  |  範囲: 0.0 〜 2.0")

        # 6. repeat_last_n
        self.om_replast_edit = QLineEdit()
        add_param_row("repeat_last_n", "repeat_last_n:", self.om_replast_edit, "デフォルト: 64  |  範囲: 0 〜 2,048 tokens")

        # 7. num_predict
        self.om_predict_edit = QLineEdit()
        add_param_row("num_predict", "num_predict:", self.om_predict_edit, "デフォルト: -1 (無制限)  |  範囲: -1 または 1 〜 2,048")

        # 8. seed
        self.om_seed_edit = QLineEdit()
        add_param_row("seed", "seed:", self.om_seed_edit, "デフォルト: 0 (ランダム)  |  範囲: 0 〜 2,147,483,647")

        # 9. stop
        self.om_stop_edit = QLineEdit()
        add_param_row("stop", "stop:", self.om_stop_edit, "停止文字列（例: <|im_end|>）")

        left_layout.addLayout(f_params)

        btn_apply = QPushButton("フォームの内容を Modelfile に反映 →")
        btn_apply.clicked.connect(self._apply_form_to_modelfile)
        left_layout.addWidget(btn_apply)
        left_layout.addStretch(1)

        scroll.setWidget(left_widget)
        splitter.addWidget(scroll)

        # Right: Modelfile text editor / preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Modelfile の内容（直接編集可）:"), 1)
        btn_reset = QPushButton("再構築 / クリア")
        btn_reset.clicked.connect(self._reset_modelfile)
        header_row.addWidget(btn_reset)
        right_layout.addLayout(header_row)

        self.om_modelfile_edit = QPlainTextEdit()
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.om_modelfile_edit.setFont(mono)
        self.om_modelfile_edit.setPlaceholderText("FROM <base_model>\nSYSTEM \"\"\"...\"\"\"\nPARAMETER temperature 0.7")
        right_layout.addWidget(self.om_modelfile_edit)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        l2.addWidget(splitter)
        layout.addWidget(g2, 1)

        # 3. Target model & Run
        g3 = QGroupBox("3. 作成設定 & 実行")
        l3 = QVBoxLayout(g3)
        row_cfg = QHBoxLayout()
        self.om_name_edit = QLineEdit()
        self.om_name_edit.setPlaceholderText("例: my-model:latest（英小文字、数字、. _ - :）")
        self.om_name_edit.textEdited.connect(self._on_om_name_edited)
        self.om_overwrite_chk = QCheckBox("同名のモデルを上書きする")
        self.om_overwrite_chk.setChecked(True)
        row_cfg.addWidget(QLabel("新規モデル名:"), 0)
        row_cfg.addWidget(self.om_name_edit, 1)
        row_cfg.addWidget(self.om_overwrite_chk, 0)
        l3.addLayout(row_cfg)

        row_run = QHBoxLayout()
        self.om_run_btn = QPushButton("モデルを作成（ollama create）")
        self.om_run_btn.setMinimumHeight(34)
        f = self.om_run_btn.font()
        f.setBold(True)
        self.om_run_btn.setFont(f)
        self.om_run_btn.clicked.connect(self._start_om_create)

        self.om_cancel_btn = QPushButton("取り消し")
        self.om_cancel_btn.setEnabled(False)
        self.om_cancel_btn.clicked.connect(self._cancel_om_create)

        self.om_progress = QProgressBar()
        self.om_progress.setRange(0, 1000)
        self.om_progress.setTextVisible(False)

        row_run.addWidget(self.om_run_btn)
        row_run.addWidget(self.om_cancel_btn)
        row_run.addWidget(self.om_progress, 1)
        l3.addLayout(row_run)

        self.om_status = QLabel("")
        self.om_status.setWordWrap(True)
        self.om_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        l3.addWidget(self.om_status)

        layout.addWidget(g3)

        # 4. Logs
        self.om_log = QPlainTextEdit()
        self.om_log.setReadOnly(True)
        self.om_log.setMaximumBlockCount(3000)
        self.om_log.setFont(mono)
        self.om_log.setPlaceholderText("実行ログ")
        self.om_log.setMaximumHeight(160)
        layout.addWidget(self.om_log)

        return w

    def _build_hf_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. Repository input & inspection
        g1 = QGroupBox("1. Hugging Face リポジトリ")
        l1 = QVBoxLayout(g1)

        r1 = QHBoxLayout()
        self.hf_repo_edit = QLineEdit()
        self.hf_repo_edit.setPlaceholderText("Repo ID または URL（例: Qwen/Qwen2.5-Coder-7B-Instruct または bartowski/Llama-3.2-1B-Instruct-GGUF）")
        self.hf_repo_edit.returnPressed.connect(self._inspect_hf_repo)
        self.hf_inspect_btn = QPushButton("情報を取得")
        self.hf_inspect_btn.clicked.connect(self._inspect_hf_repo)
        r1.addWidget(QLabel("リポジトリ:"), 0)
        r1.addWidget(self.hf_repo_edit, 1)
        r1.addWidget(self.hf_inspect_btn)
        l1.addLayout(r1)

        r2 = QHBoxLayout()
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setPlaceholderText("（任意）非公開モデル用の Hugging Face Access Token（環境変数 HF_TOKEN があれば自動使用）")
        r2.addWidget(QLabel("Token:"), 0)
        r2.addWidget(self.hf_token_edit, 1)
        l1.addLayout(r2)

        self.hf_info_label = QLabel("-")
        self.hf_info_label.setStyleSheet("color: palette(placeholder-text);")
        self.hf_info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        l1.addWidget(self.hf_info_label)
        layout.addWidget(g1)

        # 2. Download target selection
        g2 = QGroupBox("2. ダウンロード対象")
        l2 = QVBoxLayout(g2)

        self.hf_mode_repo_radio = QRadioButton("モデルリポジトリ全体（HF形式: config.json + safetensors + tokenizer）")
        self.hf_mode_repo_radio.setChecked(True)
        self.hf_mode_repo_radio.toggled.connect(self._on_hf_mode_changed)
        l2.addWidget(self.hf_mode_repo_radio)

        row_file = QHBoxLayout()
        self.hf_mode_file_radio = QRadioButton("単体 GGUF ファイルを選択:")
        self.hf_mode_file_radio.toggled.connect(self._on_hf_mode_changed)
        self.hf_file_combo = QComboBox()
        self.hf_file_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.hf_file_combo.setEnabled(False)
        row_file.addWidget(self.hf_mode_file_radio, 0)
        row_file.addWidget(self.hf_file_combo, 1)
        l2.addLayout(row_file)

        layout.addWidget(g2)

        # 3. Output directory & Download Run
        g3 = QGroupBox("3. 保存先 & ダウンロード実行")
        l3 = QVBoxLayout(g3)

        row_out = QHBoxLayout()
        self.hf_out_edit = QLineEdit()
        default_dir = str(Path("models").resolve())
        self.hf_out_edit.setText(default_dir)
        self.hf_pick_out_btn = QPushButton("選ぶ…")
        self.hf_pick_out_btn.clicked.connect(self._pick_hf_out_dir)
        row_out.addWidget(QLabel("保存先フォルダ:"), 0)
        row_out.addWidget(self.hf_out_edit, 1)
        row_out.addWidget(self.hf_pick_out_btn)
        l3.addLayout(row_out)

        row_run = QHBoxLayout()
        self.hf_download_btn = QPushButton("ダウンロード開始")
        self.hf_download_btn.setMinimumHeight(34)
        f = self.hf_download_btn.font()
        f.setBold(True)
        self.hf_download_btn.setFont(f)
        self.hf_download_btn.clicked.connect(self._start_hf_download)

        self.hf_cancel_btn = QPushButton("取り消し")
        self.hf_cancel_btn.setEnabled(False)
        self.hf_cancel_btn.clicked.connect(self._cancel_hf_download)

        self.hf_progress = QProgressBar()
        self.hf_progress.setRange(0, 1000)
        self.hf_progress.setTextVisible(False)

        self.hf_set_base_btn = QPushButton("このモデルを量子化ベースに設定 →")
        self.hf_set_base_btn.setEnabled(False)
        self.hf_set_base_btn.clicked.connect(self._set_downloaded_as_base)

        row_run.addWidget(self.hf_download_btn)
        row_run.addWidget(self.hf_cancel_btn)
        row_run.addWidget(self.hf_progress, 1)
        row_run.addWidget(self.hf_set_base_btn)
        l3.addLayout(row_run)

        self.hf_status = QLabel("")
        self.hf_status.setWordWrap(True)
        self.hf_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        l3.addWidget(self.hf_status)

        layout.addWidget(g3)

        # 4. Logs
        self.hf_log = QPlainTextEdit()
        self.hf_log.setReadOnly(True)
        self.hf_log.setMaximumBlockCount(3000)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.hf_log.setFont(mono)
        self.hf_log.setPlaceholderText("ダウンロード実行ログ")
        layout.addWidget(self.hf_log, 1)

        return w

    def _build_chat_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 1. Model & parameters
        g1 = QGroupBox("1. モデル・設定")
        l1 = QVBoxLayout(g1)

        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("実行基盤:"))
        self.chat_mlx_radio = QRadioButton("MLX (ローカル)")
        self.chat_ollama_radio = QRadioButton("Ollama")
        if tools.is_apple_silicon() and tools.mlx_lm_available():
            self.chat_mlx_radio.setChecked(True)
        else:
            self.chat_mlx_radio.setEnabled(False)
            self.chat_mlx_radio.setToolTip("MLX は Apple Silicon Mac (mlx-lm 導入済み) 専用です")
            self.chat_ollama_radio.setChecked(True)
        self.chat_mlx_radio.toggled.connect(self._on_chat_backend_changed)
        row_top.addWidget(self.chat_mlx_radio)
        row_top.addWidget(self.chat_ollama_radio)
        row_top.addSpacing(12)

        row_top.addWidget(QLabel("モデル:"))
        self.chat_model_combo = QComboBox()
        self.chat_model_combo.setEditable(True)
        self.chat_model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row_top.addWidget(self.chat_model_combo, 1)

        self.chat_refresh_btn = QPushButton("更新")
        self.chat_refresh_btn.clicked.connect(self._refresh_chat_models)
        self.chat_pick_btn = QPushButton("フォルダを選ぶ…")
        self.chat_pick_btn.clicked.connect(self._pick_chat_mlx_dir)
        self.chat_unload_btn = QPushButton("メモリ解放")
        self.chat_unload_btn.setToolTip("ロード中の MLX モデルをメモリから解放します")
        self.chat_unload_btn.clicked.connect(self._unload_chat_model)
        row_top.addWidget(self.chat_refresh_btn)
        row_top.addWidget(self.chat_pick_btn)
        row_top.addWidget(self.chat_unload_btn)
        l1.addLayout(row_top)

        row_param = QHBoxLayout()
        row_param.addWidget(QLabel("System:"))
        self.chat_system_edit = QLineEdit()
        self.chat_system_edit.setPlaceholderText("（任意）システムプロンプト（例: 日本語で簡潔に答えてください）")
        row_param.addWidget(self.chat_system_edit, 1)

        row_param.addSpacing(12)
        row_param.addWidget(QLabel("Temp:"))
        self.chat_temp_spin = QDoubleSpinBox()
        self.chat_temp_spin.setRange(0.0, 2.0)
        self.chat_temp_spin.setSingleStep(0.1)
        self.chat_temp_spin.setValue(0.7)
        row_param.addWidget(self.chat_temp_spin)

        row_param.addWidget(QLabel("Max tokens:"))
        self.chat_max_tok_spin = QSpinBox()
        self.chat_max_tok_spin.setRange(64, 32768)
        self.chat_max_tok_spin.setSingleStep(128)
        self.chat_max_tok_spin.setValue(1024)
        row_param.addWidget(self.chat_max_tok_spin)
        l1.addLayout(row_param)
        layout.addWidget(g1)

        # 2. Conversation view
        self.chat_view = QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setPlaceholderText("ここに会話が表示されます")
        layout.addWidget(self.chat_view, 1)

        # 3. Input & controls
        input_row = QHBoxLayout()
        self.chat_input = QPlainTextEdit()
        self.chat_input.setMaximumHeight(84)
        self.chat_input.setPlaceholderText("メッセージを入力（Enter で送信 / Shift+Enter で改行）")
        self.chat_input.installEventFilter(self)
        input_row.addWidget(self.chat_input, 1)

        btn_col = QVBoxLayout()
        self.chat_send_btn = QPushButton("送信")
        self.chat_send_btn.setMinimumHeight(32)
        f = self.chat_send_btn.font()
        f.setBold(True)
        self.chat_send_btn.setFont(f)
        self.chat_send_btn.clicked.connect(self._send_chat)

        sub_row = QHBoxLayout()
        self.chat_stop_btn = QPushButton("停止")
        self.chat_stop_btn.setEnabled(False)
        self.chat_stop_btn.clicked.connect(self._stop_chat)
        self.chat_clear_btn = QPushButton("クリア")
        self.chat_clear_btn.clicked.connect(self._clear_chat)
        sub_row.addWidget(self.chat_stop_btn)
        sub_row.addWidget(self.chat_clear_btn)

        btn_col.addWidget(self.chat_send_btn)
        btn_col.addLayout(sub_row)
        input_row.addLayout(btn_col)
        layout.addLayout(input_row)

        self.chat_status = QLabel("モデルを選んでメッセージを送信してください")
        self.chat_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.chat_status)

        self._on_chat_backend_changed()
        return w

    def _on_tab_changed(self, index: int) -> None:
        if index == 1 and not self._om_initialized:
            self._refresh_ollama_models()
            self._om_initialized = True
        elif index == 3 and not self._chat_initialized:
            self._refresh_chat_models()
            self._chat_initialized = True

        vram = tools.gpu_vram_bytes()
        self.info_labels["GPU"].setText(f"VRAM {human_bytes(vram)}" if vram else "（検出なし）")

    # ------------------------------------------------------------ tools
    def target_format(self) -> str:
        return catalog.FMT_MLX if self.fmt_mlx_radio.isChecked() else catalog.FMT_GGUF

    def _on_format_changed(self) -> None:
        is_mlx = self.target_format() == catalog.FMT_MLX
        for w in (self.imatrix_label, self.imatrix_edit, self.imatrix_btn, self.keep_inter_chk, self.ollama_group):
            w.setVisible(not is_mlx)
        self.files_title_label.setText("できるフォルダ:" if is_mlx else "できるファイル:")
        self.table.setHorizontalHeaderLabels(["", "種類", "サイズ", "メモリ" if is_mlx else "GPU", "品質", "説明・注意"])
        self._refresh_tools()
        if self.base is not None:
            self.table.setRowCount(0)
            self._render_base_info()
            self._rebuild_options()

    def _refresh_tools(self) -> None:
        st = tools.tool_status()
        probs = st.mlx_missing if self.target_format() == catalog.FMT_MLX else st.missing
        if probs:
            self._set_status("<br>".join(probs), self.c_red)
        elif self.base is None:
            self._set_status("", None)

    # ------------------------------------------------------------ pickers / dnd
    def _start_dir(self) -> str:
        t = self.path_edit.text().strip()
        if t:
            p = Path(t)
            return str(p.parent if p.is_file() else p)
        models_dir = Path("models").resolve()
        if models_dir.is_dir():
            return str(models_dir)
        return str(Path.cwd())

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
        self._render_base_info()

        if self._out_dir_auto or not self.out_edit.text().strip():
            p = Path(path)
            self.out_edit.setText(str(p.parent if p.is_file() else p.parent))
            self._out_dir_auto = True
        if self._ollama_stem_auto or not self.ollama_edit.text().strip():
            self.ollama_edit.setText(ollama_name(base.stem, "x").split(":")[0])
            self._ollama_stem_auto = True
        self._rebuild_options()

    def _render_base_info(self) -> None:
        base = self.base
        if base is None:
            return
        fmt = self.target_format()
        L = self.info_labels
        kind = {
            KIND_HF: "HF 形式（変換 → 量子化）",
            KIND_GGUF: "GGUF（量子化のみ）",
            KIND_MLX: "MLX 量子化済み",
        }.get(base.kind, "対象外")
        L["形式"].setText(kind)
        L["名前"].setText(base.name or "-")
        if fmt == catalog.FMT_MLX and base.mlx_model_type:
            arch = base.mlx_model_type
            if base.mlx_supported is True:
                arch += "（MLX 対応）"
            elif base.mlx_supported is False:
                arch += "（MLX 非対応）"
        else:
            arch = base.architecture or "-"
            if base.kind == KIND_HF and base.arch_supported:
                arch += "（変換可）"
        if base.is_moe:
            arch += " / MoE"
        L["構造"].setText(arch)
        src = base.source_type or "-"
        if base.is_quantized and base.kind != KIND_MLX:
            src += "（量子化済み）"
        L["元の形式"].setText(src)
        L["パラメータ"].setText(human_params(base.n_params) if base.n_params else "-")
        L["重み"].setText(human_bytes(base.weight_bytes) if base.weight_bytes else "-")
        L["チャットテンプレート"].setText({True: "あり", False: "なし", None: "-"}[base.has_chat_template])
        for k in ("名前", "構造"):
            L[k].setToolTip(L[k].text())

        msgs = []
        red, amber = self.c_red.name(), self.c_amber.name()
        for p in base.problems_for(fmt):
            msgs.append(f'<span style="color:{red}">作れません: {_esc(p)}</span>')
        for w in base.warnings:
            msgs.append(f'<span style="color:{amber}">注意: {_esc(w)}</span>')
        self.msg_label.setText("<br>".join(msgs))
        self.msg_label.setVisible(bool(msgs))

    def _imatrix(self) -> Path | None:
        t = self.imatrix_edit.text().strip().strip('"')
        return Path(t) if t else None

    def _rebuild_options(self) -> None:
        if self.base is None:
            return
        fmt = self.target_format()
        checked = set(self.selected_types())
        env = Environment.detect(Path(self.out_edit.text().strip() or "."))
        self.options = build_options(self.base, env, self._imatrix() if fmt == catalog.FMT_GGUF else None, target_format=fmt)
        self._fill_table(checked)
        if fmt == catalog.FMT_GGUF:
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
        fmt = self.target_format()
        sel = self.selected_types()
        stem = self.ollama_edit.text().strip().lower() or ollama_name(self.base.stem, "x").split(":")[0]
        is_gguf = fmt == catalog.FMT_GGUF
        return JobSpec(
            base=self.base, targets=sel, out_dir=Path(self.out_edit.text().strip() or "."),
            imatrix=self._imatrix() if is_gguf else None,
            keep_intermediate=self.keep_inter_chk.isChecked() if is_gguf else False,
            overwrite=self.overwrite_chk.isChecked(),
            register_ollama=self.ollama_chk.isChecked() if is_gguf else False,
            ollama_names={t: f"{stem}:{t.lower()}" for t in sel} if is_gguf else {},
            target_format=fmt,
        )

    def _update_plan(self) -> None:
        spec = self._spec()
        running = self.runner is not None
        if spec is None:
            self.run_btn.setEnabled(False)
            return
        sel = spec.targets
        self.files_label.setText("\n".join(spec.output_path(t).name for t in sel) or "-")
        self.ollama_edit.setEnabled(spec.register_ollama)
        self.ollama_preview.setText(
            ("\n".join(spec.ollama_name_for(t) for t in sel) or "-") if spec.register_ollama else "（登録しない）")
        errs = check_outputs(spec)
        self.run_btn.setEnabled(not errs and not running)
        if running:
            return
        if errs:
            self._set_status("<br>".join(_esc(e) for e in errs),
                             self.c_red if spec.base.ok_for(spec.target_format) and sel else self.c_grey)
        else:
            total = sum(o.size_bytes for o in self.options if o.name in sel)
            steps = plan(spec)
            label = "MLX モデル" if spec.target_format == catalog.FMT_MLX else "GGUF"
            self._set_status(f"準備よし: {len(sel)} 個の {label}（合計 約 {human_bytes(total)}）を {len(steps)} 手順で作ります",
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
                if self._om_initialized:
                    self._refresh_ollama_models()
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

    # ------------------------------------------------------------ Ollama Modelfile tab
    def _refresh_ollama_models(self) -> None:
        tools.forget_ollama_cache()
        models = tools.ollama_list_models(max_age=0.0)
        self.om_models = models or []
        current = self.om_base_combo.currentText()
        self.om_base_combo.blockSignals(True)
        self.om_base_combo.clear()
        if not self.om_models:
            self.om_base_combo.addItem("（モデルが見つかりません）")
            self.om_info_label.setText("Ollama にダウンロード済みのモデルがありません。'ollama pull <name>' でモデルを取得してください。")
            self.om_base_combo.blockSignals(False)
            return

        for m in self.om_models:
            self.om_base_combo.addItem(m.name)

        idx = 0
        if current:
            found = self.om_base_combo.findText(current)
            if found >= 0:
                idx = found
        self.om_base_combo.setCurrentIndex(idx)
        self.om_base_combo.blockSignals(False)
        self._on_om_model_selected(idx)

    def _on_om_model_selected(self, index: int) -> None:
        if not self.om_models or index < 0 or index >= len(self.om_models):
            self.om_info_label.setText("-")
            return
        m = self.om_models[index]
        self._update_model_info_and_guides(m.name, m)

        # Auto-suggest new model name if not manually edited
        if self._om_name_auto or not self.om_name_edit.text().strip():
            stem = m.name.split(":")[0].split("/")[-1]
            self.om_name_edit.setText(f"{stem}-custom:latest")
            self._om_name_auto = True

        # If modelfile is empty or just FROM, set new FROM
        current_mf = self.om_modelfile_edit.toPlainText().strip()
        if not current_mf or (current_mf.startswith("FROM ") and "\n" not in current_mf):
            self.om_modelfile_edit.setPlainText(f"FROM {m.name}\n")

    def _update_model_info_and_guides(self, model_name: str, item: tools.OllamaModelItem | None = None) -> None:
        details = tools.ollama_show_details(model_name)
        self.om_details = details

        ctx_len = details.context_length if details else 2048
        arch = details.architecture if details else ""
        param_cnt = details.parameters_size if details else ""
        quant = details.quantization if details else ""
        size_str = item.size if item else ""

        info_parts = []
        if item and item.id:
            info_parts.append(f"ID: {item.id}")
        if arch:
            info_parts.append(f"アーキテクチャ: {arch}")
        if param_cnt:
            info_parts.append(f"パラメータ: {param_cnt}")
        info_parts.append(f"最大コンテキスト: {ctx_len:,} tokens")
        if quant:
            info_parts.append(f"量子化: {quant}")
        if size_str:
            info_parts.append(f"サイズ: {size_str}")

        self.om_info_label.setText("  |  ".join(info_parts) if info_parts else "-")

        # Update parameter guides and dropdown options based on model info
        specs = om.get_param_specs(context_length=ctx_len, model_params=details.model_parameters if details else None)
        specs_dict = {s.key: s for s in specs}

        # temperature
        if "temperature" in self.om_param_guides and "temperature" in specs_dict:
            s = specs_dict["temperature"]
            self.om_param_guides["temperature"].setText(f"デフォルト: {s.default}  |  {s.range_text()}")
            self.om_temp_edit.setPlaceholderText(f"既定: {s.default}")

        # num_ctx
        if "num_ctx" in self.om_param_guides and "num_ctx" in specs_dict:
            s = specs_dict["num_ctx"]
            self.om_param_guides["num_ctx"].setText(f"デフォルト: {s.default}  |  {s.range_text()} (モデル上限)")
            cur_ctx = self.om_ctx_combo.currentText()
            candidates = [c for c in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576] if c <= ctx_len]
            if ctx_len not in candidates and ctx_len > 512:
                candidates.append(ctx_len)
            self.om_ctx_combo.blockSignals(True)
            self.om_ctx_combo.clear()
            self.om_ctx_combo.addItem("")
            for c in sorted(candidates):
                self.om_ctx_combo.addItem(str(c))
            if cur_ctx:
                self.om_ctx_combo.setEditText(cur_ctx)
            self.om_ctx_combo.setPlaceholderText(f"未入力時はベース ({s.default}) に従う")
            self.om_ctx_combo.blockSignals(False)

        # top_p
        if "top_p" in self.om_param_guides and "top_p" in specs_dict:
            s = specs_dict["top_p"]
            self.om_param_guides["top_p"].setText(f"デフォルト: {s.default}  |  {s.range_text()}")
            self.om_topp_edit.setPlaceholderText(f"既定: {s.default}")

        # top_k
        if "top_k" in self.om_param_guides and "top_k" in specs_dict:
            s = specs_dict["top_k"]
            self.om_param_guides["top_k"].setText(f"デフォルト: {s.default}  |  {s.range_text()}")
            self.om_topk_edit.setPlaceholderText(f"既定: {s.default}")

        # repeat_penalty
        if "repeat_penalty" in self.om_param_guides and "repeat_penalty" in specs_dict:
            s = specs_dict["repeat_penalty"]
            self.om_param_guides["repeat_penalty"].setText(f"デフォルト: {s.default}  |  {s.range_text()}")
            self.om_rep_edit.setPlaceholderText(f"既定: {s.default}")

        # repeat_last_n
        if "repeat_last_n" in self.om_param_guides and "repeat_last_n" in specs_dict:
            s = specs_dict["repeat_last_n"]
            self.om_param_guides["repeat_last_n"].setText(f"デフォルト: {s.default}  |  {s.range_text()}")
            self.om_replast_edit.setPlaceholderText(f"既定: {s.default}")

        # num_predict
        if "num_predict" in self.om_param_guides and "num_predict" in specs_dict:
            s = specs_dict["num_predict"]
            self.om_param_guides["num_predict"].setText(f"デフォルト: -1 (無制限)  |  範囲: -1 または 1 〜 {ctx_len:,}")
            self.om_predict_edit.setPlaceholderText("既定: -1")

        # seed
        if "seed" in self.om_param_guides and "seed" in specs_dict:
            s = specs_dict["seed"]
            self.om_param_guides["seed"].setText(f"デフォルト: 0 (ランダム)  |  {s.range_text()}")
            self.om_seed_edit.setPlaceholderText("既定: 0")

    def _on_om_name_edited(self, _text: str) -> None:
        self._om_name_auto = False

    def _load_om_base_settings(self) -> None:
        idx = self.om_base_combo.currentIndex()
        if not self.om_models or idx < 0 or idx >= len(self.om_models):
            self._set_om_status("モデルが選択されていません", self.c_red)
            return
        m = self.om_models[idx]
        self._set_om_status(f"'{m.name}' の Modelfile を取得中...", None)
        QApplication.processEvents()

        raw_show = tools.ollama_show_modelfile(m.name)
        if not raw_show:
            self._set_om_status(f"'{m.name}' の設定を取得できませんでした（ollama show に失敗しました）", self.c_red)
            return

        cleaned = om.clean_show_modelfile(raw_show, m.name)
        self.om_modelfile_edit.setPlainText(cleaned)

        sec = om.parse_modelfile_sections(cleaned)
        self.om_system_edit.setPlainText(sec["system"])

        # Populate param inputs if present in Modelfile or in model_parameters
        params = dict(sec["parameters"])
        if self.om_details and self.om_details.model_parameters:
            for k, v in self.om_details.model_parameters.items():
                if k not in params:
                    params[k] = v

        if "temperature" in params:
            self.om_temp_edit.setText(params["temperature"])
        if "num_ctx" in params:
            self.om_ctx_combo.setEditText(params["num_ctx"])
        if "top_p" in params:
            self.om_topp_edit.setText(params["top_p"])
        if "top_k" in params:
            self.om_topk_edit.setText(params["top_k"])
        if "repeat_penalty" in params:
            self.om_rep_edit.setText(params["repeat_penalty"])
        if "repeat_last_n" in params:
            self.om_replast_edit.setText(params["repeat_last_n"])
        if "num_predict" in params:
            self.om_predict_edit.setText(params["num_predict"])
        if "seed" in params:
            self.om_seed_edit.setText(params["seed"])
        if "stop" in params:
            self.om_stop_edit.setText(params["stop"])

        self._set_om_status(f"'{m.name}' の Modelfile 設定を読み込みました", None)

    def _apply_form_to_modelfile(self) -> None:
        idx = self.om_base_combo.currentIndex()
        base_name = self.om_models[idx].name if (self.om_models and 0 <= idx < len(self.om_models)) else "base"

        # Check existing modelfile text to preserve template or extras
        cur_text = self.om_modelfile_edit.toPlainText().strip()
        sec = om.parse_modelfile_sections(cur_text) if cur_text else {}
        template = sec.get("template", "")
        extra = sec.get("extra", "")

        params: list[tuple[str, str]] = []
        if self.om_temp_edit.text().strip():
            params.append(("temperature", self.om_temp_edit.text().strip()))
        if self.om_ctx_combo.currentText().strip():
            params.append(("num_ctx", self.om_ctx_combo.currentText().strip()))
        if self.om_topp_edit.text().strip():
            params.append(("top_p", self.om_topp_edit.text().strip()))
        if self.om_topk_edit.text().strip():
            params.append(("top_k", self.om_topk_edit.text().strip()))
        if self.om_rep_edit.text().strip():
            params.append(("repeat_penalty", self.om_rep_edit.text().strip()))
        if self.om_replast_edit.text().strip():
            params.append(("repeat_last_n", self.om_replast_edit.text().strip()))
        if self.om_predict_edit.text().strip():
            params.append(("num_predict", self.om_predict_edit.text().strip()))
        if self.om_seed_edit.text().strip():
            params.append(("seed", self.om_seed_edit.text().strip()))
        if self.om_stop_edit.text().strip():
            params.append(("stop", self.om_stop_edit.text().strip()))

        # Preserve any additional parameters not in standard form
        existing_param_keys = {p[0] for p in params}
        for k, v in sec.get("parameters", []):
            if k not in existing_param_keys:
                params.append((k, v))

        mf = om.build_modelfile(
            from_model=base_name,
            system=self.om_system_edit.toPlainText().strip(),
            parameters=params,
            template=template,
            extra=extra,
        )
        self.om_modelfile_edit.setPlainText(mf)
        self._set_om_status("フォームの内容を Modelfile に反映しました", None)

    def _reset_modelfile(self) -> None:
        idx = self.om_base_combo.currentIndex()
        base_name = self.om_models[idx].name if (self.om_models and 0 <= idx < len(self.om_models)) else ""
        self.om_modelfile_edit.setPlainText(f"FROM {base_name}\n" if base_name else "")
        self.om_system_edit.clear()
        self.om_temp_edit.clear()
        self.om_ctx_combo.setCurrentIndex(0)
        self.om_topp_edit.clear()
        self.om_topk_edit.clear()
        self.om_rep_edit.clear()
        self.om_replast_edit.clear()
        self.om_predict_edit.clear()
        self.om_seed_edit.clear()
        self.om_stop_edit.clear()
        self._set_om_status("Modelfile を初期化しました", None)

    def _start_om_create(self) -> None:
        target_name = self.om_name_edit.text().strip()
        err = om.validate_ollama_model_name(target_name)
        if err:
            self._set_om_status(_esc(err), self.c_red)
            return

        mf_content = self.om_modelfile_edit.toPlainText().strip()
        if not mf_content:
            self._set_om_status("Modelfile が空です。内容を入力してください。", self.c_red)
            return
        if not any(line.strip().startswith("FROM ") for line in mf_content.splitlines()):
            self._set_om_status("Modelfile に 'FROM <ベースモデル>' がありません。", self.c_red)
            return

        overwrite = self.om_overwrite_chk.isChecked()
        if not overwrite:
            existing = tools.ollama_model_names()
            if existing and target_name.lower() in existing:
                self._set_om_status(f"同名のモデル '{target_name}' が存在します（上書きに印を付けてください）", self.c_red)
                return

        self.om_run_btn.setEnabled(False)
        self.om_cancel_btn.setEnabled(True)
        self.om_progress.setValue(0)
        self._set_om_status(f"モデル '{target_name}' を作成中...", None)
        self.om_log.clear()

        job = om.OllamaCreateJob(
            name=target_name,
            modelfile_content=mf_content,
            overwrite=overwrite,
            on_log=self.om_sig.log.emit,
            on_progress=self.om_sig.progress.emit,
        )
        self.om_job = job

        def work() -> None:
            ok = job.run()
            self.om_sig.finished.emit(ok)

        threading.Thread(target=work, daemon=True).start()

    def _cancel_om_create(self) -> None:
        if self.om_job:
            self.om_job.cancel()
            self._set_om_status("取り消し中...", self.c_amber)

    def _append_om_log(self, text: str) -> None:
        self.om_log.appendPlainText(text)

    def _on_om_progress(self, frac: float, text: str) -> None:
        self.om_progress.setValue(int(frac * 1000))
        self._set_om_status(f"実行中 {frac:.0%}: {_esc(text)}", None)

    def _on_om_finished(self, success: bool) -> None:
        self.om_job = None
        self.om_run_btn.setEnabled(True)
        self.om_cancel_btn.setEnabled(False)
        if success:
            self.om_progress.setValue(1000)
            target = self.om_name_edit.text().strip()
            self._set_om_status(f"作成完了: {target} （実行: ollama run {target}）", None)
            self._refresh_ollama_models()
        else:
            self.om_progress.setValue(0)
            self._set_om_status("モデル作成に失敗しました（ログを確認してください）", self.c_red)

    def _set_om_status(self, html: str, color: QColor | None) -> None:
        if color is not None:
            html = f'<span style="color:{color.name()}">{html}</span>'
        self.om_status.setText(html)

    # ------------------------------------------------------------ Hugging Face tab
    def _inspect_hf_repo(self) -> None:
        raw_input = self.hf_repo_edit.text().strip()
        repo_id, target_file = hfd.parse_hf_input(raw_input)
        if not repo_id:
            self._set_hf_status("リポジトリ名または URL を入力してください", self.c_red)
            return

        token = self.hf_token_edit.text().strip() or None
        self._set_hf_status(f"'{repo_id}' の情報を取得中...", None)
        self.hf_inspect_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            info = hfd.inspect_hf_repo(repo_id, token=token)
            self.hf_info = info
            if not info.is_valid:
                self._set_hf_status(_esc(info.error_message), self.c_red)
                self.hf_info_label.setText("-")
                return

            parts = [f"リポジトリ: {info.repo_id}", f"総ファイル数: {info.total_files}"]
            if info.has_hf_weights:
                parts.append("HF形式の重み: あり (safetensors/bin)")
            if info.is_mlx:
                parts.append("MLXモデル: はい")
            if info.gguf_files:
                parts.append(f"GGUFファイル: {len(info.gguf_files)} 個")

            self.hf_info_label.setText("  |  ".join(parts))

            # Populate GGUF combo
            self.hf_file_combo.clear()
            for g in info.gguf_files:
                sz = info.file_sizes.get(g)
                sz_str = f" ({human_bytes(sz)})" if sz else ""
                self.hf_file_combo.addItem(f"{g}{sz_str}", g)

            # Auto-suggest mode and target
            if target_file and target_file in info.gguf_files:
                idx = self.hf_file_combo.findData(target_file)
                if idx >= 0:
                    self.hf_file_combo.setCurrentIndex(idx)
                self.hf_mode_file_radio.setChecked(True)
            elif not info.has_hf_weights and info.gguf_files:
                self.hf_mode_file_radio.setChecked(True)
            else:
                self.hf_mode_repo_radio.setChecked(True)

            # Suggest local output directory
            clean_name = repo_id.replace("/", "_")
            suggested_dir = (Path("models") / clean_name).resolve()
            self.hf_out_edit.setText(str(suggested_dir))

            if hfd.has_incomplete_cache(suggested_dir):
                self.hf_download_btn.setText("続きから再開")
                self._set_hf_status(f"'{repo_id}' の情報を取得しました（途中のキャッシュあり：続きから再開できます）", None)
            else:
                self.hf_download_btn.setText("ダウンロード開始")
                self._set_hf_status(f"'{repo_id}' の情報を取得しました", None)

        finally:
            self.hf_inspect_btn.setEnabled(True)

    def _on_hf_mode_changed(self) -> None:
        is_file = self.hf_mode_file_radio.isChecked()
        self.hf_file_combo.setEnabled(is_file)

    def _pick_hf_out_dir(self) -> None:
        cur = self.hf_out_edit.text().strip() or str(Path("models").resolve())
        d = QFileDialog.getExistingDirectory(self, "保存先フォルダを選択", cur)
        if d:
            self.hf_out_edit.setText(d)

    def _start_hf_download(self) -> None:
        raw_input = self.hf_repo_edit.text().strip()
        repo_id, explicit_file = hfd.parse_hf_input(raw_input)
        if not repo_id:
            self._set_hf_status("リポジトリ名または URL を入力してください", self.c_red)
            return

        out_dir_str = self.hf_out_edit.text().strip()
        if not out_dir_str:
            self._set_hf_status("保存先フォルダを指定してください", self.c_red)
            return

        target_file = None
        if self.hf_mode_file_radio.isChecked():
            target_file = self.hf_file_combo.currentData() or explicit_file
            if not target_file:
                self._set_hf_status("ダウンロードする GGUF ファイルを選択してください", self.c_red)
                return

        token = self.hf_token_edit.text().strip() or None
        out_dir = Path(out_dir_str)

        self.hf_download_btn.setEnabled(False)
        self.hf_cancel_btn.setEnabled(True)
        self.hf_set_base_btn.setEnabled(False)
        self.hf_progress.setValue(0)
        self._set_hf_status("ダウンロードを開始します...", None)
        self.hf_log.clear()

        job = hfd.HfDownloadJob(
            repo_id=repo_id,
            local_dir=out_dir,
            filename=target_file,
            token=token,
            on_log=self.hf_sig.log.emit,
            on_progress=self.hf_sig.progress.emit,
        )
        self.hf_job = job

        def work() -> None:
            try:
                dest = job.run()
                self.hf_sig.finished.emit(dest, None)
            except BaseException as e:
                self.hf_sig.finished.emit(None, e)

        threading.Thread(target=work, daemon=True).start()

    def _cancel_hf_download(self) -> None:
        if self.hf_job:
            self.hf_job.cancel()
            self._set_hf_status("取り消し中...", self.c_amber)

    def _append_hf_log(self, text: str) -> None:
        self.hf_log.appendPlainText(text)

    def _on_hf_progress(self, frac: float, text: str) -> None:
        self.hf_progress.setValue(int(frac * 1000))
        self._set_hf_status(f"ダウンロード中 {frac:.0%}: {_esc(text)}", None)

    def _on_hf_finished(self, dest_path: object, err: object) -> None:
        self.hf_job = None
        self.hf_download_btn.setEnabled(True)
        self.hf_cancel_btn.setEnabled(False)

        if err is None and dest_path:
            self.hf_progress.setValue(1000)
            p = Path(str(dest_path))
            self.hf_downloaded_path = p
            self.hf_set_base_btn.setEnabled(True)
            self.hf_download_btn.setText("ダウンロード開始")
            self._set_hf_status(f"ダウンロード完了: {p.name} （「このモデルを量子化ベースに設定」で直接量子化へ進めます）", None)
            self._append_hf_log(f"\n[完了] 保存先: {p}")
        else:
            self.hf_progress.setValue(0)
            is_cancel = isinstance(err, InterruptedError) or (err and "取り消し" in str(err))
            msg = "ダウンロードを取り消しました" if is_cancel else (str(err) if err else "不明なエラー")
            out_dir = Path(self.hf_out_edit.text().strip() or ".")
            if hfd.has_incomplete_cache(out_dir):
                self.hf_download_btn.setText("続きから再開")
                msg += "（ボタンを押すと続きから再開できます）"
            else:
                self.hf_download_btn.setText("ダウンロード開始")
            self._set_hf_status(f"{_esc(msg)}", self.c_amber if is_cancel else self.c_red)
            self._append_hf_log(f"\n[{'取り消し' if is_cancel else 'エラー'}] {msg}")

    def _set_downloaded_as_base(self) -> None:
        if not self.hf_downloaded_path or not self.hf_downloaded_path.exists():
            return
        p_str = str(self.hf_downloaded_path)
        self.path_edit.setText(p_str)
        self.tabs.setCurrentIndex(0)
        self.load_base(p_str)
        if self.base and self.base.kind == KIND_MLX and self.fmt_mlx_radio.isEnabled():
            self.fmt_mlx_radio.setChecked(True)

    def _set_hf_status(self, html: str, color: QColor | None) -> None:
        if color is not None:
            html = f'<span style="color:{color.name()}">{html}</span>'
        self.hf_status.setText(html)

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.runner is not None:
            ans = QMessageBox.question(self, "実行中", "量子化を実行中です。取り消して閉じますか？")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.runner.cancel()
        if self.om_job is not None:
            ans = QMessageBox.question(self, "実行中", "Ollama モデル作成を実行中です。取り消して閉じますか？")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.om_job.cancel()
        if self.hf_job is not None:
            ans = QMessageBox.question(self, "実行中", "Hugging Face からのダウンロードを実行中です。取り消して閉じますか？")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.hf_job.cancel()
        if self.chat_cancel is not None:
            self.chat_cancel.set()
        self.chat_session.unload()
        e.accept()

    # ------------------------------------------------------------ chat tab
    def _chat_backend(self) -> str:
        return "mlx" if self.chat_mlx_radio.isChecked() else "ollama"

    def _on_chat_backend_changed(self) -> None:
        is_mlx = self._chat_backend() == "mlx"
        self.chat_pick_btn.setVisible(is_mlx)
        self.chat_unload_btn.setVisible(is_mlx)
        self._refresh_chat_models()

    def _refresh_chat_models(self) -> None:
        cur = self.chat_model_combo.currentData() or self.chat_model_combo.currentText().strip()
        self.chat_model_combo.clear()
        if self._chat_backend() == "mlx":
            dirs = ce.list_local_mlx_models()
            for d in dirs:
                self.chat_model_combo.addItem(d.name, str(d))
            if cur:
                idx = self.chat_model_combo.findData(cur)
                if idx >= 0:
                    self.chat_model_combo.setCurrentIndex(idx)
                elif Path(cur).is_dir():
                    self.chat_model_combo.addItem(Path(cur).name, cur)
                    self.chat_model_combo.setCurrentIndex(self.chat_model_combo.count() - 1)
        else:
            items = tools.ollama_list_models(max_age=0.0) or []
            for m in items:
                self.chat_model_combo.addItem(m.name, m.name)
            if cur:
                idx = self.chat_model_combo.findData(cur)
                if idx >= 0:
                    self.chat_model_combo.setCurrentIndex(idx)

    def _pick_chat_mlx_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "MLX モデルフォルダを選択", self._start_dir())
        if not d:
            return
        resolved = str(Path(d).resolve())
        idx = self.chat_model_combo.findData(resolved)
        if idx < 0:
            self.chat_model_combo.addItem(Path(resolved).name, resolved)
            idx = self.chat_model_combo.count() - 1
        self.chat_model_combo.setCurrentIndex(idx)

    def _unload_chat_model(self) -> None:
        if self.chat_running:
            return
        self.chat_session.unload()
        self.chat_status.setText("MLX モデルをメモリから解放しました")

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is getattr(self, "chat_input", None) and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                self._send_chat()
                return True
        return super().eventFilter(obj, event)

    def _append_chat_text(self, text: str) -> None:
        cur = self.chat_view.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.insertText(text)
        self.chat_view.setTextCursor(cur)
        self.chat_view.ensureCursorVisible()

    def _send_chat(self) -> None:
        if self.chat_running:
            return
        prompt_text = self.chat_input.toPlainText().strip()
        if not prompt_text:
            return
        model_id = str(self.chat_model_combo.currentData() or self.chat_model_combo.currentText()).strip()
        if not model_id:
            self.chat_status.setText("モデルを選択または指定してください")
            return

        self.chat_input.clear()
        if self.chat_history:
            self._append_chat_text("\n\n")
        self._append_chat_text(f"あなた:\n{prompt_text}\n\nアシスタント:\n")

        self.chat_history.append({"role": "user", "content": prompt_text})
        msgs: list[dict[str, str]] = []
        sys_text = self.chat_system_edit.text().strip()
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})
        msgs.extend(self.chat_history)

        backend = self._chat_backend()
        temp = float(self.chat_temp_spin.value())
        max_tok = int(self.chat_max_tok_spin.value())
        cancel_ev = threading.Event()
        self.chat_cancel = cancel_ev
        self.chat_running = True
        self._chat_reply_buf = []

        self.chat_send_btn.setEnabled(False)
        self.chat_stop_btn.setEnabled(True)
        self.chat_clear_btn.setEnabled(False)
        self.chat_mlx_radio.setEnabled(False)
        self.chat_ollama_radio.setEnabled(False)

        def work() -> None:
            try:
                if backend == "mlx":
                    metrics = self.chat_session.stream_chat(
                        Path(model_id),
                        msgs,
                        temperature=temp,
                        max_tokens=max_tok,
                        cancel_event=cancel_ev,
                        on_chunk=self.chat_sig.chunk.emit,
                        on_status=self.chat_sig.status.emit,
                    )
                else:
                    metrics = ce.stream_ollama_chat(
                        model_id,
                        msgs,
                        temperature=temp,
                        max_tokens=max_tok,
                        cancel_event=cancel_ev,
                        on_chunk=self.chat_sig.chunk.emit,
                        on_status=self.chat_sig.status.emit,
                    )
                self.chat_sig.finished.emit(metrics, None)
            except BaseException as e:
                self.chat_sig.finished.emit(None, e)

        threading.Thread(target=work, daemon=True).start()

    def _stop_chat(self) -> None:
        if self.chat_cancel is not None:
            self.chat_cancel.set()
            self.chat_status.setText("生成を停止中...")

    def _clear_chat(self) -> None:
        if self.chat_running:
            return
        self.chat_history.clear()
        self.chat_view.clear()
        self.chat_status.setText("会話履歴をクリアしました")

    def _on_chat_chunk(self, chunk: str) -> None:
        self._chat_reply_buf.append(chunk)
        self._append_chat_text(chunk)

    def _on_chat_status(self, msg: str) -> None:
        self.chat_status.setText(msg)

    def _on_chat_finished(self, metrics: object, err: object) -> None:
        was_canceled = bool(self.chat_cancel and self.chat_cancel.is_set())
        self.chat_running = False
        self.chat_cancel = None
        self.chat_send_btn.setEnabled(True)
        self.chat_stop_btn.setEnabled(False)
        self.chat_clear_btn.setEnabled(True)
        self.chat_mlx_radio.setEnabled(tools.is_apple_silicon() and tools.mlx_lm_available())
        self.chat_ollama_radio.setEnabled(True)

        reply = "".join(self._chat_reply_buf)
        if reply:
            self.chat_history.append({"role": "assistant", "content": reply})
        elif err is not None:
            # Remove unpaired user message if nothing was generated on error
            if self.chat_history and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()

        if err is not None:
            self._append_chat_text(f"\n[エラー] {err}")
            self.chat_status.setText(f"エラー: {err}")
        elif was_canceled:
            self._append_chat_text("\n[停止]")
            self.chat_status.setText("生成を停止しました")
        elif isinstance(metrics, ce.ChatMetrics):
            tps_str = f" ({metrics.tps:.1f} tok/s)" if metrics.tps else ""
            self.chat_status.setText(f"生成完了: {metrics.tokens} tokens{tps_str} / {metrics.elapsed:.1f}s")
        else:
            self.chat_status.setText("完了")


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
