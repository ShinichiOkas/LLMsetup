# LLMQuant

LLM を llama.cpp で GGUF に量子化し、そのまま Ollama に登録する GUI ツール。
Hugging Face からのモデルダウンロード、およびダウンロード済みの Ollama モデルから Modelfile を作成・カスタマイズして新しいモデルを作成（`ollama create`）する機能も備えています。

ベース（HF 形式のモデルフォルダ、または `.gguf`）を指定すると、作れる量子化の種類が
サイズの目安・GPU に載るか・品質・注意と一緒に一覧で出る。印を付けて「量子化を実行」すると
GGUF ができ、望めば `ollama create` まで済む。

さらに、以下の機能も備えています:
- **HF ダウンロード**: Hugging Face の Repo ID や URL からモデルリポジトリ全体（HF形式）や単体 `.gguf` ファイルを直接ダウンロードし、ワンクリックで量子化ベースに設定できます。
- **Ollama モデル作成**: ダウンロード済みモデルから Modelfile を自動取得・直接編集、システムプロンプトやパラメータ（temperature, num_ctx 等。モデル情報の既定値・範囲付き）を設定して新規モデルを作成できます。

## 使い方

```
install.bat      :: 初回と git pull の後。venv・依存・llama.cpp（固定版）をそろえる
run.bat          :: GUI を開く（run.bat "D:\models\Qwen3-8B" でベースを指定して開く）
```

CLI でも同じことができる（GUI と同じ芯を使う）:

```
.venv\Scripts\python -m llmquant hf-inspect "Qwen/Qwen2.5-Coder-7B-Instruct"
.venv\Scripts\python -m llmquant hf-download "Qwen/Qwen2.5-Coder-7B-Instruct"
.venv\Scripts\python -m llmquant inspect "D:\models\Qwen3-8B"
.venv\Scripts\python -m llmquant run "D:\models\Qwen3-8B" -t Q4_K_M -t Q6_K --ollama
.venv\Scripts\python -m llmquant ollama-list
.venv\Scripts\python -m llmquant ollama-create qwen2.5:7b my-qwen:latest --system "あなたはアシスタントです" --param temperature=0.7
.venv\Scripts\python -m llmquant doctor
```

## 入力と選択肢

| ベース | 処理 |
|---|---|
| HF 形式のフォルダ（`config.json` + `*.safetensors` + トークナイザ） | `convert_hf_to_gguf.py` で BF16/F16 の中間 GGUF → `llama-quantize` |
| `.gguf`（F16/BF16/F32） | `llama-quantize` のみ |
| `.gguf`（量子化済み） | より小さい型への再量子化だけ可（品質が二重に落ちる旨を表示） |
| LoRA・画像生成モデル・mmproj・単体 safetensors | 「作れません」と理由を表示 |

作れない種類は理由つきで灰色に出る: 元より大きくなるだけの型、imatrix が要る型
（IQ1/IQ2 系・IQ3_XXS・IQ3_XS・Q2_K_S。imatrix ファイルを指定すれば解禁）、
MoE 専用・三値モデル専用の型、空き容量の足りない型。
おすすめは「GPU に載る中で最も品質の高いもの」（Q8_0 → Q6_K → Q5_K_M → Q4_K_M、下限 Q4_K_M）。

GGUF のベースでは、`llama-quantize --dry-run` で正確なサイズを後から差し替える。

## 安全

- 出力は `<名前>.gguf.part` に書き、成功したら改名する。取り消し・失敗時は途中のファイルと中間ファイルを消す
- 同名のファイルがあれば、「上書き」に印を付けるまで実行しない
- 作業フォルダに既にある中間ファイルは再利用し、消さない
- できた GGUF は読み戻して形式を確かめる。Ollama 登録後はチャットテンプレートが素通しでないかを確かめる

## 固定している版

- llama.cpp `v0.9.0`（ビルド `b12110`・コミット `cd959bae`）: `vendor/llama.cpp`（submodule）と
  Windows CPU 版バイナリ（`vendor/llama-bin`、sha256 照合）
- 変換器の Python 依存は、その版の `requirements/requirements-convert_hf_to_gguf.txt` をそのまま使う

## 受入

```
.venv\Scripts\python -X utf8 scripts\acceptance_e2e.py
```

小さな実モデル（SmolLM2-135M-Instruct）で HF → GGUF → Ollama → 応答まで通す
（GUI の入口から 1 回、API から 1 回、再量子化・imatrix 拒否・取り消しも確かめる）。
登録したモデルは最後に消す。
