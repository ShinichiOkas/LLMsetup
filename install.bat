@echo off
setlocal
rem LLMQuant installer. After clone or "git pull", run this and nothing else.
rem Always operate on the directory this script lives in, not the caller's
rem current directory (a caller-cwd venv got the installs once in Mnemo).
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
echo === LLMQuant install ===

rem ---------------------------------------------------------------------------
echo [1/8] Checking the system python...
where python >nul 2>&1
if errorlevel 1 goto no_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 goto old_python

rem ---------------------------------------------------------------------------
rem Refuse to run while a python from this venv is running (the GUI or a
rem quantize job): pip cannot replace files a running process holds open and
rem may still exit 0. Never kills anything - the user may be mid-work.
echo [2/8] Checking whether LLMQuant (this venv) is running...
python scripts\check_running.py "%~dp0.venv"
if errorlevel 2 goto running_unknown
if errorlevel 1 goto running
goto running_done
:running
echo.
echo   [ERROR] LLMQuant is running from this venv (process list above).
echo   [ERROR] Close it first, then run install.bat again. Nothing was killed.
echo.
exit /b 1
:running_unknown
echo   [WARN] Could not check running processes. Continuing.
:running_done

rem ---------------------------------------------------------------------------
echo [3/8] Cleaning up stale install artifacts...
if exist ".venv\Lib\site-packages" (
    for /d %%D in (".venv\Lib\site-packages\~*") do rmdir /s /q "%%D" 2>nul
    del /s /q ".venv\*.deleteme" >nul 2>&1
)

rem ---------------------------------------------------------------------------
rem "git pull" does not update submodules; an empty vendor\llama.cpp would
rem silently leave the converter missing.
echo [4/8] Fetching submodules (llama.cpp, pinned)...
git submodule sync --recursive
if errorlevel 1 goto submodule_failed
rem core.longpaths: llama.cpp has paths beyond MAX_PATH when cloned into a deep folder.
git -c core.longpaths=true submodule update --init --recursive --depth 1
if errorlevel 1 goto submodule_failed
if not exist "vendor\llama.cpp\convert_hf_to_gguf.py" goto submodule_empty
if not exist "vendor\llama.cpp\gguf-py\pyproject.toml" goto submodule_empty

rem ---------------------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [5/8] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 goto venv_failed
) else (
    echo [5/8] Using existing virtual environment .venv
)
set "VPY=%~dp0.venv\Scripts\python.exe"

rem ---------------------------------------------------------------------------
rem The converter's requirements come from the pinned llama.cpp commit as-is
rem (torch CPU wheel index, transformers, sentencepiece, protobuf, numpy).
rem gguf is then installed editable from the same commit, replacing PyPI's.
echo [6/8] Installing dependencies...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto pip_failed
"%VPY%" -m pip install -r "vendor\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt"
if errorlevel 1 goto pip_failed
"%VPY%" -m pip install -e "vendor\llama.cpp\gguf-py"
if errorlevel 1 goto pip_failed
"%VPY%" -m pip install -e ".[dev]"
if errorlevel 1 goto pip_failed

rem ---------------------------------------------------------------------------
echo [7/8] Fetching llama.cpp binaries (pinned, sha256-checked)...
"%VPY%" scripts\fetch_llama_bin.py
if errorlevel 1 goto fetch_failed

rem ---------------------------------------------------------------------------
echo [8/8] Self-check...
"%VPY%" scripts\selfcheck.py
if errorlevel 1 goto selfcheck_failed

echo.
echo === Done. Start the GUI with run.bat ===
endlocal
exit /b 0

:no_python
echo   [ERROR] python not found on PATH. Install Python 3.10+ first.
exit /b 1
:old_python
echo   [ERROR] Python 3.10 or newer is required.
exit /b 1
:submodule_failed
echo   [ERROR] git submodule update failed (network? git installed?).
exit /b 1
:submodule_empty
echo   [ERROR] vendor\llama.cpp is empty or incomplete after submodule update.
exit /b 1
:venv_failed
echo   [ERROR] Could not create .venv
exit /b 1
:pip_failed
echo   [ERROR] pip install failed (see above). The environment is NOT ready.
exit /b 1
:fetch_failed
echo   [ERROR] Could not fetch the llama.cpp binaries (see above).
exit /b 1
:selfcheck_failed
echo   [ERROR] Self-check failed (see above). The environment is NOT ready.
exit /b 1
