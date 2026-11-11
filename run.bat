@echo off
setlocal
rem Start the LLMQuant GUI. Extra arguments are passed through
rem (e.g. run.bat "D:\models\Qwen3-8B").
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo .venv is missing. Run install.bat first.
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m llmquant gui %*
endlocal
exit /b 0
