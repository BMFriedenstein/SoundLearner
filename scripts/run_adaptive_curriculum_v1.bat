@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Missing "%PYTHON_EXE%". Create the virtual environment first.
  exit /b 1
)

set "EXTRA_ARGS=%*"

"%PYTHON_EXE%" -m deep_trainer.adaptive_curriculum --config deep_trainer\configs\adaptive_curriculum_v1.toml %EXTRA_ARGS%
exit /b %ERRORLEVEL%
