@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Missing "%PYTHON_EXE%". Create the virtual environment first.
  exit /b 1
)

if "%DATASET_WORKERS%"=="" set "DATASET_WORKERS=8"
if "%EXPECTED_SAMPLES%"=="" set "EXPECTED_SAMPLES=1000"

set "FORCE_FLAG="
if /i "%FORCE_REBUILD%"=="1" set "FORCE_FLAG=--force"

"%PYTHON_EXE%" scripts\build_datasets.py --curriculum complexity_v1 --workers %DATASET_WORKERS% --expected-samples %EXPECTED_SAMPLES% %FORCE_FLAG%
exit /b %ERRORLEVEL%
