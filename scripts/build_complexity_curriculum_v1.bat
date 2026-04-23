@echo off
setlocal

cd /d "%~dp0\.."

set "CURRICULUM_ROOT=datasets\curricula\complexity_curriculum_v1"
set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Missing "%PYTHON_EXE%". Create the virtual environment first.
  exit /b 1
)

echo [1/8] Building C++ tools...
wsl bash -lc "cd /mnt/c/Users/Brandon/Documents/wip/SoundLearner && meson compile -C build"
if errorlevel 1 exit /b 1

if not exist "%CURRICULUM_ROOT%\README.md" (
  echo Missing curriculum layout under "%CURRICULUM_ROOT%".
  exit /b 1
)

call :build_stage [2/8] c1_1to3_clean_1024x512_1k 1 3 0 0
if errorlevel 1 exit /b 1
call :build_stage [3/8] c2_1to5_clean_1024x512_1k 1 5 0 0
if errorlevel 1 exit /b 1
call :build_stage [4/8] c3_1to8_clean_1024x512_1k 1 8 0 0
if errorlevel 1 exit /b 1
call :build_stage [5/8] c4_1to12_plus2_uncoupled_1024x512_1k 1 12 0 2
if errorlevel 1 exit /b 1
call :build_stage [6/8] c5_1to20_plus4_uncoupled_1024x512_1k 1 20 0 4
if errorlevel 1 exit /b 1
call :build_stage [7/8] c6_1to32_plus8_uncoupled_1024x512_1k 1 32 0 8
if errorlevel 1 exit /b 1
call :build_stage [8/8] c7_1to64_plus12_uncoupled_1024x512_1k 1 64 0 12
if errorlevel 1 exit /b 1

echo Complexity curriculum v1 datasets are ready.
exit /b 0

:build_stage
set "STEP=%~1"
set "STAGE_DIR=%~2"
set "MIN_COUPLED=%~3"
set "MAX_COUPLED=%~4"
set "MIN_UNCOUPLED=%~5"
set "MAX_UNCOUPLED=%~6"
set "FULL_STAGE=%CURRICULUM_ROOT%\%STAGE_DIR%"

if not exist "%FULL_STAGE%\metadata" (
  echo %STEP% Building %STAGE_DIR%...
  wsl bash -lc "cd /mnt/c/Users/Brandon/Documents/wip/SoundLearner/datasets/curricula/complexity_curriculum_v1 && mkdir -p %STAGE_DIR% && cd %STAGE_DIR% && /mnt/c/Users/Brandon/Documents/wip/SoundLearner/build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size %MIN_COUPLED% --max-instrument-size %MAX_COUPLED% --min-uncoupled-oscilators %MIN_UNCOUPLED% --max-uncoupled-oscilators %MAX_UNCOUPLED% --freq-bins 1024 --time-frames 512 --fft-size-multiplier 4"
  if errorlevel 1 exit /b 1
) else (
  echo %STEP% %STAGE_DIR% already exists, skipping.
)

goto :eof
