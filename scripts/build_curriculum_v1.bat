@echo off
setlocal

cd /d "%~dp0\.."

set "CURRICULUM_ROOT=datasets\curricula\oscillator_curriculum_v1"
set "BASE_DATASET=%CURRICULUM_ROOT%\00_clean_varcount_1024x512_1k"
set "LIGHT_DATASET=%CURRICULUM_ROOT%\10_realish_light_1024x512_1k"
set "MEDIUM_DATASET=%CURRICULUM_ROOT%\20_realish_medium_1024x512_1k"
set "HEAVY_DATASET=%CURRICULUM_ROOT%\30_realish_heavy_1024x512_1k"
set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Missing "%PYTHON_EXE%". Create the virtual environment first.
  exit /b 1
)

echo [1/5] Building C++ tools...
wsl bash -lc "cd /mnt/c/Users/Brandon/Documents/wip/SoundLearner && meson compile -C build"
if errorlevel 1 exit /b 1

if not exist "%CURRICULUM_ROOT%\README.md" (
  echo Missing curriculum layout under "%CURRICULUM_ROOT%".
  exit /b 1
)

if not exist "%BASE_DATASET%\metadata" (
  echo [2/5] Building clean curriculum dataset...
  wsl bash -lc "cd /mnt/c/Users/Brandon/Documents/wip/SoundLearner/datasets/curricula/oscillator_curriculum_v1 && mkdir -p 00_clean_varcount_1024x512_1k && cd 00_clean_varcount_1024x512_1k && /mnt/c/Users/Brandon/Documents/wip/SoundLearner/build/dataset_builder/dataset_builder -n 1000 -t 5 --min-instrument-size 8 --max-instrument-size 64 --min-uncoupled-oscilators 0 --max-uncoupled-oscilators 12 --freq-bins 1024 --time-frames 512 --fft-size-multiplier 4"
  if errorlevel 1 exit /b 1
) else (
  echo [2/5] Clean curriculum dataset already exists, skipping.
)

if not exist "%LIGHT_DATASET%\metadata" (
  echo [3/5] Building light realish curriculum...
  "%PYTHON_EXE%" -m deep_trainer.dataset_augmentor --input-root "%BASE_DATASET%" --output-root "%LIGHT_DATASET%" --variants-per-input 1 --tool-mode wsl --skip-previews --gain-db-min -2 --gain-db-max 2 --snr-db-min 34 --snr-db-max 44 --low-shelf-db-min -2 --low-shelf-db-max 2 --high-shelf-db-min -2 --high-shelf-db-max 2 --reverb-mix-max 0.04 --saturation-max 0.05
  if errorlevel 1 exit /b 1
) else (
  echo [3/5] Light realish curriculum already exists, skipping.
)

if not exist "%MEDIUM_DATASET%\metadata" (
  echo [4/5] Building medium realish curriculum...
  "%PYTHON_EXE%" -m deep_trainer.dataset_augmentor --input-root "%BASE_DATASET%" --output-root "%MEDIUM_DATASET%" --variants-per-input 1 --tool-mode wsl --skip-previews --gain-db-min -4 --gain-db-max 4 --snr-db-min 26 --snr-db-max 38 --low-shelf-db-min -4 --low-shelf-db-max 4 --high-shelf-db-min -4 --high-shelf-db-max 4 --reverb-mix-max 0.10 --saturation-max 0.10
  if errorlevel 1 exit /b 1
) else (
  echo [4/5] Medium realish curriculum already exists, skipping.
)

if not exist "%HEAVY_DATASET%\metadata" (
  echo [5/5] Building heavy realish curriculum...
  "%PYTHON_EXE%" -m deep_trainer.dataset_augmentor --input-root "%BASE_DATASET%" --output-root "%HEAVY_DATASET%" --variants-per-input 1 --tool-mode wsl --skip-previews --gain-db-min -6 --gain-db-max 6 --snr-db-min 18 --snr-db-max 30 --low-shelf-db-min -6 --low-shelf-db-max 6 --high-shelf-db-min -6 --high-shelf-db-max 6 --reverb-mix-max 0.18 --saturation-max 0.18
  if errorlevel 1 exit /b 1
) else (
  echo [5/5] Heavy realish curriculum already exists, skipping.
)

echo Curriculum v1 dataset ladder is ready.
exit /b 0
