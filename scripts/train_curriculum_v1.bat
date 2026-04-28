@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Missing "%PYTHON_EXE%". Create the virtual environment first.
  exit /b 1
)

set "STAGE=%~1"
if "%STAGE%"=="" set "STAGE=all"
set "TRAIN_MODE=%~2"
if "%TRAIN_MODE%"=="" set "TRAIN_MODE=parameter"

if /i "%TRAIN_MODE%"=="parameter" (
  set "RUN_SUFFIX="
  set "EXTRA_TRAIN_ARGS="
) else if /i "%TRAIN_MODE%"=="renderloss" (
  set "RUN_SUFFIX=_renderloss"
  set "EXTRA_TRAIN_ARGS=--activity-loss-weight 1.0 --parameter-loss-weight 2.0 --render-loss-weight 100.0 --render-rms-loss-weight 25.0 --render-loss-seconds 1.0 --render-loss-sample-rate 11025"
) else if /i "%TRAIN_MODE%"=="renderloss_light" (
  set "RUN_SUFFIX=_renderloss_light"
  set "EXTRA_TRAIN_ARGS=--activity-loss-weight 1.0 --parameter-loss-weight 2.0 --render-loss-weight 50.0 --render-rms-loss-weight 10.0 --render-loss-seconds 0.5 --render-loss-sample-rate 8000"
) else (
  echo Unknown training mode "%TRAIN_MODE%".
  echo Use: parameter ^| renderloss ^| renderloss_light
  exit /b 1
)

call :run_stage clean deep_trainer\configs\curriculum_v1_clean.toml datasets\curricula\oscillator_curriculum_v1\00_clean_varcount_1024x512_1k runs\curriculum_v1_clean_w96_b8_e40 sounds\eval\curriculum_v1_clean_w96_b8_e40
if errorlevel 1 exit /b 1
call :run_stage light deep_trainer\configs\curriculum_v1_light.toml datasets\curricula\oscillator_curriculum_v1\10_realish_light_1024x512_1k runs\curriculum_v1_light_w96_b8_e40 sounds\eval\curriculum_v1_light_w96_b8_e40
if errorlevel 1 exit /b 1
call :run_stage medium deep_trainer\configs\curriculum_v1_medium.toml datasets\curricula\oscillator_curriculum_v1\20_realish_medium_1024x512_1k runs\curriculum_v1_medium_w96_b8_e40 sounds\eval\curriculum_v1_medium_w96_b8_e40
if errorlevel 1 exit /b 1
call :run_stage heavy deep_trainer\configs\curriculum_v1_heavy.toml datasets\curricula\oscillator_curriculum_v1\30_realish_heavy_1024x512_1k runs\curriculum_v1_heavy_w96_b8_e40 sounds\eval\curriculum_v1_heavy_w96_b8_e40
if errorlevel 1 exit /b 1

echo Curriculum v1 training ladder complete.
exit /b 0

:run_stage
set "NAME=%~1"
set "CONFIG=%~2"
set "DATASET_ROOT=%~3"
set "BASE_RUN_DIR=%~4"
set "BASE_EVAL_DIR=%~5"
set "RUN_DIR=%BASE_RUN_DIR%%RUN_SUFFIX%"
set "EVAL_DIR=%BASE_EVAL_DIR%%RUN_SUFFIX%"

if /i not "%STAGE%"=="all" if /i not "%STAGE%"=="%NAME%" goto :eof

echo ============================================================
echo Running stage: %NAME%
echo Config: %CONFIG%
echo Mode: %TRAIN_MODE%
echo Run dir: %RUN_DIR%
echo ============================================================

"%PYTHON_EXE%" -m deep_trainer.train --config "%CONFIG%" --output-dir "%RUN_DIR%" %EXTRA_TRAIN_ARGS%
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m deep_trainer.evaluate --checkpoint "%RUN_DIR%\best.pt" --manifest sounds\manifest.csv --output-dir "%EVAL_DIR%" --freq-bins 1024 --time-frames 512 --device cpu
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m deep_trainer.analyze_parameter_space --dataset-root "%DATASET_ROOT%" --evaluation-root "%EVAL_DIR%" --output-dir "%EVAL_DIR%\analysis"
if errorlevel 1 exit /b 1

goto :eof
