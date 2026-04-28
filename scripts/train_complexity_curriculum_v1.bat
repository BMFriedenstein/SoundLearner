@echo off
setlocal

cd /d "%~dp0\.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Missing "%PYTHON_EXE%". Create the virtual environment first.
  exit /b 1
)

set "TARGET_STAGE=%~1"
if "%TARGET_STAGE%"=="" set "TARGET_STAGE=c7"
set "TRAIN_MODE=%~2"
if "%TRAIN_MODE%"=="" set "TRAIN_MODE=parameter"
set "START_STAGE=%~3"
if "%START_STAGE%"=="" set "START_STAGE=c1"

if /i "%TRAIN_MODE%"=="parameter" (
  set "RUN_SUFFIX=_balanced"
  set "EXTRA_TRAIN_ARGS=--activity-loss-weight 3.0 --activity-positive-weight 0.0 --f0-loss-weight 5.0 --crowding-loss-weight 0.002"
) else if /i "%TRAIN_MODE%"=="renderloss" (
  set "RUN_SUFFIX=_renderloss_balanced"
  set "EXTRA_TRAIN_ARGS=--activity-loss-weight 3.0 --activity-positive-weight 0.0 --parameter-loss-weight 5.0 --f0-loss-weight 10.0 --crowding-loss-weight 0.005 --render-loss-weight 20.0 --render-rms-loss-weight 2.5 --render-loss-seconds 1.0 --render-loss-sample-rate 11025"
) else if /i "%TRAIN_MODE%"=="renderloss_light" (
  set "RUN_SUFFIX=_renderloss_light_balanced"
  set "EXTRA_TRAIN_ARGS=--activity-loss-weight 3.0 --activity-positive-weight 0.0 --parameter-loss-weight 5.0 --f0-loss-weight 5.0 --crowding-loss-weight 0.002 --render-loss-weight 10.0 --render-rms-loss-weight 1.0 --render-loss-seconds 0.5 --render-loss-sample-rate 8000"
) else (
  echo Unknown training mode "%TRAIN_MODE%".
  echo Use: parameter ^| renderloss ^| renderloss_light
  exit /b 1
)

set "PREV_CHECKPOINT="
set "IN_ACTIVE_RANGE=0"

call :run_stage c1 deep_trainer\configs\complexity_curriculum_v1_c1.toml datasets\curricula\complexity_curriculum_v1\c1_1to3_clean_1024x512_1k runs\complexity_curriculum_v1_c1_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c1_w128_b8_e20
if errorlevel 1 exit /b 1
if /i "%TARGET_STAGE%"=="c1" goto :done
call :run_stage c2 deep_trainer\configs\complexity_curriculum_v1_c2.toml datasets\curricula\complexity_curriculum_v1\c2_1to5_clean_1024x512_1k runs\complexity_curriculum_v1_c2_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c2_w128_b8_e20
if errorlevel 1 exit /b 1
if /i "%TARGET_STAGE%"=="c2" goto :done
call :run_stage c3 deep_trainer\configs\complexity_curriculum_v1_c3.toml datasets\curricula\complexity_curriculum_v1\c3_1to8_clean_1024x512_1k runs\complexity_curriculum_v1_c3_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c3_w128_b8_e20
if errorlevel 1 exit /b 1
if /i "%TARGET_STAGE%"=="c3" goto :done
call :run_stage c4 deep_trainer\configs\complexity_curriculum_v1_c4.toml datasets\curricula\complexity_curriculum_v1\c4_1to12_plus2_uncoupled_1024x512_1k runs\complexity_curriculum_v1_c4_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c4_w128_b8_e20
if errorlevel 1 exit /b 1
if /i "%TARGET_STAGE%"=="c4" goto :done
call :run_stage c5 deep_trainer\configs\complexity_curriculum_v1_c5.toml datasets\curricula\complexity_curriculum_v1\c5_1to20_plus4_uncoupled_1024x512_1k runs\complexity_curriculum_v1_c5_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c5_w128_b8_e20
if errorlevel 1 exit /b 1
if /i "%TARGET_STAGE%"=="c5" goto :done
call :run_stage c6 deep_trainer\configs\complexity_curriculum_v1_c6.toml datasets\curricula\complexity_curriculum_v1\c6_1to32_plus8_uncoupled_1024x512_1k runs\complexity_curriculum_v1_c6_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c6_w128_b8_e20
if errorlevel 1 exit /b 1
if /i "%TARGET_STAGE%"=="c6" goto :done
call :run_stage c7 deep_trainer\configs\complexity_curriculum_v1_c7.toml datasets\curricula\complexity_curriculum_v1\c7_1to64_plus12_uncoupled_1024x512_1k runs\complexity_curriculum_v1_c7_w128_b8_e20 sounds\eval\complexity_curriculum_v1_c7_w128_b8_e20
if errorlevel 1 exit /b 1

:done
echo Complexity curriculum v1 training ladder complete through %TARGET_STAGE%.
exit /b 0

:run_stage
set "NAME=%~1"
set "CONFIG=%~2"
set "DATASET_ROOT=%~3"
set "BASE_RUN_DIR=%~4"
set "BASE_EVAL_DIR=%~5"
set "RUN_DIR=%BASE_RUN_DIR%%RUN_SUFFIX%"
set "EVAL_DIR=%BASE_EVAL_DIR%%RUN_SUFFIX%"

if /i "%NAME%"=="%START_STAGE%" set "IN_ACTIVE_RANGE=1"
if not "%IN_ACTIVE_RANGE%"=="1" goto :eof

echo ============================================================
echo Running complexity stage: %NAME%
echo Config: %CONFIG%
echo Mode: %TRAIN_MODE%
echo Start stage: %START_STAGE%
echo Run dir: %RUN_DIR%
if defined PREV_CHECKPOINT (
  echo Init checkpoint: %PREV_CHECKPOINT%
)
echo ============================================================

if defined PREV_CHECKPOINT (
  "%PYTHON_EXE%" -m deep_trainer.train --config "%CONFIG%" --output-dir "%RUN_DIR%" --init-checkpoint "%PREV_CHECKPOINT%" %EXTRA_TRAIN_ARGS%
) else (
  "%PYTHON_EXE%" -m deep_trainer.train --config "%CONFIG%" --output-dir "%RUN_DIR%" %EXTRA_TRAIN_ARGS%
)
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m deep_trainer.evaluate --checkpoint "%RUN_DIR%\best.pt" --manifest sounds\manifest.csv --output-dir "%EVAL_DIR%" --freq-bins 1024 --time-frames 512 --device cpu
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m deep_trainer.analyze_parameter_space --dataset-root "%DATASET_ROOT%" --evaluation-root "%EVAL_DIR%" --output-dir "%EVAL_DIR%\analysis"
if errorlevel 1 exit /b 1

set "PREV_CHECKPOINT=%RUN_DIR%\best.pt"
goto :eof
