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

set "PREV_CHECKPOINT="

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
set "RUN_DIR=%~4"
set "EVAL_DIR=%~5"

echo ============================================================
echo Running complexity stage: %NAME%
echo Config: %CONFIG%
if defined PREV_CHECKPOINT (
  echo Init checkpoint: %PREV_CHECKPOINT%
)
echo ============================================================

if defined PREV_CHECKPOINT (
  "%PYTHON_EXE%" -m deep_trainer.train --config "%CONFIG%" --init-checkpoint "%PREV_CHECKPOINT%"
) else (
  "%PYTHON_EXE%" -m deep_trainer.train --config "%CONFIG%"
)
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m deep_trainer.evaluate --checkpoint "%RUN_DIR%\best.pt" --manifest sounds\manifest.csv --output-dir "%EVAL_DIR%" --freq-bins 1024 --time-frames 512 --device cpu
if errorlevel 1 exit /b 1

"%PYTHON_EXE%" -m deep_trainer.analyze_parameter_space --dataset-root "%DATASET_ROOT%" --evaluation-root "%EVAL_DIR%" --output-dir "%EVAL_DIR%\analysis"
if errorlevel 1 exit /b 1

set "PREV_CHECKPOINT=%RUN_DIR%\best.pt"
goto :eof
