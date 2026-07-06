@echo off
REM ===========================================================================
REM  Trains all 5 folds SEQUENTIALLY (each ~a day). Run from your activated conda
REM  env terminal:   training\train_folds.bat
REM  It sets the nnUNet_* env vars for you (via set_env.bat).
REM
REM  If your dataset was planned with the DEFAULT planner instead of ResEnc-L,
REM  change PLANS to nnUNetPlans (or remove the -p flag below).
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
call set_env.bat

set DATASET=201
set CONFIG=3d_fullres
set PLANS=nnUNetResEncUNetLPlans

for %%f in (0 1 2 3 4) do (
    echo.
    echo ================= Training fold %%f  (plans %PLANS%) =================
    nnUNetv2_train %DATASET% %CONFIG% %%f -p %PLANS%
    if errorlevel 1 (
        echo.
        echo Fold %%f FAILED - stopping. See the messages above.
        pause
        exit /b 1
    )
)

echo.
echo All folds complete. Find the best configuration with:
echo    nnUNetv2_find_best_configuration %DATASET% -p %PLANS%
pause
