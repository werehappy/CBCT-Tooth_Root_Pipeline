@echo off
setlocal
cd /d "%~dp0"

REM ===========================================================================
REM  Double-click launcher for the CBCT app (Windows).
REM  If the app's Python packages live in a named conda environment, put its
REM  name here (e.g. set CONDA_ENV=cbct). Leave blank to use the current python.
REM ===========================================================================
set CONDA_ENV=

if not "%CONDA_ENV%"=="" (
    call conda activate %CONDA_ENV% 2>nul
)

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python was not found on PATH.
    echo Open "Anaconda Prompt", activate the environment where you installed the
    echo requirements, and run:   python launch.py
    echo.
    pause
    exit /b 1
)

python launch.py
if errorlevel 1 pause
