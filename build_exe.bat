@echo off
setlocal
cd /d "%~dp0"

REM ===========================================================================
REM  Build a double-click CBCT_App.exe launcher with PyInstaller.
REM  Run this INSIDE the environment that has the app's packages installed.
REM  The .exe is a thin launcher: it starts Streamlit using that environment's
REM  Python. It does NOT bundle torch/nnU-Net (too large); those stay in the env.
REM ===========================================================================

echo Installing PyInstaller (into the current environment)...
pip install pyinstaller || (echo Failed to install pyinstaller & pause & exit /b 1)

echo.
echo Building CBCT_App.exe ...
pyinstaller --onefile --console --name CBCT_App launch.py || (echo Build failed & pause & exit /b 1)

echo.
echo Done.  ->  dist\CBCT_App.exe
echo Copy CBCT_App.exe into this project folder (next to app.py, cbct_pipeline\,
echo and the config files), then double-click it. Keep using it from a shell/env
echo where "python" resolves to the environment with the requirements installed.
pause
