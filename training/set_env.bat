@echo off
REM ===========================================================================
REM  Sets the three nnU-Net environment variables for THIS terminal session.
REM
REM  USAGE:  run it from your working terminal by typing:   set_env.bat
REM  Do NOT double-click it (that would set them in a window that closes at once).
REM
REM  Edit BASE below if you want the data stored somewhere other than this folder.
REM ===========================================================================

set "BASE=%~dp0nnUNet"

set "nnUNet_raw=%BASE%\nnUNet_raw"
set "nnUNet_preprocessed=%BASE%\nnUNet_preprocessed"
set "nnUNet_results=%BASE%\nnUNet_results"

if not exist "%nnUNet_raw%" mkdir "%nnUNet_raw%"
if not exist "%nnUNet_preprocessed%" mkdir "%nnUNet_preprocessed%"
if not exist "%nnUNet_results%" mkdir "%nnUNet_results%"

echo nnUNet_raw           = %nnUNet_raw%
echo nnUNet_preprocessed  = %nnUNet_preprocessed%
echo nnUNet_results       = %nnUNet_results%
echo.
echo These are set for THIS terminal only. Re-run set_env.bat in each new terminal,
echo or make them permanent with setx (see TRAINING.md, Windows section).
