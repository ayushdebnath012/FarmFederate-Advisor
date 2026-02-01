@echo off
REM Copy generated plots to paper figures directory
REM Run this after downloading farmfederate_plots.zip from Colab

echo ============================================
echo Copying FarmFederate Plots to Paper
echo ============================================

REM Check if source exists
set SOURCE=%USERPROFILE%\Downloads\farmfederate_plots (1)
if not exist "%SOURCE%" (
    set SOURCE=%USERPROFILE%\Downloads\farmfederate_plots
)

if not exist "%SOURCE%" (
    echo ERROR: Cannot find farmfederate_plots folder in Downloads
    echo Please extract the downloaded zip first.
    pause
    exit /b 1
)

REM Create figures directory
if not exist figures mkdir figures

REM Copy all PNG files
echo Copying plots from: %SOURCE%
copy "%SOURCE%\*.png" figures\ /Y

echo.
echo ============================================
echo Done! Plots copied to paper\figures\
echo ============================================
echo.
dir figures\*.png /b 2>nul | find /c /v ""
echo plot files copied.
echo.
pause
