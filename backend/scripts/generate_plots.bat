@echo off
REM ========================================
REM Generate Plots from Existing Results
REM ========================================

echo ========================================
echo GENERATING COMPREHENSIVE PLOTS
echo ========================================
echo.
echo This will create 15+ comparison plots
echo from your existing training results.
echo.

cd backend

python comprehensive_plotting.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Plot generation failed!
    echo Make sure you have trained models first.
    echo Run: train_federated_all.bat
    pause
    exit /b 1
)

echo.
echo ========================================
echo PLOTS GENERATED SUCCESSFULLY!
echo ========================================
echo.
echo Plots saved to: plots/
echo.
echo Opening plots folder...
echo.

start ..\plots

pause
