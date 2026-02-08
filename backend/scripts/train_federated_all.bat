@echo off
REM ========================================
REM Federated Learning Training Script
REM ========================================

echo ========================================
echo FARMFEDERATE COMPLETE TRAINING
echo ========================================
echo.

cd backend

echo [1/2] Training all models (LLM, ViT, VLM)...
echo.
python federated_complete_training.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo [2/2] Generating comprehensive plots...
echo ========================================
echo.

python comprehensive_plotting.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Plotting failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo TRAINING COMPLETE!
echo ========================================
echo.
echo Results saved to: results/
echo Plots saved to: plots/
echo Checkpoints saved to: checkpoints/
echo.
echo Open plots/ folder to view all comparison plots
echo.

pause
