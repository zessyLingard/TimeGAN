@echo off
REM ============================================================
REM CTC-GAN Offline Decoder Script (Batch file version)
REM Run this AFTER receiver has logged all messages
REM ============================================================

echo ==============================================
echo CTC-GAN DECODER - Decoding all logged messages
echo ==============================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Check if results folder exists
if not exist "results" (
    echo ERROR: results/ folder not found!
    echo Run the receiver first to collect IPD logs.
    pause
    exit /b 1
)

REM Decode all files
python receiver_aes.py

echo.
echo ==============================================
echo DECODING COMPLETE
echo ==============================================
echo Decoded files saved to: decoded/
pause
