@echo off
REM ============================================================
REM CTC-GAN Receiver Script (Batch file version)
REM Run this FIRST in a separate terminal, then run run_sender.bat
REM ============================================================

echo ==============================================
echo CTC-GAN RECEIVER - Waiting for 100 messages
echo ==============================================
echo.
echo Make sure to run this BEFORE starting the sender!
echo Press Ctrl+C after all messages are received.
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Create results directory
if not exist "results" mkdir results

REM Run receiver for 100 messages
python receiver_log.py --port 3334 --count 100 --timeout 600

echo.
echo ==============================================
echo RECEIVING COMPLETE
echo ==============================================
echo Logs saved to: results/
echo.
echo Now run: python receiver_aes.py
echo to decode all messages offline.
pause
