@echo off
REM ============================================================
REM CTC-GAN Sender Script (Batch file version)
REM Run this AFTER starting run_receiver.bat in another terminal
REM ============================================================

echo ==============================================
echo CTC-GAN SENDER - Sending 100 messages
echo ==============================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Create encoded directory for intermediate files
if not exist "encoded" mkdir encoded

set TOTAL_SENT=0

for /L %%i in (1,1,100) do (
    REM Format number with leading zeros
    set "NUM=00%%i"
    setlocal enabledelayedexpansion
    set "NUM=!NUM:~-3!"
    
    set "MSG_FILE=data\msg_!NUM!.txt"
    set "IPD_FILE=encoded\ipd_!NUM!.csv"
    
    if exist "!MSG_FILE!" (
        echo.
        echo [%%i/100] Processing: !MSG_FILE!
        echo ----------------------------------------
        
        echo   [1/2] Encoding...
        python encoder_aes.py --file "!MSG_FILE!" --output "!IPD_FILE!"
        
        if exist "!IPD_FILE!" (
            echo   [2/2] Sending...
            python sender.py --csv "!IPD_FILE!" --ip 127.0.0.1 --port 3334
            set /a TOTAL_SENT+=1
        ) else (
            echo   ERROR: Encoding failed!
        )
        
        REM Small delay between messages
        timeout /t 2 /nobreak >nul
    ) else (
        echo [%%i] SKIP: !MSG_FILE! not found
    )
    endlocal
)

echo.
echo ==============================================
echo SENDING COMPLETE
echo ==============================================
echo.
echo Wait for receiver to finish, then run:
echo   python receiver_aes.py
pause
