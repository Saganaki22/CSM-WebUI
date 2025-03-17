@echo off
setlocal enabledelayedexpansion

echo ===============================================
echo === Hugging Face Login Helper Script        ===
echo ===============================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    echo Please make sure .venv directory exists in the current folder.
    pause
    exit /b 1
)
echo Virtual environment activated successfully.
echo.

echo ===============================================
echo === HUGGING FACE LOGIN                     ===
echo ===============================================
echo.
echo You will be prompted to log in to your Hugging Face account.
echo This token will be stored for future use by other scripts.
echo.
echo If you don't have an account, visit https://huggingface.co/join
echo.

set /p login_choice="Do you want to log in to Hugging Face now? [Y/n]: "
if /I "%login_choice%"=="n" (
    echo Skipping Hugging Face login.
) else (
    echo.
    echo Attempting to log in to Hugging Face...
    huggingface-cli login
    if %errorlevel% neq 0 (
        echo [WARNING] Hugging Face login failed.
    ) else (
        echo [SUCCESS] Successfully logged in to Hugging Face.
    )
)

echo.
echo Script completed.
echo You can now close this window or press any key to exit.
pause