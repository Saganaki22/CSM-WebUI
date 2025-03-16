@echo off
setlocal enabledelayedexpansion

REM ===========================================
REM === CSM-WebUI Windows Setup Script v1.0 ===
REM === VERBOSE MODE                        ===
REM ===========================================

echo ==========================================
echo === CSM-WebUI Windows Setup Script v1.0 ===
echo === VERBOSE MODE                        ===
echo ==========================================
echo.

REM --- PYTHON VERIFICATION ---
echo [1/8] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is required but not found.
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version (should be 3.10+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo [SUCCESS] Found Python version %pyver%
echo.

REM --- SETUP ENVIRONMENT ---
echo [2/8] Setting up virtual environment...
cd /d %~dp0
echo Current directory: %CD%

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM --- INSTALL BASE PACKAGES ---
echo.
echo [3/8] Installing base packages...
echo Upgrading pip with verbose output...
python -m pip install --upgrade pip --verbose
pip install --upgrade wheel setuptools --verbose
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install base packages.
    pause
    exit /b 1
)
echo [SUCCESS] Base packages installed.
echo.

REM --- VISUAL C++ REDISTRIBUTABLE CHECK ---
echo [4/8] Checking for required Visual C++ Redistributable...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version
if %errorlevel% neq 0 (
    echo Visual C++ Redistributable might be missing, which is required for PyTorch.
    echo Please download and install the Visual C++ Redistributable from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    set /p install_vcredist="Would you like to download and install it now? [Y/n]: "
    if /I "!install_vcredist!"=="n" (
        echo Skipping Visual C++ Redistributable installation. This might cause issues with PyTorch.
    ) else (
        echo Downloading Visual C++ Redistributable...
        powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe'}"
        echo Installing Visual C++ Redistributable...
        start /wait vc_redist.x64.exe /quiet /norestart
        echo Visual C++ Redistributable installation complete.
    )
) else (
    echo [SUCCESS] Visual C++ Redistributable is already installed.
)
echo.

REM --- INSTALL PYTORCH WITH CUDA (FIRST TO AVOID CONFLICTS) ---
echo [5/8] Installing PyTorch with CUDA support...
echo ================================================
echo INSTALLING PYTORCH WITH CUDA 12.4
echo ================================================
echo This may take several minutes...
echo.
pip install torch==2.4.0 torchvision torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --verbose
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

REM Verify CUDA support
echo.
echo Verifying PyTorch CUDA installation...
python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA available: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to verify PyTorch installation.
    pause
    exit /b 1
)
echo [SUCCESS] PyTorch with CUDA installed successfully.
echo.

REM --- INSTALL TRITON-WINDOWS ---
echo [6/8] Installing Triton for Windows...

REM Check for regular triton package and remove if found
pip list | findstr /c:"triton " /c:"triton==" 
if %errorlevel% equ 0 (
    echo Removing regular triton package...
    pip uninstall -y triton --verbose
)

REM Install triton-windows
echo Installing triton-windows...
pip install triton-windows --verbose
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install triton-windows. This might cause issues.
) else (
    echo [SUCCESS] Triton for Windows installed.
)
echo.

REM --- INSTALL REQUIRED PACKAGES ---
echo [7/8] Installing packages from requirements.txt...

REM Install specific packages mentioned in the error
echo Installing packages individually with verbose output...
echo.

echo Step 1/6: Installing core transformers packages...
pip install tokenizers==0.21.0 transformers==4.49.0 huggingface_hub==0.28.1 --verbose
if %errorlevel% neq 0 (
    echo [WARNING] Some transformers packages failed to install.
)
echo.

echo Step 2/6: Installing moshi...
pip install moshi==0.2.2 --verbose
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install moshi. Trying to install its dependencies separately...
    pip install einops sounddevice sphn --verbose
)
echo.

echo Step 3/6: Installing torchtune and torchao...
pip install torchtune==0.4.0 torchao==0.9.0 --verbose
if %errorlevel% neq 0 (
    echo [WARNING] Some torch extension packages failed to install.
)
echo.

echo Step 4/6: Installing gradio...
pip install gradio --verbose
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install gradio.
    pause
    exit /b 1
)
echo.

echo Step 5/7: Installing silentcipher from GitHub...
pip install git+https://github.com/SesameAILabs/silentcipher@master --verbose
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install silentcipher from GitHub.
    echo Trying alternative installation method...
    pip install "silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master" --verbose
    if %errorlevel% neq 0 (
        echo.
        echo [WARNING] Failed to install silentcipher. Trying one more approach...
        pip install -e git+https://github.com/SesameAILabs/silentcipher@master#egg=silentcipher --verbose
        if %errorlevel% neq 0 (
            echo [ERROR] All attempts to install silentcipher failed. This package is required.
            pause
            exit /b 1
        )
    )
)

echo.
echo Step 6/7: Installing bitsandbytes for GPU acceleration...
echo This package is required for model quantization and speech generation.

REM First try the standard package
pip install bitsandbytes --verbose
if %errorlevel% neq 0 (
    echo [WARNING] Standard bitsandbytes installation failed. Trying Windows-specific version...
    
    REM Try Windows-specific version
    pip install bitsandbytes-windows --verbose
    if %errorlevel% neq 0 (
        echo [WARNING] Windows-specific version failed. Trying alternative source...
        
        REM Try installing from wheel
        pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl --verbose
        if %errorlevel% neq 0 (
            echo [ERROR] All attempts to install bitsandbytes failed.
            echo This may cause issues with speech generation.
        ) else {
            echo [SUCCESS] Installed bitsandbytes from alternative source.
        }
    } else {
        echo [SUCCESS] Installed Windows-specific bitsandbytes package.
    }
) else {
    echo [SUCCESS] Installed standard bitsandbytes package.
}
echo.

echo Step 7/7: Installing remaining requirements from file...
if exist requirements.txt (
    echo Installing any remaining packages from requirements.txt...
    pip install -r requirements.txt --verbose
    if %errorlevel% neq 0 (
        echo [WARNING] Some packages from requirements.txt could not be installed.
        echo This may not be a problem if they were installed in previous steps.
    )
) else (
    echo requirements.txt not found, skipping additional packages.
)
echo.

echo [SUCCESS] Package installation complete.
echo.

REM Create run script
echo Creating run_gradio.bat script...
if not exist run_gradio.bat (
    (
        echo @echo off
        echo echo Starting CSM-WebUI with Gradio interface...
        echo cd /d "%%~dp0"
        echo call .venv\Scripts\activate.bat
        echo python win-gradio.py
        echo pause
    ) > run_gradio.bat
    
    if exist run_gradio.bat (
        echo [SUCCESS] Successfully created run_gradio.bat
    ) else (
        echo [WARNING] Failed to create run_gradio.bat
        echo Please manually create a file named 'run_gradio.bat' with the following content:
        echo ----------------------------------------------------------
        echo @echo off
        echo cd /d "%%~dp0"
        echo call .venv\Scripts\activate.bat
        echo python win-gradio.py
        echo pause
        echo ----------------------------------------------------------
    )
) else (
    echo run_gradio.bat already exists, skipping creation...
)

REM --- VERIFICATION ---
echo.
echo ==================== FINAL VERIFICATION ====================
echo Verifying PyTorch CUDA support...
python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"

echo.
echo Verifying critical packages:
echo.
echo Checking gradio...
python -c "import gradio; print(f'gradio {gradio.__version__}')" 2>nul || echo "FAILED: gradio not found!"

echo.
echo Checking torch...
python -c "import torch; print(f'torch {torch.__version__}')" 2>nul || echo "FAILED: torch not found!"
python -c "import torchaudio; print(f'torchaudio {torchaudio.__version__}')" 2>nul || echo "FAILED: torchaudio not found!"

echo.
echo Checking silentcipher...
python -c "try: import silentcipher; print('silentcipher found'); \nexcept: print('FAILED: silentcipher not found!')"

echo.
echo Checking transformers...
python -c "import transformers; print(f'transformers {transformers.__version__}')" 2>nul || echo "FAILED: transformers not found!"

echo.
echo Checking huggingface_hub...
python -c "import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')" 2>nul || echo "FAILED: huggingface_hub not found!"

echo.
echo Checking bitsandbytes...
python -c "try: import bitsandbytes as bnb; print(f'bitsandbytes found - version: {bnb.__version__}'); \nexcept ImportError: print('FAILED: bitsandbytes not found!'); \nexcept AttributeError: print('bitsandbytes found (version unavailable)')"

echo.
echo Checking triton packages:
pip list | findstr "triton"
echo ===========================================================

echo.
echo Setup complete!
if not exist models\csm-1b\model.safetensors (
    echo.
    echo [WARNING] Model file not found. Please download it manually from:
    echo https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors
    echo And place it in the models directory along with the config.json file
    echo.
)

echo.
echo To run the application, simply double-click the run_gradio.bat file 
echo or run it from the command line.
echo.

REM Final pause
echo Press any key to exit...
pause
