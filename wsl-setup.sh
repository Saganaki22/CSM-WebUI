#!/bin/bash

# Enhanced setup script for CSM-WebUI in WSL environment
# This script ensures all dependencies are properly installed

echo "Setting up CSM-WebUI environment in WSL..."

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is required but not found."
    echo "Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is required but not found."
    echo "Installing Python 3.10..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# Install build essentials and ffmpeg for audio processing
echo "Installing system dependencies..."
sudo apt-get install -y build-essential ffmpeg libsndfile1

# Navigate to the CSM-WebUI directory
cd ~/CSM-WebUI || { echo "Failed to navigate to CSM-WebUI directory"; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment and install requirements
echo "Activating virtual environment and installing dependencies..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install torch and torchaudio first (with CUDA support if available)
echo "Installing PyTorch with CUDA support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA support
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA: {torch.cuda.is_available()}')"
if ! python3 -c "import torch; torch.cuda.is_available() or exit(1)" 2>/dev/null; then
    echo ""
    echo "WARNING: CUDA support is not available. The application will run much slower without GPU acceleration."
    echo "Please make sure you have a compatible NVIDIA GPU and the latest drivers installed."
    echo ""
fi

# Install moshi manually to ensure we get the right version
echo "Installing moshi and dependencies..."
pip install moshi==0.2.2
pip install tqdm numpy scipy

# Install safetensors
pip install safetensors

# Install torchtune
pip install torchtune==0.4.0

# Install silentcipher (for watermarking)
pip install git+https://github.com/SesameAILabs/silentcipher.git

# Install the rest of the requirements
echo "Installing remaining requirements..."
pip install -r requirements.txt

# Install gradio
pip install gradio

# Create directory structure
mkdir -p models/csm-1b models/llama3.2 models/mimi sounds

# Create a run script
cat > run_gradio.sh << 'EOL'
#!/bin/bash
cd ~/CSM-WebUI
source .venv/bin/activate
# Set environment variables for debugging
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Check installed packages
pip list | grep -E "moshi|torch|gradio|transformers|safetensors|numpy"
# Run the application
python wsl-gradio.py
EOL

# Make the run script executable
chmod +x run_gradio.sh

echo ""
echo "================================================"
echo "MODEL FILES"
echo "================================================"
echo "You need to download the following model files manually:"
echo ""
echo "1. CSM-1B Model:"
echo "   - https://huggingface.co/drbaph/CSM-1B/tree/main"
echo "   - Save as: models/csm-1b/model.safetensors & config.json"
echo ""
echo "2. Llama-3.2-1B Tokenizer:"
echo "   - https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main"
echo "   - Save tokenizer files to: models/llama3.2/**"
echo ""
echo "3. mimi model, config & preprocessor"
echo "   - https://huggingface.co/kyutai/mimi/tree/main"
echo "   - Save tokenizer files to: models/mimi/**"
echo "Please ensure these files are in the correct locations before running the application."
echo ""

echo ""
echo "==================== FINAL VERIFICATION ===================="
echo "Verifying critical packages:"
python3 -c "import numpy; print(f'numpy {numpy.__version__}')" 2>/dev/null || echo "FAILED: numpy not found!"
python3 -c "import scipy; print(f'scipy {scipy.__version__}')" 2>/dev/null || echo "FAILED: scipy not found!"
python3 -c "import torch; print(f'torch {torch.__version__} with CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "FAILED: torch not found!"
python3 -c "import soundfile; print(f'soundfile found')" 2>/dev/null || echo "FAILED: soundfile not found!"
python3 -c "import silentcipher; print(f'silentcipher found')" 2>/dev/null || echo "FAILED: silentcipher not found!"
python3 -c "import gradio; print(f'gradio found')" 2>/dev/null || echo "FAILED: gradio not found!"
python3 -c "import safetensors; print(f'safetensors found')" 2>/dev/null || echo "FAILED: safetensors not found!"
python3 -c "import moshi; print(f'moshi found')" 2>/dev/null || echo "FAILED: moshi not found!"
# Check if moshi.models.seanet is available
python3 -c "from moshi.models import seanet; print(f'moshi.models.seanet found')" 2>/dev/null || echo "FAILED: moshi.models.seanet not found!"
echo "==========================================================="

# Create a simple test script to verify moshi
cat > test_moshi.py << 'EOL'
try:
    import moshi
    print(f"Moshi version: {moshi.__version__}")
    
    from moshi.models import seanet
    print("Successfully imported moshi.models.seanet")
    
    from moshi.models.seanet import MimiModel
    print("Successfully imported MimiModel")
    
    print("All moshi imports working correctly!")
except ImportError as e:
    print(f"Error importing moshi: {e}")
EOL

echo "Running moshi test script..."
python test_moshi.py

echo ""
echo "Setup complete!"
echo "Run ./run_gradio.sh to start the application."
echo ""
echo "Please download the model files manually as described above."
echo ""
