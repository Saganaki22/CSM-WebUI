# CSM-WebUI

This is a fork of the original CSM project that creates a complete Gradio-based web interface, making CSM accessible through an intuitive UI. The fork adds Windows and WSL compatibility and various usability improvements to make speech generation easy and accessible.

![image](https://github.com/user-attachments/assets/ca23144b-54ba-4877-88e9-25d2fe18d597)



https://github.com/user-attachments/assets/86d08077-e0c4-4b84-8ce6-912f969e12c6




https://github.com/user-attachments/assets/68e7ce21-c101-432b-a871-c4892542eb8b





## 🚀 Key Enhancements

- **Windows Compatibility**: Full support for Windows with triton-windows package
- **WSL Support**: Dedicated setup script for Linux/WSL environments
- **Model Format**: Updated to use safetensors format (more efficient and secure)
- **Model Flexibility**: Support for both local models and HuggingFace-hosted models
- **Improved Local Model Loading**: Fixed issues with Llama-3.2-1B model loading
- **Windows-Specific Generator**: Separate generator file for Windows environments
- **Robust Setup Scripts**: Comprehensive setup for both Windows and WSL
- **Git Integration**: Optional Git-based model downloading for better reliability

## 📋 Technical Details

### Model Loading Improvements

The original project had issues loading the models because:

1. The models are gated on HuggingFace and require authentication
2. The Windows version had compatibility issues
3. The tokenizer wasn't configured to use local files

Our solution:
- Use the non-gated [drbaph/CSM-1B](https://huggingface.co/drbaph/CSM-1B) version
- Create a Windows-specific generator with proper path handling
- Provide multiple fallback methods for model loading
- Create robust setup scripts for different environments

### Package Compatibility

Our setup ensures compatible versions of key packages:

- **Windows**:
  - triton-windows instead of triton
  - numpy==1.26.4
  - scipy==1.11.4
  - Flask==2.2.5
  - librosa==0.10.0
  - SoundFile==0.12.1
  - PyTorch 2.4.0 with CUDA 12.4 support

- **WSL/Linux**:
  - Standard triton package
  - Compatible numpy and scipy versions

## 📦 Model Information

The project uses the CSM-1B .safeteonsors model:
- **CSM-1B**: Available at [drbaph/CSM-1B](https://huggingface.co/drbaph/CSM-1B/tree/main)
  place in models/csm-1b/*

- **Llama-3.2-1b**: download from [drbaph/models](https://huggingface.co/drbaph/models/tree/main/llama3.2)
  place in models/llama3.2/*

### HuggingFace Authentication Requirements

***UPDATE*** Since models were gated I re-uploaded them to my repo non-gated for ease of access [HuggingFace](https://huggingface.co/drbaph/models/tree/main)

~~**IMPORTANT:** Before installation, you need to:~~  

~~1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)~~  
~~2. Visit [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and request access to the model~~  
~~3. Create a HuggingFace access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)~~  
~~4. During the installation process, you will be prompted to authenticate with your HuggingFace token - paste the token when prompted~~  

~~Without access to the Llama model and proper authentication, the program will not work as it uses the Llama backbone.~~

## ⚙️ Installation

### Windows Installation

```batch
# Clone the repository
git clone git clone --branch alpha https://github.com/Saganaki22/CSM-WebUI.git

cd CSM-WebUI

# Step 1: Run the improved Windows setup script
verbose-win-setup.bat

# Step 2: Fix PyTorch compilation issues
fix-torch-compile.bat

# Step 3: Run the application using the generated script
run_fixed.bat
```

#### Detailed Windows Installation Steps:

1. **Run verbose-win-setup.bat**
   - Installs all dependencies with detailed output
   - Sets up PyTorch with CUDA 12.4 support
   - Creates virtual environment and installs required packages
   - Optionally downloads the model file
   - **IMPORTANT**: If the script just git pulls and stops without installing dependencies, close it and run it once more
   - **IMPORTANT**: If you encounter a .venv issue, delete the .venv folder and run verbose-win-setup.bat again - this should fix the issue

2. **Run fix-torch-compile.bat**
   - Patches the Moshi library to fix PyTorch compilation errors
   - Creates the run_fixed.bat launcher
   - Makes a backup of the original file in case you need to restore it

3. **Use run_fixed.bat to launch the application**
   - This script is automatically created by fix-torch-compile.bat
   - Properly activates the virtual environment and launches the application
  
 4. **Use hf_login.bat if you get any authentication errors** 


### WSL/Linux Installation

```bash
# Clone the repository
git clone git clone --branch alpha https://github.com/Saganaki22/CSM-WebUI.git

cd CSM-WebUI

# Run the WSL setup script
bash wsl-setup.sh

# After setup completes, run the application
python wsl-gradio.py
```

## 🔧 Troubleshooting

### Windows Common Issues

1. **HuggingFace Authentication Issues**
   - Make sure you have a HuggingFace account and have requested access to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
   - Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) before running the setup
   - When prompted during installation, paste your HuggingFace token for authentication

2. **Setup Script Stopping After Git Pull**
   - If verbose-win-setup.bat only performs a git pull and stops without installing dependencies, simply close the script and run it again
   - This is a known issue that can occur on the first run

2. **Virtual Environment (.venv) Issues**
   - If you encounter errors related to the .venv folder, delete the entire .venv folder and run verbose-win-setup.bat again
   - This completely recreates the virtual environment and resolves most initialization issues

3. **PyTorch Compilation Errors**
   - If you encounter "dataclass errors" or "must be called with a dataclass type or instance" errors, make sure you've run the fix-torch-compile.bat script
   - This error occurs with PyTorch 2.4.0's compilation system on Windows

4. **Missing bitsandbytes Error**
   - If you see "No module named 'bitsandbytes'" error, run verbose-win-setup.bat again which installs this package
   - Alternatively, manually install it: `pip install bitsandbytes-windows`

5. **CUDA Not Available**
   - If PyTorch doesn't detect your CUDA GPU, verify your NVIDIA drivers are up to date
   - Check with `python -c "import torch; print(torch.cuda.is_available())"`

### WSL/Linux Model Loading Issues

If you encounter model loading errors on WSL/Linux:

```bash
# Make sure you're using the correct paths
python wsl-gradio.py --model-path models/model.safetensors
```

## 🔄 Switching Between Windows and WSL

This fork is designed to let you use both environments without conflicts:
- Windows will use the triton-windows package and the win-gradio.py file
- WSL will use the standard triton package and the wsl-gradio.py file

## 🗂️ Directory Structure

```
CSM-WebUI/
├── models/                   # Directory for model files
│   └── model.safetensors     # CSM model file (where setup scripts save model)
├── sounds/                   # Directory for example audio files
│   ├── man.mp3               # Male voice example
│   └── woman.mp3             # Female voice example
├── generator.py              # Generator for speech synthesis
├── watermarking.py           # Audio watermarking functionality
├── wsl-gradio.py             # Gradio UI for WSL/Linux
├── win-gradio.py             # Windows-specific Gradio UI
├── verbose-win-setup.bat     # Improved setup script for Windows with verbose output
├── fix-torch-compile.bat     # Script to fix PyTorch compilation issues
├── wsl-setup.sh              # Setup script for WSL/Linux
└── requirements.txt          # Python package requirements
```

## 💡 Key Differences from Original

1. **Model Storage**: Original required manual downloads, our version simplifies this
2. **File Format**: Using .safetensors for better security and compatibility 
3. **Windows Support**: Added comprehensive Windows support with separate setup script
4. **Dual Environments**: Support for both Windows and WSL without conflicts
5. **Robust Error Handling**: Multiple fallback methods for model loading
6. **Streamlined UI**: Unified interface across platforms
7. **PyTorch Compatibility**: Fixes for Windows-specific PyTorch compilation issues

---

# Original README Below

# CSM

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Usage

Generate a sentence

```python
from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

```python
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
