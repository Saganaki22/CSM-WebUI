import os
import gradio as gr
import torch
import torchaudio
import sys
import tempfile
from pathlib import Path
import traceback
import numpy as np
import requests
import shutil
import json

# Set environment variables to prevent auto-downloading
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Add the current directory and parent directories to the path to import the generator module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Print current directory structure for debugging
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# Add both to sys.path
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Also check for CSM-WebUI directory
csm_webui_dir = os.path.join(parent_dir, "CSM-WebUI")
if os.path.exists(csm_webui_dir) and csm_webui_dir not in sys.path:
    print(f"Adding CSM-WebUI directory to path: {csm_webui_dir}")
    sys.path.append(csm_webui_dir)

def debug_path_check(file_path):
    """Debug helper to check if a file exists at the given path."""
    abs_path = os.path.abspath(file_path)
    exists = os.path.exists(file_path)
    dir_exists = os.path.exists(os.path.dirname(file_path))
    
    print(f"Debug path check for: {file_path}")
    print(f"  Absolute path: {abs_path}")
    print(f"  File exists: {exists}")
    print(f"  Directory exists: {dir_exists}")
    if exists:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
    return exists

# Copy generator.py to current directory if it doesn't exist
current_generator_path = os.path.join(current_dir, "generator.py")
if not os.path.exists(current_generator_path):
    print(f"generator.py not found in current directory, checking for it in other locations...")
    
    # Possible locations for generator.py
    possible_paths = [
        os.path.join(parent_dir, "generator.py"),
        os.path.join(csm_webui_dir, "generator.py") if os.path.exists(csm_webui_dir) else None,
        os.path.join(current_dir, "csm", "generator.py")
    ]
    
    # Filter out None values
    possible_paths = [p for p in possible_paths if p]
    
    # Try to find generator.py
    generator_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found generator.py at: {path}")
            try:
                # Create a copy in the current directory
                import shutil
                shutil.copy(path, current_generator_path)
                print(f"Copied generator.py to current directory: {current_generator_path}")
                generator_found = True
                break
            except Exception as e:
                print(f"Error copying generator.py: {str(e)}")
    
    if not generator_found:
        print("Warning: generator.py not found in any standard location.")

# Import the CSM model functions
try:
    print("Attempting to import generator module...")
    from generator import load_csm_1b, Segment, debug_path_check
    # Patch the Generator.__init__ method to prioritize local mimi files
    from generator import Generator
    
    # Save the original __init__ method
    original_init = Generator.__init__
    
    # Define a new __init__ method that prioritizes local mimi files
    def new_init(self, model):
        self._model = model
        self._model.setup_caches(1)

        # Fix: Import and use the module-level function
        from generator import load_llama3_tokenizer

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        
        # Try multiple approaches to load the mimi weights
        try:
            print("Attempting to load mimi weights...")
            mimi = None
            
            # First check the models/mimi directory
            local_mimi_path = os.path.join("models", "mimi", "model.safetensors")
            config_path = os.path.join("models", "mimi", "config.json")
            preprocessor_config_path = os.path.join("models", "mimi", "preprocessor_config.json")
            
            if os.path.exists(local_mimi_path):
                try:
                    print(f"Loading mimi from local path: {local_mimi_path}")
                    
                    # Load config if available
                    config = {}
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            print(f"Loaded mimi config from {config_path}")
                        except Exception as e:
                            print(f"Error loading mimi config: {str(e)}")
                    
                    # Load preprocessor config if available
                    if os.path.exists(preprocessor_config_path):
                        try:
                            with open(preprocessor_config_path, 'r') as f:
                                preprocessor_config = json.load(f)
                            print(f"Loaded mimi preprocessor config from {preprocessor_config_path}")
                            
                            # Add relevant preprocessor settings to the config
                            if "sample_rate" in preprocessor_config:
                                config["sample_rate"] = preprocessor_config["sample_rate"]
                        except Exception as e:
                            print(f"Error loading mimi preprocessor config: {str(e)}")
                    
                    # Import moshi models properly
                    # Fix: Use try/except to handle the import error
                    try:
                        from moshi.models.seanet import MimiModel
                        print("Successfully imported MimiModel directly")
                        
                        # Create model with config
                        mimi_model = MimiModel(**config).to(device)
                    except ImportError:
                        # If direct import fails, use loaders instead
                        print("Could not import MimiModel directly - will use loaders instead")
                        from moshi.models import loaders
                        
                        # Create model using loaders
                        print("Creating mimi model using loaders.get_mimi()")
                        mimi_model = loaders.get_mimi(local_mimi_path, device=device)
                    
                    from safetensors.torch import load_file
                    
                    # Load weights from safetensors file
                    print(f"Loading mimi weights from safetensors file...")
                    state_dict = load_file(local_mimi_path, device=device)
                    
                    # Print first few keys for debugging
                    key_list = list(state_dict.keys())
                    print(f"Loaded {len(key_list)} keys from safetensors file.")
                    if key_list:
                        print(f"First few keys: {key_list[:5]}")
                    
                    # Load weights with non-strict setting to handle potential key mismatches
                    missing_keys, unexpected_keys = mimi_model.load_state_dict(state_dict, strict=False)
                    
                    # Print any missing or unexpected keys (limited to first 10)
                    if missing_keys:
                        print(f"Missing {len(missing_keys)} keys, first 10: {missing_keys[:10]}")
                    if unexpected_keys:
                        print(f"Unexpected {len(unexpected_keys)} keys, first 10: {unexpected_keys[:10]}")
                    
                    # Set default sample rate if not in config
                    if not hasattr(mimi_model, 'sample_rate'):
                        setattr(mimi_model, 'sample_rate', 24000)
                        print("Set default sample rate to 24000")
                    
                    # Set the mimi model
                    mimi = mimi_model
                    print("Successfully loaded local mimi model")
                    
                except Exception as e:
                    print(f"Error loading local mimi: {str(e)}")
                    print(traceback.format_exc())
                    mimi = None
            
            # If local loading failed, try downloading
            if mimi is None:
                print("Local mimi files not found or failed to load. Downloading from Hugging Face...")
                
                # Ask user if they want to continue with the download
                print("Do you want to proceed with downloading the mimi model? (y/n)")
                # Since we can't get user input here, we'll set a default behavior
                # Set use_local_only to True if you want to force using local files only
                use_local_only = False
                
                if not use_local_only:
                    # Temporarily disable offline mode
                    original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
                    os.environ['HF_DATASETS_OFFLINE'] = '0'
                    
                    from moshi.models import loaders
                    from huggingface_hub import hf_hub_download
                    
                    try:
                        mimi_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
                        print(f"Loading downloaded mimi weights from: {mimi_path}")
                        mimi = loaders.get_mimi(mimi_path, device=device)
                        print("Successfully loaded mimi model from downloaded file")
                        
                        # Optionally save the downloaded model to the local path for future use
                        try:
                            import shutil
                            # Save a copy to the local mimi directory
                            os.makedirs(os.path.dirname(local_mimi_path), exist_ok=True)
                            shutil.copy(mimi_path, local_mimi_path)
                            print(f"Saved a copy of the downloaded mimi model to {local_mimi_path}")
                        except Exception as e:
                            print(f"Warning: Could not save a local copy: {str(e)}")
                    except Exception as e:
                        print(f"Error downloading mimi file: {str(e)}")
                        print("Falling back to default download...")
                        mimi = loaders.get_mimi(None, device=device)
                        
                    # Restore offline mode
                    if original_hf_offline:
                        os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
                else:
                    print("Download cancelled. Attempting to run without mimi model.")
                    # Create a dummy mimi or raise an exception based on your requirements
            
            # Set codebooks and assign as audio tokenizer
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            
        except Exception as e:
            print(f"Error loading mimi weights: {str(e)}")
            print(traceback.format_exc())
            raise

        try:
            print("Loading watermarker...")
            # Look for a local watermarker file
            from watermarking import load_watermarker
            watermarker_path = os.path.join("models", "csm-1b", "watermarker.pt")
            
            if os.path.exists(watermarker_path):
                print(f"Loading watermarker from local path: {watermarker_path}")
                self._watermarker = load_watermarker(device=device, ckpt_path=watermarker_path)
            else:
                # Skip watermarking if file not available
                print("Watermarker file not found. Watermarking will be disabled.")
                self._watermarker = None
        except Exception as e:
            print(f"Error loading watermarker: {str(e)}")
            print(traceback.format_exc())
            print("Setting watermarker to None - watermarking will be skipped")
            self._watermarker = None

        self.sample_rate = mimi.sample_rate
        self.device = device
    
    # Replace the __init__ method
    Generator.__init__ = new_init
    
except ImportError as e:
    # If the generator module is not in the current directory, try alternative approaches
    print(f"Could not import generator module directly: {str(e)}")
    print("Attempting to write the generator module directly...")
    
    try:
        # Create generator.py in the current directory
        with open(os.path.join(current_dir, "generator.py"), "w") as f:
            f.write('''from dataclasses import dataclass
from typing import List, Tuple
import os
import json

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
# Fix import for MimiModel
try:
    from moshi.models.seanet import MimiModel
except ImportError:
    print("Could not import MimiModel directly, will use loaders instead")
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
import traceback
from safetensors.torch import load_file

# Set environment variables to prevent auto-downloading
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    Load Llama tokenizer from local path if available, otherwise from HF
    """
    # Use os.path.join for proper path handling
    local_tokenizer_path = os.path.join("models", "llama3.2")
    
    if os.path.exists(os.path.join(local_tokenizer_path, "tokenizer.json")):
        # List all tokenizer files found in the directory
        tokenizer_files = [f for f in os.listdir(local_tokenizer_path) 
                          if f.startswith("tokenizer") or f == "special_tokens_map.json"]
        
        print(f"Found local Llama tokenizer files in {local_tokenizer_path}:")
        for file in tokenizer_files:
            file_size = os.path.getsize(os.path.join(local_tokenizer_path, file)) / 1024  # KB
            print(f"  - {file} ({file_size:.2f} KB)")
            
        print("Loading local tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
        
        # Verify tokenizer was loaded correctly by showing vocab size
        print(f"Successfully loaded local tokenizer!")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"Tokenizer model: {tokenizer.name_or_path}")
        
        # Optional: Show a test tokenization
        test_text = "Hello, this is a test of the local tokenizer."
        tokens = tokenizer.encode(test_text)
        print(f"Test tokenization of \'{test_text}\':")
        print(f"Token IDs: {tokens[:10]}... (showing first 10 tokens)")
    else:
        print(f"Local tokenizer not found at {local_tokenizer_path}. Downloading from Hugging Face.")
        tokenizer_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Downloaded tokenizer from {tokenizer_name}")
    
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


def load_local_mimi(device, local_path=None):
    """
    Load the MIMI model from local files in safetensors format.
    
    Args:
        device: The device to load the model on
        local_path: Path to the local mimi model directory
    
    Returns:
        The loaded MimiModel or None if loading failed
    """
    if local_path is None:
        local_path = os.path.join("models", "mimi")
    
    model_path = os.path.join(local_path, "model.safetensors")
    config_path = os.path.join(local_path, "config.json")
    preprocessor_config_path = os.path.join(local_path, "preprocessor_config.json")
    
    if not os.path.exists(model_path):
        print(f"Mimi model file not found at {model_path}")
        return None
    
    print(f"Loading mimi from local path: {model_path}")
    
    # Load config if available
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded mimi config from {config_path}")
        except Exception as e:
            print(f"Error loading mimi config: {str(e)}")
    
    # Load preprocessor config if available
    if os.path.exists(preprocessor_config_path):
        try:
            with open(preprocessor_config_path, 'r') as f:
                preprocessor_config = json.load(f)
            print(f"Loaded mimi preprocessor config from {preprocessor_config_path}")
            
            # Add relevant preprocessor settings to the config
            if "sample_rate" in preprocessor_config:
                config["sample_rate"] = preprocessor_config["sample_rate"]
        except Exception as e:
            print(f"Error loading mimi preprocessor config: {str(e)}")
    
    try:
        # Use loaders.get_mimi instead of direct MimiModel instantiation
        mimi_model = loaders.get_mimi(model_path, device=device)
        
        # Set default sample rate if not in config
        if not hasattr(mimi_model, 'sample_rate'):
            setattr(mimi_model, 'sample_rate', 24000)
            print("Set default sample rate to 24000")
        
        print("Successfully loaded local mimi model")
        return mimi_model
    except Exception as e:
        print(f"Error loading local mimi model: {str(e)}")
        print(traceback.format_exc())
        return None


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        
        # Try to load mimi from local files first
        try:
            print("Attempting to load mimi weights...")
            mimi = None
            
            # First try loading from local path
            local_mimi_dir = os.path.join("models", "mimi")
            print(f"Checking for local mimi files in {local_mimi_dir}")
            
            # Check if the model file exists
            model_path = os.path.join(local_mimi_dir, "model.safetensors")
            if os.path.exists(model_path):
                print(f"Found local mimi file: {model_path}")
                mimi = load_local_mimi(device, local_mimi_dir)
            
            # If local loading failed, try downloading
            if mimi is None:
                print("Local mimi loading failed or files not found.")
                print("Do you want to download mimi from Hugging Face? (downloading automatically)")
                
                # Temporarily disable offline mode for download
                original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
                os.environ['HF_DATASETS_OFFLINE'] = '0'
                
                try:
                    mimi_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
                    print(f"Loading downloaded mimi weights from: {mimi_path}")
                    mimi = loaders.get_mimi(mimi_path, device=device)
                    print("Successfully loaded mimi model from downloaded file")
                    
                    # Save downloaded model to local path for future use
                    try:
                        os.makedirs(local_mimi_dir, exist_ok=True)
                        import shutil
                        local_model_path = os.path.join(local_mimi_dir, "model.safetensors")
                        shutil.copy(mimi_path, local_model_path)
                        print(f"Saved downloaded mimi model to {local_model_path} for future use")
                    except Exception as e:
                        print(f"Error saving downloaded model locally: {str(e)}")
                except Exception as e:
                    print(f"Error downloading mimi: {str(e)}")
                    # Fall back to default loader with None path
                    print("Falling back to default mimi loader...")
                    mimi = loaders.get_mimi(None, device=device)
                
                # Restore offline mode
                if original_hf_offline:
                    os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
            
            # Set codebooks and assign as audio tokenizer
            mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            
        except Exception as e:
            print(f"Error during mimi initialization: {str(e)}")
            print(traceback.format_exc())
            raise
        
        try:
            print("Loading watermarker...")
            # Look for a local watermarker file
            watermarker_path = os.path.join("models", "csm-1b", "watermarker.pt")
            
            if os.path.exists(watermarker_path):
                print(f"Loading watermarker from local path: {watermarker_path}")
                self._watermarker = load_watermarker(device=device, ckpt_path=watermarker_path)
            else:
                # Skip watermarking if file not available
                print("Watermarker file not found. Watermarking will be disabled.")
                self._watermarker = None
        except Exception as e:
            print(f"Error loading watermarker: {str(e)}")
            print(traceback.format_exc())
            print("Setting watermarker to None - watermarking will be skipped")
            self._watermarker = None

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_audio_frames = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048 - max_audio_frames
        if curr_tokens.size(1) >= max_seq_len:
            raise ValueError(f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}")

        for _ in range(max_audio_frames):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # Apply watermark if watermarker is available
        if self._watermarker is not None:
            # This applies an imperceptible watermark to identify audio as AI-generated.
            # Watermarking ensures transparency, dissuades misuse, and enables traceability.
            # Please be a responsible AI citizen and keep the watermarking in place.
            # If using CSM 1B in another application, use your own private key and keep it secret.
            audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
            audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        else:
            print("Skipping watermarking - watermarker not available")

        return audio


def debug_path_check(file_path):
    """Debug helper to check if a file exists at the given path."""
    abs_path = os.path.abspath(file_path)
    exists = os.path.exists(file_path)
    dir_exists = os.path.exists(os.path.dirname(file_path))
    
    print(f"Debug path check for: {file_path}")
    print(f"  Absolute path: {abs_path}")
    print(f"  File exists: {exists}")
    print(f"  Directory exists: {dir_exists}")
    if exists:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
    return exists


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda", config_path: str = None) -> Generator:
    """
    Load the CSM model from the given path.
    
    Args:
        ckpt_path: Path to the model weights file
        device: Device to load the model on ('cuda' or 'cpu')
        config_path: Optional path to a config.json file
    """
    print(f"\\n===== LOADING CSM MODEL =====")
    print(f"Loading CSM model from: {ckpt_path}")
    debug_path_check(ckpt_path)
    
    # Check for config file with proper path handling
    if not config_path:
        config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    
    # Debug the config path
    print(f"Checking for config file at: {config_path}")
    config_exists = os.path.exists(config_path)
    print(f"Config file exists: {config_exists}")
    
    # Default model args
    default_args = {
        "backbone_flavor": "llama-1B",
        "decoder_flavor": "llama-100M",
        "text_vocab_size": 128256,
        "audio_vocab_size": 2051,
        "audio_num_codebooks": 32,
    }
    
    # Try to load config if it exists
    if config_exists:
        print(f"Using configuration from: {config_path}")
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"Loaded configuration with keys: {list(config.keys())}")
                
                # Update default args with values from config
                for key in default_args:
                    if key in config:
                        default_args[key] = config[key]
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            print(f"Using default parameters instead")
    else:
        print(f"No configuration file found at {config_path}, using default parameters")
    
    # Create model with arguments
    model_args = ModelArgs(
        backbone_flavor=default_args["backbone_flavor"],
        decoder_flavor=default_args["decoder_flavor"],
        text_vocab_size=default_args["text_vocab_size"],
        audio_vocab_size=default_args["audio_vocab_size"],
        audio_num_codebooks=default_args["audio_num_codebooks"],
    )
    
    print(f"Initializing model with arguments: {model_args}")
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    
    # Load the model weights
    print(f"Loading model weights...")
    if ckpt_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            print(f"Using safetensors to load {ckpt_path}")
            state_dict = load_file(ckpt_path, device=device)
            print(f"Successfully loaded state dict with {len(state_dict)} keys")
        except ImportError:
            raise ImportError("safetensors is required to load .safetensors files. Please install it with 'pip install safetensors'")
    else:
        print(f"Using torch.load to load {ckpt_path}")
        state_dict = torch.load(ckpt_path)
    
    print(f"Loading state dict into model...")
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully")
    
    # Create the generator
    print(f"Creating generator...")
    generator = Generator(model)
    print(f"Generator created successfully")
    return generator''')
        
        print("Successfully created generator.py in the current directory")
        
        # Try to import again
        print("Attempting to import the newly created generator module...")
        from generator import load_csm_1b, Segment, debug_path_check, Generator
        print("Successfully imported generator module!")
        
    except ImportError:
        print("Failed to import from dynamically created generator.py as well.")
        print("Please ensure that the necessary dependencies (torch, torchaudio, etc.) are installed.")
        print("You may need to manually create the generator.py file in the CSM-WebUI directory.")
        sys.exit(1)

# Global variable to store the generator
generator = None

# A mapping from voice name to speaker ID
VOICE_TO_SPEAKER = {
    "conversational_a": 0,
    "conversational_b": 1,
    "read_speech_a": 2,
    "read_speech_b": 3,
    "read_speech_c": 4,
    "read_speech_d": 5
}

# A mapping from voice name to default audio file
VOICE_TO_AUDIO = {
    "conversational_a": os.path.join("sounds", "woman.mp3"),
    "conversational_b": os.path.join("sounds", "man.mp3"),
    "read_speech_a": os.path.join("sounds", "read_speech_a.wav"),
    "read_speech_b": os.path.join("sounds", "read_speech_b.wav"),
    "read_speech_c": os.path.join("sounds", "read_speech_c.wav"),
    "read_speech_d": os.path.join("sounds", "read_speech_d.wav")
}

# A mapping from voice name to default text prompt
VOICE_TO_PROMPT = {
    "conversational_a": "like revising for an exam I'd have to try and like keep up the momentum because I'd start really early I'd be like okay I'm gonna start revising now and then like you're revising for ages and then I just like start losing steam I didn't do that for the exam we had recently to be fair that was a more of a last minute scenario but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I sort of started my day I sort of just like get a bit when I start",
    "conversational_b": "like a super Mario level. Like it's very like high detail. And like, once you get into the park, it just like, everything looks like a computer game and they have all these, like, you know, it. If there's like a you know, like is a Mario game, they will have like a question block. And if you hit you know, and it's just like, it's just like for like the everyone, when they come into the park, they get like this little bracelet and then you can go punching question blocks around.",
    "read_speech_a": "And Lake turned round upon me, a little abruptly, his odd yellowish eyes, a little like those of the sea eagle, and the ghost of his smile that flickered on his singularly pale face, with a stern and insidious look, confronted me.",
    "read_speech_b": "He was such a big boy that he wore high boots and carried a jack knife. He gazed and gazed at the cap, and could not keep from fingering the blue tassel.",
    "read_speech_c": "All passed so quickly, there was so much going on around him, the Tree quite forgot to look to himself.",
    "read_speech_d": "Suddenly I was back in the old days Before you felt we ought to drift apart. It was some trick-the way your eyebrows raise."
}

def ensure_directories():
    """Ensure all required directories exist."""
    dirs_to_create = [
        os.path.join("models", "csm-1b"),
        os.path.join("models", "llama3.2"),
        os.path.join("models", "mimi"),
        "sounds"
    ]
    
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

# Call this function early in the script
ensure_directories()

def load_model(model_path):
    """Load the CSM model from the given path."""
    global generator
    try:
        print("\n===== LOADING MODEL =====")
        print(f"Current working directory: {os.getcwd()}")
        debug_path_check(model_path)
        
        # Also check for config.json in the same directory
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        debug_path_check(config_path)
        
        # Check for mimi files
        mimi_path = os.path.join("models", "mimi", "model.safetensors")
        mimi_config_path = os.path.join("models", "mimi", "config.json")
        debug_path_check(mimi_path)
        debug_path_check(mimi_config_path)
        
        if not os.path.exists(model_path):
            return f"Error: Model file not found at {model_path}"
        
        # Ensure the model directory is in sys.path to help with imports
        model_dir = os.path.dirname(model_path)
        if model_dir not in sys.path:
            sys.path.append(model_dir)
            print(f"Added {model_dir} to sys.path")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on device: {device}")
        generator = load_csm_1b(model_path, device)
        
        return f"Model loaded successfully on {device}."
    except Exception as e:
        print(f"Error stack trace:")
        print(traceback.format_exc())
        return f"Error loading model: {str(e)}"

def download_model(output_path):
    """Download the CSM model and all necessary files from Hugging Face."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Files to download from the CSM-1B repository
        files_to_download = {
            "model.safetensors": "https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors?download=true",
        }
        
        # Download each file
        for filename, url in files_to_download.items():
            file_path = os.path.join(os.path.dirname(output_path), filename)
            
            if os.path.exists(file_path):
                print(f"File {filename} already exists at {file_path}. Skipping download.")
                continue
                
            print(f"Downloading {filename} from {url} to {file_path}...")
            response = requests.get(url, stream=True)
            
            # Check if the request was successful
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                print(f"Successfully downloaded {filename}")
            else:
                # Check for 401 Unauthorized error which may indicate lack of access
                if response.status_code == 401:
                    return f"Error: Unauthorized access. Make sure you have requested and been granted access to {filename} on Hugging Face."
                else:
                    return f"Error downloading {filename}: HTTP status code {response.status_code}"
        
        # Create a config.json file in the same directory
        config_path = os.path.join(os.path.dirname(output_path), "config.json")
        config_content = {
            "model_type": "csm-1b",
            "backbone_flavor": "llama-1B",
            "decoder_flavor": "llama-100M",
            "text_vocab_size": 128256,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 32
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_content, f, indent=2)
        print(f"Created config.json at {config_path}")
            
        return f"Model downloaded successfully to {output_path}. A config.json file was also created."
    except Exception as e:
        return f"Error downloading model: {str(e)}"

def load_audio_file(audio_input):
    """Load an audio file or convert an audio array to a tensor."""
    if not audio_input:
        print("Audio input is None or empty")
        return None
    
    try:
        # Handle different input types from Gradio
        if isinstance(audio_input, tuple) and len(audio_input) == 2:
            # This is a tuple of (sample_rate, audio_array)
            sample_rate, audio_array = audio_input
            print(f"Received audio as tuple: sample_rate={sample_rate}, array shape={audio_array.shape}")
            
            # Convert numpy array to torch tensor
            if isinstance(audio_array, np.ndarray):
                # Audio array might be stereo (shape: [samples, 2]) or mono (shape: [samples])
                # Ensure it's the right shape and convert to float
                if len(audio_array.shape) == 2 and audio_array.shape[1] == 2:
                    # Convert stereo to mono by averaging channels
                    audio_array = audio_array.mean(axis=1)
                
                # Convert to float32 and normalize if it's integer data
                if np.issubdtype(audio_array.dtype, np.integer):
                    max_value = np.iinfo(audio_array.dtype).max
                    audio_array = audio_array.astype(np.float32) / max_value
                
                # Create tensor (ensuring mono)
                audio_tensor = torch.from_numpy(audio_array).float()
                print(f"Converted to torch tensor with shape: {audio_tensor.shape}")
            else:
                raise ValueError(f"Unexpected audio array type: {type(audio_array)}")
                
        elif isinstance(audio_input, str) or hasattr(audio_input, '__fspath__'):
            # This is a file path
            print(f"Loading audio from file path: {audio_input}")
            audio_tensor, sample_rate = torchaudio.load(audio_input)
            # If stereo, convert to mono
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            audio_tensor = audio_tensor.squeeze(0)  # Remove channel dimension for mono
            print(f"Loaded audio tensor with shape: {audio_tensor.shape}, sample_rate: {sample_rate}")
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        
        # Resample if the generator is loaded and we know the target sample rate
        if generator:
            target_sr = generator.sample_rate
            print(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz")
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=sample_rate, 
                new_freq=target_sr
            )
            print(f"Resampled audio shape: {audio_tensor.shape}")
        else:
            print("Warning: Generator not loaded, skipping resampling")
        
        return audio_tensor
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        print(traceback.format_exc())
        return None

def create_segment(text, speaker_id, audio_input):
    """Create a segment for context."""
    if not text.strip():
        print("Empty text, skipping segment creation")
        return None
    
    print(f"Creating segment with text: '{text}', speaker: {speaker_id}")
    audio = load_audio_file(audio_input) if audio_input else None
    if audio is None:
        print("No audio loaded for this segment")
    
    try:
        segment = Segment(
            text=text,
            speaker=int(speaker_id),
            audio=audio
        )
        print("Segment created successfully")
        return segment
    except Exception as e:
        print(f"Error creating segment: {str(e)}")
        print(traceback.format_exc())
        return None

def generate_speech_simple(text, speaker_id, max_audio_length_ms):
    """Generate speech from text without context - simplified version for the Simple tab."""
    global generator
    
    if generator is None:
        return None, "Model not loaded. Please load the model first."
    
    if not text.strip():
        return None, "Please enter text to generate speech."
    
    try:
        # Generate audio
        print(f"Generating speech for text: '{text}', speaker: {speaker_id}")
        audio = generator.generate(
            text=text,
            speaker=int(speaker_id),
            context=[],  # Empty context
            max_audio_length_ms=int(max_audio_length_ms)
        )
        
        # Save temporary audio file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "csm_output.wav")
        print(f"Saving output to {output_path}")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        return output_path, "Speech generated successfully."
    except Exception as e:
        print(f"Error in generate_speech: {str(e)}")
        print(traceback.format_exc())
        return None, f"Error generating speech: {str(e)}"

def generate_speech(
    text, 
    speaker_id, 
    max_audio_length_ms,
    context_text_1="", context_speaker_1="0", context_audio_1=None,
    context_text_2="", context_speaker_2="0", context_audio_2=None,
    context_text_3="", context_speaker_3="0", context_audio_3=None,
    context_text_4="", context_speaker_4="0", context_audio_4=None
):
    """Generate speech from text with optional context."""
    global generator
    
    if generator is None:
        return None, "Model not loaded. Please load the model first."
    
    if not text.strip():
        return None, "Please enter text to generate speech."
    
    try:
        # Create context segments
        context_segments = []
        
        # Add context segments if they have text
        contexts = [
            (context_text_1, context_speaker_1, context_audio_1),
            (context_text_2, context_speaker_2, context_audio_2),
            (context_text_3, context_speaker_3, context_audio_3),
            (context_text_4, context_speaker_4, context_audio_4)
        ]
        
        for i, (ctx_text, ctx_speaker, ctx_audio) in enumerate(contexts):
            if ctx_text.strip():
                print(f"Processing context {i+1}")
                segment = create_segment(ctx_text, ctx_speaker, ctx_audio)
                if segment:
                    context_segments.append(segment)
                    print(f"Added context segment {i+1}")
                else:
                    print(f"Failed to create segment for context {i+1}")
        
        # Generate audio
        print(f"Generating speech for text: '{text}', speaker: {speaker_id}")
        audio = generator.generate(
            text=text,
            speaker=int(speaker_id),
            context=context_segments,
            max_audio_length_ms=int(max_audio_length_ms)
        )
        
        # Save temporary audio file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "csm_output.wav")
        print(f"Saving output to {output_path}")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        return output_path, "Speech generated successfully."
    except Exception as e:
        print(f"Error in generate_speech: {str(e)}")
        print(traceback.format_exc())
        return None, f"Error generating speech: {str(e)}"

def generate_conversation(speaker_a_text, speaker_a_voice, speaker_a_audio, speaker_b_text, speaker_b_voice, speaker_b_audio, conversation_text):
    """Generate a conversation between two speakers."""
    global generator
    global VOICE_TO_SPEAKER
    
    if generator is None:
        return None, "Model not loaded. Please load the model first."
    
    try:
        # Debugging information
        print("\n==== DEBUG INFORMATION ====")
        print(f"Speaker A voice: {speaker_a_voice}")
        print(f"Speaker B voice: {speaker_b_voice}")
        print(f"Speaker A audio: {speaker_a_audio}")
        print(f"Speaker B audio: {speaker_b_audio}")
        
        # Convert voice names to numerical speaker IDs
        speaker_a_id = VOICE_TO_SPEAKER.get(speaker_a_voice, 0)
        speaker_b_id = VOICE_TO_SPEAKER.get(speaker_b_voice, 1)
        
        print(f"Speaker A ID: {speaker_a_id}")
        print(f"Speaker B ID: {speaker_b_id}")
        
        # Load the audio files for voice cloning
        speaker_a_audio_tensor = load_audio_file(speaker_a_audio)
        speaker_b_audio_tensor = load_audio_file(speaker_b_audio)
        
        if speaker_a_audio_tensor is None:
            print("WARNING: Could not load Speaker A audio file")
        else:
            print(f"Loaded Speaker A audio tensor with shape: {speaker_a_audio_tensor.shape}")
            # Move to the same device as the model
            speaker_a_audio_tensor = speaker_a_audio_tensor.to(generator.device)
            
        if speaker_b_audio_tensor is None:
            print("WARNING: Could not load Speaker B audio file")
        else:
            print(f"Loaded Speaker B audio tensor with shape: {speaker_b_audio_tensor.shape}")
            # Move to the same device as the model
            speaker_b_audio_tensor = speaker_b_audio_tensor.to(generator.device)
        
        # Create speaker segments for voice cloning
        speaker_a_segment = None
        speaker_b_segment = None
        
        if speaker_a_audio_tensor is not None:
            speaker_a_segment = Segment(
                text=speaker_a_text,
                speaker=speaker_a_id,
                audio=speaker_a_audio_tensor
            )
            print("Created Speaker A segment for voice cloning")
        
        if speaker_b_audio_tensor is not None:
            speaker_b_segment = Segment(
                text=speaker_b_text,
                speaker=speaker_b_id,
                audio=speaker_b_audio_tensor
            )
            print("Created Speaker B segment for voice cloning")
        
        # Parse the conversation text into lines
        lines = conversation_text.strip().split('\n')
        print(f"Found {len(lines)} lines in the conversation")
        
        # Process each line
        turns = []
        for i, line in enumerate(lines):
            if line.strip():
                # First line (index 0) is Speaker A, second line (index 1) is Speaker B, etc.
                is_speaker_a = (i % 2 == 0)
                speaker_id = speaker_a_id if is_speaker_a else speaker_b_id
                speaker_label = "A" if is_speaker_a else "B"
                
                print(f"Line {i+1}: \"{line}\" --> Speaker {speaker_label} (ID={speaker_id})")
                turns.append((line, speaker_id, is_speaker_a))
        
        # Generate audio for the conversation
        combined_audio = None
        sample_rate = generator.sample_rate
        device = generator.device
        
        for i, (text, speaker_id, is_speaker_a) in enumerate(turns):
            speaker_label = "A" if is_speaker_a else "B"
            print(f"Generating audio for turn {i+1}: Speaker {speaker_label} (ID={speaker_id}), Text: '{text}'")
            
            # Set the appropriate context for voice cloning
            context = []
            if is_speaker_a and speaker_a_segment is not None:
                context.append(speaker_a_segment)
                print("Using Speaker A voice cloning context")
            elif not is_speaker_a and speaker_b_segment is not None:
                context.append(speaker_b_segment)
                print("Using Speaker B voice cloning context")
            
            # Generate the audio
            audio = generator.generate(
                text=text,
                speaker=speaker_id,
                context=context,  # Now using the audio context for voice cloning
                max_audio_length_ms=10000
            )
            
            # Combine audio
            if combined_audio is None:
                combined_audio = audio
            else:
                pause_length = int(0.5 * sample_rate)
                silence = torch.zeros(pause_length, device=device)
                combined_audio = torch.cat([combined_audio, silence, audio])
        
        # Save temporary audio file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "csm_conversation.wav")
        print(f"Saving output to {output_path}")
        torchaudio.save(output_path, combined_audio.unsqueeze(0).cpu(), sample_rate)
        
        return output_path, "Conversation generated successfully."
    except Exception as e:
        print(f"Error in generate_conversation: {str(e)}")
        print(traceback.format_exc())
        return None, f"Error generating conversation: {str(e)}"

# Function to update audio file and prompt based on selected voice
def update_speaker_audio_and_prompt(voice):
    """Return the path to the audio file corresponding to the selected voice."""
    global VOICE_TO_AUDIO
    return VOICE_TO_AUDIO.get(voice, None)

def update_speaker_prompt(voice):
    """Return the text prompt corresponding to the selected voice."""
    global VOICE_TO_PROMPT
    return VOICE_TO_PROMPT.get(voice, "")

# Define the Gradio interface
with gr.Blocks(title="CSM-WebUI (WSL)") as app:
    # Use HTML directly for the title to ensure proper rendering and linking, with adjusted font size
    gr.HTML("""
    <div style="text-align: center; margin: 20px 0;">
        <a href="https://github.com/Saganaki22/CSM-WebUI" style="text-decoration: none; color: inherit;">
            <h1 style="font-size: 2.3rem; font-weight: bold; margin: 0;">CSM-WebUI (WSL)</h1>
        </a>
    </div>
    """)
    
    gr.Markdown("""
    CSM (Conversational Speech Model) is a speech generation model from Sesame that generates RVQ audio codes from text and audio inputs.
    
    This interface allows you to generate speech from text, with optional conversation context. 
    
    This is the WSL (Windows Subsystem for Linux) version.
    """, elem_classes=["center-aligned"])
    
    # Add CSS for center alignment
    gr.HTML("""
    <style>
        .center-aligned {
            text-align: center !important;
        }
        .center-aligned a {
            text-decoration: none;
            color: inherit;
        }
    </style>
    """)
    
    with gr.Row():
        model_path = gr.Textbox(
            label="Model Path",
            placeholder="Path to the CSM model file model.safetensors",
            value=os.path.join("models", "csm-1b", "model.safetensors")
        )
        with gr.Column():
            load_button = gr.Button("Load Model")
            download_button = gr.Button("Download Model")
        model_status = gr.Textbox(label="Model Status", interactive=False)
    
    gr.Markdown("""
    **Note:** Make sure you have the models from HF [csm-1b](https://huggingface.co/drbaph/CSM-1B/tree/main) [Llama-3.2-1b](https://huggingface.co/unsloth/Llama-3.2-1B/tree/main) in the correct directories.
    
    **WSL Note:** This version is designed to run in Linux environment.
    """)
    
    load_button.click(load_model, inputs=[model_path], outputs=[model_status])
    download_button.click(download_model, inputs=[model_path], outputs=[model_status])
    
    with gr.Tab("Simple Generation"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter text to convert to speech",
                    lines=3
                )
                speaker_id = gr.Dropdown(
                    label="Speaker ID",
                    choices=["0", "1", "2", "3", "4", "5"],
                    value="0"
                )
                max_audio_length = gr.Slider(
                    label="Max Audio Length (ms)",
                    minimum=1000,
                    maximum=30000,
                    value=10000,
                    step=1000
                )
                generate_button = gr.Button("Generate Speech")
            
            with gr.Column():
                output_audio = gr.Audio(label="Generated Speech")
                generation_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("Generation with Context"):
        with gr.Row():
            with gr.Column():
                input_text_ctx = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter text to convert to speech",
                    lines=3
                )
                speaker_id_ctx = gr.Dropdown(
                    label="Speaker ID",
                    choices=["0", "1", "2", "3", "4", "5"],
                    value="0"
                )
                max_audio_length_ctx = gr.Slider(
                    label="Max Audio Length (ms)",
                    minimum=1000,
                    maximum=30000,
                    value=10000,
                    step=1000
                )
            
            with gr.Column():
                output_audio_ctx = gr.Audio(label="Generated Speech")
                generation_status_ctx = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("### Conversation Context")
        gr.Markdown("Add previous utterances as context to improve the speech generation quality.")
        
        with gr.Accordion("Context Utterance 1", open=True):
            with gr.Row():
                context_text_1 = gr.Textbox(
                    label="Text",
                    placeholder="Text of the previous utterance",
                    lines=2
                )
                context_speaker_1 = gr.Dropdown(
                    label="Speaker ID",
                    choices=["0", "1", "2", "3", "4", "5"],
                    value="0"
                )
            context_audio_1 = gr.Audio(
                label="Upload/Record Audio (Optional)",
                type="filepath"
            )
            gr.Markdown("*You can upload your own audio file or record from microphone. This helps the model match the voice style.*")
        
        with gr.Accordion("Context Utterance 2", open=False):
            with gr.Row():
                context_text_2 = gr.Textbox(
                    label="Text",
                    placeholder="Text of the previous utterance",
                    lines=2
                )
                context_speaker_2 = gr.Dropdown(
                    label="Speaker ID",
                    choices=["0", "1", "2", "3", "4", "5"],
                    value="1"
                )
            context_audio_2 = gr.Audio(
                label="Upload/Record Audio (Optional)",
                type="filepath"
            )
            gr.Markdown("*You can upload your own audio file or record from microphone.*")
        
        with gr.Accordion("Context Utterance 3", open=False):
            with gr.Row():
                context_text_3 = gr.Textbox(
                    label="Text",
                    placeholder="Text of the previous utterance",
                    lines=2
                )
                context_speaker_3 = gr.Dropdown(
                    label="Speaker ID",
                    choices=["0", "1", "2", "3", "4", "5"],
                    value="0"
                )
            context_audio_3 = gr.Audio(
                label="Upload/Record Audio (Optional)",
                type="filepath"
            )
            gr.Markdown("*You can upload your own audio file or record from microphone.*")
        
        with gr.Accordion("Context Utterance 4", open=False):
            with gr.Row():
                context_text_4 = gr.Textbox(
                    label="Text",
                    placeholder="Text of the previous utterance",
                    lines=2
                )
                context_speaker_4 = gr.Dropdown(
                    label="Speaker ID",
                    choices=["0", "1", "2", "3", "4", "5"],
                    value="1"
                )
            context_audio_4 = gr.Audio(
                label="Upload/Record Audio (Optional)",
                type="filepath"
            )
            gr.Markdown("*You can upload your own audio file or record from microphone.*")
        
        generate_button_ctx = gr.Button("Generate Speech with Context")
    
    with gr.Tab("Official Demo"):
        gr.Markdown("# Voices")
        
        with gr.Row():
            # Speaker A column
            with gr.Column():
                gr.Markdown("### Speaker A")
                speaker_a_voice = gr.Dropdown(
                    label="Select a predefined speaker",
                    choices=["conversational_a", "conversational_b", "read_speech_a", "read_speech_b", "read_speech_c", "read_speech_d"],
                    value="conversational_a"
                )
                
                with gr.Accordion("Or add your own voice prompt", open=False):
                    speaker_a_prompt = gr.Textbox(
                        label="Speaker prompt",
                        placeholder="Enter text for the voice prompt",
                        value=VOICE_TO_PROMPT["conversational_a"],
                        lines=5
                    )
                
                speaker_a_audio = gr.Audio(
                    label="Speaker prompt",
                    type="filepath",
                    value=os.path.join("sounds", "woman.mp3")  # Using os.path.join
                )
            
            # Speaker B column
            with gr.Column():
                gr.Markdown("### Speaker B")
                speaker_b_voice = gr.Dropdown(
                    label="Select a predefined speaker",
                    choices=["conversational_a", "conversational_b", "read_speech_a", "read_speech_b", "read_speech_c", "read_speech_d"],
                    value="conversational_b"
                )
                
                with gr.Accordion("Or add your own voice prompt", open=False):
                    speaker_b_prompt = gr.Textbox(
                        label="Speaker prompt",
                        placeholder="Enter text for the voice prompt",
                        value=VOICE_TO_PROMPT["conversational_b"],
                        lines=5
                    )
                
                speaker_b_audio = gr.Audio(
                    label="Speaker prompt",
                    type="filepath",
                    value=os.path.join("sounds", "man.mp3")  # Using os.path.join
                )
        
        gr.Markdown("## Conversation content")
        gr.Markdown("Each line is an utterance in the conversation to generate. Speakers alternate between A and B, starting with speaker A.")
        
        conversation_text = gr.Textbox(
            label="conversation",
            placeholder="Enter conversation script, each line is a new turn, alternating between speaker A and B",
            value="Hey how are you doing.\nPretty good, pretty good.\nI'm glad, so happy to be speaking to you.\nMe too! What have you been up to?\nYeah, I've been reading more about speech generation, and it really seems like context is important.\nDefinitely!",
            lines=10
        )
        
        generate_conv_button = gr.Button("Generate conversation", variant="primary")
        
        synthesized_audio = gr.Audio(label="Synthesized audio")
        conversation_status = gr.Textbox(label="Status", interactive=False, visible=False)
    
    # Set up the event handlers
    # Use the simplified function for the Simple Generation tab
    generate_button.click(
        generate_speech_simple,
        inputs=[input_text, speaker_id, max_audio_length],
        outputs=[output_audio, generation_status]
    )
    
    # Use the original function with context for the Context tab
    generate_button_ctx.click(
        generate_speech,
        inputs=[
            input_text_ctx, speaker_id_ctx, max_audio_length_ctx,
            context_text_1, context_speaker_1, context_audio_1,
            context_text_2, context_speaker_2, context_audio_2,
            context_text_3, context_speaker_3, context_audio_3,
            context_text_4, context_speaker_4, context_audio_4
        ],
        outputs=[output_audio_ctx, generation_status_ctx]
    )
    
    # Add event handler for the conversation generation button
    generate_conv_button.click(
        generate_conversation,
        inputs=[
            speaker_a_prompt, speaker_a_voice, speaker_a_audio,
            speaker_b_prompt, speaker_b_voice, speaker_b_audio,
            conversation_text
        ],
        outputs=[synthesized_audio, conversation_status]
    )
    
    # Connect the voice dropdown to the audio file and prompt selection
    speaker_a_voice.change(
        update_speaker_audio_and_prompt,
        inputs=[speaker_a_voice],
        outputs=[speaker_a_audio]
    )
    
    speaker_a_voice.change(
        update_speaker_prompt,
        inputs=[speaker_a_voice],
        outputs=[speaker_a_prompt]
    )
    
    speaker_b_voice.change(
        update_speaker_audio_and_prompt,
        inputs=[speaker_b_voice],
        outputs=[speaker_b_audio]
    )
    
    speaker_b_voice.change(
        update_speaker_prompt,
        inputs=[speaker_b_voice],
        outputs=[speaker_b_prompt]
    )
    
    # Add additional CSS for alignment
    gr.HTML("""
    <style>
        .left-aligned {
            text-align: left !important;
        }
        .right-aligned {
            text-align: right !important;
        }
        .no-bullets {
            list-style-type: none !important;
            padding-left: 0 !important;
        }
        .no-bullets a {
            display: inline-block;
            margin: 0 10px;
        }
    </style>
    """)
    
    # Notes section (centered, with disclaimer added as bullet point)
    gr.Markdown("""
    ### Notes
    - This interface requires the CSM model to be downloaded locally at the specified path.
    - Speaker IDs (0-5) represent different voices the model can generate.
    - Adding conversation context can improve the quality and naturalness of the generated speech.
    - **Audio Upload**: You can upload your own audio files (.wav, .mp3, .ogg, etc.) or record directly with your microphone.
    - **Voice Cloning**: For best results, upload audio samples that match the voice you want to replicate and use the same Speaker ID.
    - As mentioned in the CSM documentation, this model should not be used for: Impersonation or Fraud, Misinformation or Deception, Illegal or Harmful Activities.
    - **WSL Note**: This version is designed to run in Linux environment.
    """, elem_classes=["left-aligned"])
    
    # Add official links section without bullet points
    gr.Markdown("""
    ### Links
    
    [GitHub Repository](https://github.com/Saganaki22/CSM-WebUI) [CSM Repository](https://github.com/SesameAILabs/csm)   [Official Hugging Face Repository](https://huggingface.co/sesame/csm-1b)  [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
    """, elem_classes=["center-aligned", "no-bullets"])

# Launch the app
if __name__ == "__main__":
    app.launch()
