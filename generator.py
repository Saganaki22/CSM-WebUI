from dataclasses import dataclass
from typing import List, Tuple
import os
import json
import sys
import traceback

# Make sure the current directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Ensure we can import requirements
try:
    import torch
    import torchaudio
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    
    # Try importing the models module
    from models import Model, ModelArgs
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure all dependencies are installed with:")
    print("pip install torch torchaudio transformers tokenizers safetensors")
    print("See requirements.txt for all dependencies")
    # Continue execution - we'll handle errors gracefully later

try:
    # Attempt to manually import moshi components
    import moshi
    from moshi.models import loaders
    try:
        from moshi.models.seanet import MimiModel
    except ImportError:
        print("Warning: Could not import MimiModel from moshi.models.seanet")
        print("Will attempt to use the standard moshi API instead")
        
    from huggingface_hub import hf_hub_download
    
    try:
        from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
    except ImportError:
        print("Warning: watermarking module not available")
except ImportError as e:
    print(f"Warning: {e}")
    print("Some dependencies might be missing. Please install them with:")
    print("pip install moshi==0.2.2 huggingface_hub")

# Set environment variables to prevent auto-downloading
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


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


def load_llama3_tokenizer():
    """
    Load Llama tokenizer from local path if available, otherwise from HF
    """
    try:
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
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, local_files_only=True)
            
            # Verify tokenizer was loaded correctly by showing vocab size
            print(f"Successfully loaded local tokenizer!")
            print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
            print(f"Tokenizer model: {tokenizer.name_or_path}")
            
            # Optional: Show a test tokenization
            test_text = "Hello, this is a test of the local tokenizer."
            tokens = tokenizer.encode(test_text)
            print(f"Test tokenization of '{test_text}':")
            print(f"Token IDs: {tokens[:10]}... (showing first 10 tokens)")
        else:
            print(f"Local tokenizer not found at {local_tokenizer_path}. Downloading from Hugging Face.")
            
            # Temporarily disable offline mode
            original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
            os.environ['HF_DATASETS_OFFLINE'] = '0'
            
            tokenizer_name = "meta-llama/Llama-3.2-1B"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"Downloaded tokenizer from {tokenizer_name}")
            
            # Restore offline mode
            if original_hf_offline:
                os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
        
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
        )

        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print(traceback.format_exc())
        raise


class MimiModelStub:
    """Stub implementation for when proper moshi import fails"""
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.num_codebooks = 32
        print("WARNING: Using MimiModelStub - functionality will be limited")
        
    def set_num_codebooks(self, num_codebooks):
        self.num_codebooks = num_codebooks
        
    def encode(self, audio):
        # Return empty tensors of the right shape
        # This won't work for actual generation but prevents crashes
        batch, channel, samples = audio.size()
        return [torch.zeros((self.num_codebooks, 100), device=audio.device)]
        
    def decode(self, tokens):
        # Return a short silence
        return torch.zeros((1, 24000), device=tokens.device)


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
        # Create a MimiModel instance with the config
        try:
            # First try direct import
            from moshi.models.seanet import MimiModel
            mimi_model = MimiModel(**config).to(device)
        except ImportError:
            # If direct import fails, try to dynamically find the module
            import importlib
            seanet_module = None
            mimi_model = None
            
            # Try different potential import paths
            potential_paths = [
                'moshi.models.seanet',
                'moshi.seanet',
                'moshi.models.mimi'
            ]
            
            for path in potential_paths:
                try:
                    seanet_module = importlib.import_module(path)
                    MimiClass = getattr(seanet_module, 'MimiModel')
                    mimi_model = MimiClass(**config).to(device)
                    print(f"Successfully imported MimiModel from {path}")
                    break
                except (ImportError, AttributeError):
                    continue
            
            if mimi_model is None:
                print("Failed to import MimiModel through dynamic imports")
                print("Using standard moshi.models.loaders API")
                
                # Use the standard moshi loaders API
                mimi_model = loaders.get_mimi(model_path, device=device)
        
        # Load weights from safetensors file
        print(f"Loading mimi weights from safetensors file...")
        state_dict = load_file(model_path, device=device)
        
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
        
        print("Successfully loaded local mimi model")
        return mimi_model
    except Exception as e:
        print(f"Error loading local mimi model: {str(e)}")
        print(traceback.format_exc())
        
        # Return the stub implementation as a fallback
        print("Returning MimiModelStub as fallback")
        return MimiModelStub()


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        # Load tokenizer
        print("Loading text tokenizer...")
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
                    try:
                        mimi_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
                        print(f"Loading downloaded mimi weights from: {mimi_path}")
                        mimi = loaders.get_mimi(mimi_path, device=device)
                        print("Successfully loaded mimi model from downloaded file")
                    except Exception as e:
                        print(f"Error downloading from moshiko: {e}")
                        print("Trying kyutai/mimi repository...")
                        
                        mimi_path = hf_hub_download("kyutai/mimi", "model.safetensors")
                        print(f"Loading downloaded mimi weights from: {mimi_path}")
                        mimi = load_local_mimi(device, os.path.dirname(mimi_path))
                        
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
                    try:
                        mimi = loaders.get_mimi(None, device=device)
                    except Exception as loader_err:
                        print(f"Error with default loader: {loader_err}")
                        print("Using stub implementation as last resort")
                        mimi = MimiModelStub()
                
                # Restore offline mode
                if original_hf_offline:
                    os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
            
            # Set codebooks and assign as audio tokenizer
            if hasattr(mimi, 'set_num_codebooks'):
                mimi.set_num_codebooks(32)
            self._audio_tokenizer = mimi
            
        except Exception as e:
            print(f"Error during mimi initialization: {str(e)}")
            print(traceback.format_exc())
            print("Using stub implementation as last resort")
            self._audio_tokenizer = MimiModelStub()
        
        # Load watermarker
        try:
            print("Loading watermarker...")
            # Look for a local watermarker file
            watermarker_path = os.path.join("models", "csm-1b", "watermarker.pt")
            
            if os.path.exists(watermarker_path):
                try:
                    from watermarking import load_watermarker
                    print(f"Loading watermarker from local path: {watermarker_path}")
                    self._watermarker = load_watermarker(device=device, ckpt_path=watermarker_path)
                except ImportError:
                    print("Watermarking module not available")
                    self._watermarker = None
            else:
                # Skip watermarking if file not available
                print("Watermarker file not found. Watermarking will be disabled.")
                self._watermarker = None
        except Exception as e:
            print(f"Error loading watermarker: {str(e)}")
            print(traceback.format_exc())
            print("Setting watermarker to None - watermarking will be skipped")
            self._watermarker = None

        # Set sample rate - default to 24000 if not available from mimi
        if hasattr(self._audio_tokenizer, 'sample_rate'):
            self.sample_rate = self._audio_tokenizer.sample_rate
        else:
            self.sample_rate = 24000
        
        self.device = device
        print(f"Generator initialized with sample rate: {self.sample_rate}")

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
        try:
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
        except Exception as e:
            print(f"Error in _tokenize_audio: {e}")
            print(traceback.format_exc())
            
            # Return empty tensors as fallback
            empty_frame = torch.zeros(1, 33).long().to(self.device)
            empty_mask = torch.zeros(1, 33).bool().to(self.device)
            return empty_frame, empty_mask

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
        try:
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
                try:
                    from watermarking import watermark, CSM_1B_GH_WATERMARK
                    # This applies an imperceptible watermark to identify audio as AI-generated.
                    # Watermarking ensures transparency, dissuades misuse, and enables traceability.
                    # Please be a responsible AI citizen and keep the watermarking in place.
                    # If using CSM 1B in another application, use your own private key and keep it secret.
                    audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
                    audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
                except Exception as e:
                    print(f"Error during watermarking: {e}")
                    print("Continuing without watermark")
            else:
                print("Skipping watermarking - watermarker not available")

            return audio
        except Exception as e:
            print(f"Error in generate: {e}")
            print(traceback.format_exc())
            
            # Return silent audio as fallback
            return torch.zeros(self.sample_rate).to(self.device)


def load_csm_1b(ckpt_path: str = "ckpt.pt", device: str = "cuda", config_path: str = None) -> Generator:
    """
    Load the CSM model from the given path.
    
    Args:
        ckpt_path: Path to the model weights file
        device: Device to load the model on ('cuda' or 'cpu')
        config_path: Optional path to a config.json file
    """
    print(f"\n===== LOADING CSM MODEL =====")
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
    return generator
