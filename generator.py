from dataclasses import dataclass
from typing import List, Tuple
import os

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
import traceback

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
        print(f"Test tokenization of '{test_text}':")
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


def download_mimi_directly():
    """Download the mimi file directly from Hugging Face."""
    try:
        # Temporarily disable offline mode
        original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
        os.environ['HF_DATASETS_OFFLINE'] = '0'
        
        print("Downloading mimi file from Hugging Face...")
        # The exact name might differ for the bin format vs safetensors format
        mimi_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors")
        
        # Restore offline mode
        if original_hf_offline:
            os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
            
        print(f"Successfully downloaded mimi file to: {mimi_path}")
        return mimi_path
    except Exception as e:
        print(f"Error downloading mimi: {str(e)}")
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
        
        # Try multiple approaches to load the mimi weights
        try:
            print("Attempting to load mimi weights...")
            mimi = None
            
            # First try downloading directly - this is most likely to work
            mimi_path = download_mimi_directly()
            
            if mimi_path:
                try:
                    print(f"Loading downloaded mimi weights from: {mimi_path}")
                    mimi = loaders.get_mimi(mimi_path, device=device)
                    print("Successfully loaded mimi model from downloaded file")
                except Exception as e:
                    print(f"Error loading downloaded mimi: {str(e)}")
                    mimi = None
            
            # If still no mimi, try local paths
            if mimi is None:
                # Check in models/mimi directory
                local_mimi_path = os.path.join("models", "mimi", "model.safetensors")
                
                if os.path.exists(local_mimi_path):
                    try:
                        # Temporarily disable offline mode
                        original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
                        os.environ['HF_DATASETS_OFFLINE'] = '0'
                        
                        print(f"Trying to load mimi from local path: {local_mimi_path}")
                        # Skip strict loading which would reject mismatched keys
                        from moshi.models.seanet import MimiModel
                        import json
                        
                        # Try to load config if available
                        config_path = os.path.join("models", "mimi", "config.json")
                        config = {}
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                        
                        # Create model instance
                        mimi_model = MimiModel(**config).to(device)
                        
                        # Load state dict with strict=False to ignore mismatches
                        from safetensors.torch import load_file
                        state_dict = load_file(local_mimi_path, device=device)
                        mimi_model.load_state_dict(state_dict, strict=False)
                        
                        # Set up as mimi
                        mimi = mimi_model
                        # Standard sample rate for mimi
                        setattr(mimi, 'sample_rate', 24000)
                        print("Successfully loaded local mimi model with non-strict loading")
                        
                        # Restore offline mode
                        if original_hf_offline:
                            os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
                    except Exception as local_e:
                        print(f"Error loading local mimi: {str(local_e)}")
                        print(traceback.format_exc())
                        mimi = None
                
                # If still no mimi, fall back to standard download
                if mimi is None:
                    print("All local loading attempts failed. Falling back to standard download...")
                    # Temporarily disable offline mode
                    original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
                    os.environ['HF_DATASETS_OFFLINE'] = '0'
                    
                    mimi = loaders.get_mimi(None, device=device)  # Let it download the default
                    
                    # Restore offline mode
                    if original_hf_offline:
                        os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
            
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
    return generator
