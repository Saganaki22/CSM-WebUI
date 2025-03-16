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

# Add the current directory to the path to import the generator module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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

        self._text_tokenizer = self.load_llama3_tokenizer()

        device = next(model.parameters()).device
        
        # Try multiple approaches to load the mimi weights
        try:
            print("Attempting to load mimi weights...")
            mimi = None
            
            # First check the models/mimi directory
            local_mimi_path = os.path.join("models", "mimi", "model.safetensors")
            if os.path.exists(local_mimi_path):
                try:
                    print(f"Loading mimi from local path: {local_mimi_path}")
                    # Temporarily disable offline mode
                    original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
                    os.environ['HF_DATASETS_OFFLINE'] = '0'
                    
                    # Try standard loader with local file
                    from moshi.models import loaders
                    mimi = loaders.get_mimi(local_mimi_path, device=device)
                    print("Successfully loaded local mimi model")
                    
                    # Restore offline mode
                    if original_hf_offline:
                        os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
                except Exception as e:
                    print(f"Error loading local mimi: {str(e)}")
                    mimi = None
            
            # Next check CSM-1B folder
            if mimi is None:
                csm_mimi_path = os.path.join("models", "csm-1b", "mimi.bin")
                if os.path.exists(csm_mimi_path):
                    try:
                        print(f"Loading mimi from CSM-1B folder: {csm_mimi_path}")
                        # Temporarily disable offline mode
                        original_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
                        os.environ['HF_DATASETS_OFFLINE'] = '0'
                        
                        from moshi.models import loaders
                        mimi = loaders.get_mimi(csm_mimi_path, device=device)
                        print("Successfully loaded mimi from CSM-1B folder")
                        
                        # Restore offline mode
                        if original_hf_offline:
                            os.environ['HF_DATASETS_OFFLINE'] = original_hf_offline
                    except Exception as e:
                        print(f"Error loading mimi from CSM-1B folder: {str(e)}")
                        mimi = None
            
            # As a last resort, download from Hugging Face
            if mimi is None:
                print("Local mimi files not found or failed to load. Downloading from Hugging Face...")
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
                except Exception as e:
                    print(f"Error downloading mimi file: {str(e)}")
                    print("Falling back to default download...")
                    mimi = loaders.get_mimi(None, device=device)
                    
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
    
except ImportError:
    # If the generator module is not in the current directory, try to find it in the csm repository
    print("Could not import generator module directly. Make sure you've cloned the CSM repository.")
    print("Attempting to import from csm directory...")
    sys.path.append(os.path.join(current_dir, "csm"))
    try:
        from generator import load_csm_1b, Segment, debug_path_check
    except ImportError:
        print("Failed to import from csm directory as well.")
        print("Please ensure the generator.py file is in the current directory or in a 'csm' subdirectory.")
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
        
        # Check for mimi.bin in the same directory
        mimi_path = os.path.join(os.path.dirname(model_path), "mimi.bin")
        debug_path_check(mimi_path)
        
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
                    choices=["0", "1", "2", "3", "4"],
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
                    choices=["0", "1", "2", "3", "4"],
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
                    choices=["0", "1", "2", "3", "4"],
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
                    choices=["0", "1", "2", "3", "4"],
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
                    choices=["0", "1", "2", "3", "4"],
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
                    choices=["0", "1", "2", "3", "4"],
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
    - Speaker IDs (0-4) represent different voices the model can generate.
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
