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

# Add the current directory to the path to import the generator module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the CSM model functions
try:
    from generator import load_csm_1b, Segment
except ImportError:
    # If the generator module is not in the current directory, try to find it in the csm repository
    print("Could not import generator module directly. Make sure you've cloned the CSM repository.")
    print("Attempting to import from csm directory...")
    sys.path.append(os.path.join(current_dir, "csm"))
    from generator import load_csm_1b, Segment

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

def load_model(model_path):
    """Load the CSM model from the given path."""
    global generator
    try:
        if not os.path.exists(model_path):
            return f"Error: Model file not found at {model_path}"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = load_csm_1b(model_path, device)
        return f"Model loaded successfully on {device}."
    except Exception as e:
        return f"Error loading model: {str(e)}"

def download_model(output_path):
    """Download the CSM model from Hugging Face."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # URL to download the model from Hugging Face
        model_url = "https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors?download=true"
        
        # Download the model
        print(f"Downloading CSM model to {output_path}...")
        response = requests.get(model_url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            return f"Model downloaded successfully to {output_path}."
        else:
            # Check for 401 Unauthorized error which may indicate lack of access
            if response.status_code == 401:
                return "Error: Unauthorized access. Make sure you have requested and been granted access to the model on Hugging Face."
            else:
                return f"Error downloading model: HTTP status code {response.status_code}"
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

# Define the Gradio interface
with gr.Blocks(title="CSM-WebUI") as app:
    # Use HTML directly for the title to ensure proper rendering and linking, with adjusted font size
    gr.HTML("""
    <div style="text-align: center; margin: 20px 0;">
        <a href="https://github.com/Saganaki22/CSM-WebUI" style="text-decoration: none; color: inherit;">
            <h1 style="font-size: 2.3rem; font-weight: bold; margin: 0;">CSM-WebUI</h1>
        </a>
    </div>
    """)
    
    gr.Markdown("""
    CSM (Conversational Speech Model) is a speech generation model from Sesame that generates RVQ audio codes from text and audio inputs.
    
    This interface allows you to generate speech from text, with optional conversation context.
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
            value="models/model.safetensors"
        )
        with gr.Column():
            load_button = gr.Button("Load Model")
            download_button = gr.Button("Download Model")
        model_status = gr.Textbox(label="Model Status", interactive=False)
    
    gr.Markdown("""
    **Note:** Make sure you request access to the model on [Hugging Face](https://huggingface.co/sesame/csm-1b) first or you won't be able to download it.
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
                        value="like revising for an exam I'd have to try and like keep up the momentum because I'd start really early I'd be like okay I'm gonna start revising now and then like you're revising for ages and then I just like start losing steam I didn't do that for the exam we had recently to be fair that was a more of a last minute scenario but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I sort of started my day I sort of just like get a bit when I start",
                        lines=5
                    )
                
                speaker_a_audio = gr.Audio(
                    label="Speaker prompt",
                    type="filepath",
                    value="sounds/woman.mp3"  # Default to woman.mp3
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
                        value="like a super Mario level. Like it's very like high detail. And like, once you get into the park, it just like, everything looks like a computer game and they have all these, like, you know, it. If there's like a you know, like is a Mario game, they will have like a question block. And if you hit you know, and it's just like, it's just like for like the everyone, when they come into the park, they get like this little bracelet and then you can go punching question blocks around.",
                        lines=5
                    )
                
                speaker_b_audio = gr.Audio(
                    label="Speaker prompt",
                    type="filepath",
                    value="sounds/man.mp3"  # Default to man.mp3
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
    """, elem_classes=["left-aligned"])
    
    # Add official links section without bullet points
    gr.Markdown("""
    ### Links
    
    [GitHub Repository](https://github.com/Saganaki22/CSM-WebUI) [CSM Repository](https://github.com/SesameAILabs/csm)   [Official Hugging Face Repository](https://huggingface.co/sesame/csm-1b)  [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
    """, elem_classes=["center-aligned", "no-bullets"])

# Launch the app
if __name__ == "__main__":
    app.launch()
