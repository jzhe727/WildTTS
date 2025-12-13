"""
FastAPI Demo - Audio File Communication Backend for Unity Integration

Unity Functions Supported:
1. Record voice -> Send to Python for saving reference audio and text
2. Play recorded voice (local, no server communication)
3. Send text -> Get synthesized voice (text-to-speech with reference audio) -> Play synthesized voice
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from pathlib import Path
import shutil
import uvicorn
import os
import json
from datetime import datetime
from typing import Optional

from tts import create_tts, BaseTTS

app = FastAPI(
    title="Audio API Demo",
    description="Backend API for audio file communication with Unity",
    version="3.0.0"
)

# Directories for audio files
UPLOAD_DIR = Path("uploaded_audio")
SYNTHESIZED_DIR = Path("synthesized_audio")
REFERENCE_DIR = Path("user_references")
UPLOAD_DIR.mkdir(exist_ok=True)
SYNTHESIZED_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

# TTS Engine Configuration
# Set to 'cosyvoice' or 'fishaudio' based on your setup
TTS_ENGINE = os.environ.get("TTS_ENGINE", "cosyvoice")
TTS_MODEL_PATH = os.environ.get("TTS_MODEL_PATH", "CosyVoice/pretrained_models/CosyVoice2-0.5B")
FISHAUDIO_API_KEY = os.environ.get("FISHAUDIO_API_KEY", None)

# Initialize TTS engines (lazy loading) - cache for each engine type
_tts_engines: dict[str, BaseTTS] = {}

# Supported TTS models
SUPPORTED_MODELS = ["cosyvoice", "fishaudio", "fishaudio_enhance"]


def get_tts_engine(model: Optional[str] = None) -> BaseTTS:
    """
    Get or initialize a TTS engine (lazy loading with caching).
    
    Args:
        model: TTS model name ('cosyvoice' or 'fishaudio'). 
               If None, uses the default TTS_ENGINE from environment.
    
    Returns:
        BaseTTS engine instance
    """
    global _tts_engines
    
    # Use default engine if not specified
    engine_name = model.lower() if model else TTS_ENGINE
    
    # Validate engine name
    if engine_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported TTS model: {engine_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    
    # Return cached engine if available
    if engine_name in _tts_engines:
        return _tts_engines[engine_name]
    
    # Initialize new engine
    if engine_name == "cosyvoice":
        _tts_engines[engine_name] = create_tts(
            "cosyvoice",
            model_path=TTS_MODEL_PATH,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
    elif engine_name == "fishaudio" or engine_name == "fishaudio_enhance":
        # Both fishaudio and fishaudio_enhance use the same FishAudio engine
        # The difference is in the synthesize call parameters
        base_engine = "fishaudio"
        if base_engine not in _tts_engines:
            _tts_engines[base_engine] = create_tts(
                "fishaudio",
                api_key=FISHAUDIO_API_KEY
            )
        # Cache the same engine instance for fishaudio_enhance
        _tts_engines[engine_name] = _tts_engines[base_engine]
    
    return _tts_engines[engine_name]


# ============== USER REFERENCE MANAGEMENT ==============
def save_user_reference(
    audio_data: bytes,
    user_id: str,
    reference_text: str,
    file_extension: str = ".wav"
) -> dict:
    """
    Save user's reference audio and text for voice cloning.
    
    Args:
        audio_data: Raw bytes of the user's voice recording
        user_id: Unique identifier for the user
        reference_text: Transcript of the reference audio (required for TTS)
        file_extension: Audio file extension
    
    Returns:
        Dictionary containing reference data and metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save reference audio file
    audio_filename = f"{user_id}_reference_{timestamp}{file_extension}"
    audio_path = REFERENCE_DIR / audio_filename
    with open(audio_path, "wb") as f:
        f.write(audio_data)
    
    # Create reference metadata
    reference_data = {
        "user_id": user_id,
        "audio_filename": audio_filename,
        "audio_path": str(audio_path.absolute()),
        "reference_text": reference_text,
        "audio_size_bytes": len(audio_data),
        "created_at": datetime.now().isoformat()
    }
    
    # Save reference metadata as JSON
    metadata_filename = f"{user_id}_reference_{timestamp}.json"
    metadata_path = REFERENCE_DIR / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(reference_data, f, indent=2, ensure_ascii=False)
    
    return reference_data


def get_user_reference(user_id: str) -> Optional[dict]:
    """
    Get the most recent reference data for a user.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        Dictionary with reference data or None if not found
    """
    reference_files = list(REFERENCE_DIR.glob(f"{user_id}_reference_*.json"))
    
    if not reference_files:
        return None
    
    # Get the most recent one
    latest_file = max(reference_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ============== AUDIO SYNTHESIS ==============
def synthesize_audio(
    text: str,
    user_reference: dict = None,
    model: Optional[str] = None
) -> tuple[bytes, str, str]:
    """
    Synthesize speech from text using the TTS engine.
    
    Uses the user's reference audio for voice cloning.
    
    Args:
        text: The text to be synthesized into speech
        user_reference: User reference data containing audio path and text
        model: TTS model name ('cosyvoice' or 'fishaudio'). Uses default if None.
    
    Returns:
        Tuple of (synthesized_audio_bytes, output_filename, model_used)
    
    Raises:
        ValueError: If user_reference is missing required data
    """
    if user_reference is None:
        raise ValueError("User reference is required for synthesis")
    
    # Validate reference data
    reference_audio_path = user_reference.get("audio_path")
    reference_text = user_reference.get("reference_text")
    
    if not reference_audio_path or not os.path.exists(reference_audio_path):
        raise ValueError(f"Reference audio not found: {reference_audio_path}")
    
    if not reference_text:
        raise ValueError("Reference text is required for TTS synthesis")
    
    # Determine which model to use
    model_used = model.lower() if model else TTS_ENGINE
    
    # Generate output filename with timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_prefix = user_reference.get("user_id", "default")
    output_filename = f"{user_prefix}_{model_used}_synthesized_{timestamp}.wav"
    output_path = SYNTHESIZED_DIR / output_filename
    
    # Get TTS engine and synthesize
    tts_engine = get_tts_engine(model=model)
    
    # Prepare synthesize kwargs
    synthesize_kwargs = {
        "text": text,
        "prompt_wav_path": reference_audio_path,
        "output_wav_path": str(output_path),
        "style_text": reference_text
    }
    
    # Add enhance_audio_quality for fishaudio_enhance model
    if model_used == "fishaudio_enhance":
        synthesize_kwargs["enhance_audio_quality"] = True
    
    tts_engine.synthesize(**synthesize_kwargs)
    
    # Read the synthesized audio
    with open(output_path, "rb") as f:
        synthesized_data = f.read()
    
    return synthesized_data, output_filename, model_used


# ============== ENDPOINT 1: SAVE USER REFERENCE ==============
@app.post("/audio/reference")
async def create_user_reference(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...),
    user_id: str = Form("default_user")
):
    """
    Receives user's voice recording and reference text for voice cloning.
    
    Unity Function: Record user's voice -> Send to Python with transcript
    
    Args:
        audio_file: The user's voice recording (passed via form-data)
        reference_text: Transcript of what is spoken in the audio (form field)
        user_id: Unique identifier for the user (form field)
    
    Returns:
        JSON response with reference data and status
    """
    try:
        # Validate file type
        allowed_extensions = {".m4a", ".mp3", ".wav", ".ogg", ".aac"}
        file_extension = Path(audio_file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Validate reference text
        if not reference_text or not reference_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Reference text (transcript of the audio) is required"
            )
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Save reference audio and metadata
        reference_data = save_user_reference(
            audio_data=audio_data,
            user_id=user_id,
            reference_text=reference_text.strip(),
            file_extension=file_extension
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "User reference saved successfully",
                "user_id": user_id,
                "audio_filename": reference_data["audio_filename"],
                "reference_text": reference_data["reference_text"],
                "audio_size_bytes": reference_data["audio_size_bytes"],
                "created_at": reference_data["created_at"]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reference saving failed: {str(e)}")


@app.get("/audio/reference/{user_id}")
async def get_user_reference_endpoint(user_id: str):
    """
    Retrieves the most recent reference data for a user.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        JSON response with the user's reference data
    """
    reference_data = get_user_reference(user_id)
    
    if not reference_data:
        raise HTTPException(
            status_code=404,
            detail=f"No reference found for user '{user_id}'"
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "reference_data": reference_data
        }
    )


# ============== ENDPOINT 2: SYNTHESIZE AND RETURN ==============
@app.post("/audio/synthesize")
async def synthesize_and_return(
    text: str = Form(...),
    user_id: str = Form(...),
    model: Optional[str] = Form(None)
):
    """
    Text-to-speech synthesis endpoint: Receives text, synthesizes speech using user's reference audio.
    
    Unity Function: Request synthesized voice -> Play synthesized voice
    
    Args:
        text: The text to be synthesized into speech (form field)
        user_id: User ID to use their reference audio for voice cloning (required)
        model: TTS model name ('cosyvoice' or 'fishaudio'). Uses default if not specified.
    
    Returns:
        The synthesized audio file as binary response
    """
    try:
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text input is required for synthesis"
            )
        
        # Validate model if provided
        if model and model.lower() not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model}. Supported models: {', '.join(SUPPORTED_MODELS)}"
            )
        
        # Load user reference (required for voice cloning)
        user_reference = get_user_reference(user_id)
        if not user_reference:
            raise HTTPException(
                status_code=404,
                detail=f"No reference found for user '{user_id}'. Please upload a reference audio first."
            )
        
        # Synthesize audio from text using user's reference
        synthesized_data, output_filename, model_used = synthesize_audio(
            text=text.strip(),
            user_reference=user_reference,
            model=model
        )
        
        # Determine media type based on output filename
        file_extension = Path(output_filename).suffix.lower()
        media_types = {
            ".m4a": "audio/mp4",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".aac": "audio/aac"
        }
        media_type = media_types.get(file_extension, "audio/wav")
        
        # Return synthesized audio directly
        return Response(
            content=synthesized_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Synthesized-Filename": output_filename,
                "X-Input-Text": text[:100],  # First 100 chars of text
                "X-Output-Size": str(len(synthesized_data)),
                "X-User-Id": user_id,
                "X-Model-Used": model_used
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


# ============== UPLOAD & DOWNLOAD ENDPOINTS ==============
@app.post("/audio/upload")
async def receive_audio_file(audio_file: UploadFile = File(...)):
    """
    Receives an audio file from client (Unity).
    """
    try:
        allowed_extensions = {".m4a", ".mp3", ".wav", ".ogg", ".aac"}
        file_extension = Path(audio_file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        file_path = UPLOAD_DIR / audio_file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Audio file uploaded successfully",
                "filename": audio_file.filename,
                "size_bytes": os.path.getsize(file_path),
                "path": str(file_path)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/audio/download/{filename}")
async def send_audio_file(filename: str):
    """
    Sends an audio file to client (Unity).
    """
    # Check all directories
    for directory in [UPLOAD_DIR, SYNTHESIZED_DIR]:
        file_path = directory / filename
        if file_path.exists():
            break
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Audio file '{filename}' not found"
        )
    
    media_types = {
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".aac": "audio/aac"
    }
    
    file_extension = Path(filename).suffix.lower()
    media_type = media_types.get(file_extension, "application/octet-stream")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


@app.get("/audio/list")
async def list_audio_files():
    """
    Lists all available audio files on the server.
    """
    uploaded_files = []
    synthesized_files = []
    reference_files = []
    
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            uploaded_files.append({
                "filename": file_path.name,
                "size_bytes": os.path.getsize(file_path)
            })
    
    for file_path in SYNTHESIZED_DIR.iterdir():
        if file_path.is_file():
            synthesized_files.append({
                "filename": file_path.name,
                "size_bytes": os.path.getsize(file_path)
            })
    
    for file_path in REFERENCE_DIR.iterdir():
        if file_path.is_file():
            reference_files.append({
                "filename": file_path.name,
                "size_bytes": os.path.getsize(file_path)
            })
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "uploaded_files": uploaded_files,
            "synthesized_files": synthesized_files,
            "reference_files": reference_files,
            "total_count": len(uploaded_files) + len(synthesized_files) + len(reference_files)
        }
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Audio API Demo is active",
        "version": "4.1.0",
        "default_tts_engine": TTS_ENGINE,
        "supported_models": SUPPORTED_MODELS,
        "unity_functions": {
            "1_save_reference": "POST /audio/reference - Upload reference audio + text for voice cloning",
            "2_playback_local": "No server communication needed",
            "3_synthesize_and_play": "POST /audio/synthesize - Send text + model, get synthesized audio"
        },
        "other_endpoints": {
            "get_reference": "GET /audio/reference/{user_id}",
            "get_models": "GET /audio/models - List supported TTS models",
            "upload": "POST /audio/upload",
            "download": "GET /audio/download/{filename}",
            "list": "GET /audio/list"
        }
    }


@app.get("/audio/models")
async def get_supported_models():
    """
    Returns the list of supported TTS models.
    
    Returns:
        JSON with supported models and default model
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "default_model": TTS_ENGINE,
            "supported_models": SUPPORTED_MODELS,
            "model_info": {
                "cosyvoice": {
                    "name": "CosyVoice2",
                    "type": "local",
                    "description": "CosyVoice2-0.5B local TTS model"
                },
                "fishaudio": {
                    "name": "FishAudio",
                    "type": "cloud",
                    "description": "FishAudio cloud-based TTS API"
                },
                "fishaudio_enhance": {
                    "name": "FishAudio Enhanced",
                    "type": "cloud",
                    "description": "FishAudio cloud-based TTS API with enhanced audio quality"
                }
            }
        }
    )


# ============== TEST CODE ==============
def test_save_reference(file_path: Path, user_id: str, reference_text: str):
    """
    Test function to save user reference audio and text.
    
    Args:
        file_path: Path to the voice recording file
        user_id: User identifier
        reference_text: Transcript of the reference audio
    """
    import requests
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None
    
    url = "http://localhost:8000/audio/reference"
    
    with open(file_path, "rb") as audio_file:
        files = {"audio_file": (file_path.name, audio_file, "audio/wav")}
        data = {
            "user_id": user_id,
            "reference_text": reference_text
        }
        response = requests.post(url, files=files, data=data)
    
    print(f"Reference Save Response Status: {response.status_code}")
    print(f"Reference Save Response: {response.json()}")
    return response


def test_get_reference(user_id: str):
    """
    Test function to retrieve user reference.
    """
    import requests
    
    url = f"http://localhost:8000/audio/reference/{user_id}"
    response = requests.get(url)
    
    print(f"Get Reference Response Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"User ID: {data['reference_data']['user_id']}")
        print(f"Reference Text: {data['reference_data']['reference_text']}")
        print(f"Audio File: {data['reference_data']['audio_filename']}")
    else:
        print(f"Response: {response.json()}")
    return response


def test_synthesize_with_user(text: str, output_file_path: Path, user_id: str, model: str = None):
    """
    Test text-to-speech synthesis with user reference.
    
    Args:
        text: Text to synthesize
        output_file_path: Path to save the output audio
        user_id: User ID
        model: Optional TTS model name ('cosyvoice' or 'fishaudio')
    """
    import requests
    
    url = "http://localhost:8000/audio/synthesize"
    data = {
        "text": text,
        "user_id": user_id
    }
    if model:
        data["model"] = model
    
    print(f"Sending text for synthesis (user: {user_id}, model: {model or 'default'}): '{text[:50]}...'")
    
    response = requests.post(url, data=data)
    
    print(f"Synthesis Response Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"User ID used: {response.headers.get('X-User-Id', 'none')}")
        print(f"Model used: {response.headers.get('X-Model-Used', 'N/A')}")
        print(f"Input text: {response.headers.get('X-Input-Text', 'N/A')}")
        print(f"Output size: {response.headers.get('X-Output-Size', 'N/A')} bytes")
        with open(output_file_path, "wb") as f:
            f.write(response.content)
        print(f"Synthesized audio saved to: {output_file_path}")
    else:
        print(f"Synthesis failed: {response.text}")
    
    return response


def test_list_files():
    """Test function to list all files on the server."""
    import requests
    
    url = "http://localhost:8000/audio/list"
    response = requests.get(url)
    
    print(f"List Response Status: {response.status_code}")
    data = response.json()
    print(f"Uploaded files: {len(data.get('uploaded_files', []))}")
    print(f"Synthesized files: {len(data.get('synthesized_files', []))}")
    print(f"Reference files: {len(data.get('reference_files', []))}")
    return response


def run_tests():
    """
    Run all test functions simulating Unity workflow
    """
    import time
    
    print("=" * 70)
    print("Running API Tests - Unity Workflow Simulation")
    print("=" * 70)
    
    # Test configuration - update these paths as needed
    sample_file = Path("zero_shot_0.wav")  # Reference audio file
    test_user_id = "unity_test_user"
    synthesized_output = Path("synthesized_output.wav")
    
    # Reference text should match what is spoken in the reference audio
    reference_text = "This is a sample reference audio for voice cloning."
    
    # Text to synthesize
    test_text = "Hello, this is a test of the text-to-speech synthesis system."
    
    # Unity Function 1: Record voice -> Save reference audio and text
    print("\n" + "-" * 70)
    print("[UNITY FUNCTION 1] Record Voice -> Save Reference Audio + Text")
    print("-" * 70)
    test_save_reference(
        file_path=sample_file,
        user_id=test_user_id,
        reference_text=reference_text
    )
    
    time.sleep(0.5)
    
    # Verify reference was saved
    print("\n[VERIFY] Retrieving saved reference...")
    test_get_reference(user_id=test_user_id)
    
    time.sleep(0.5)
    
    # Unity Function 2: Play recorded voice (local - no server test needed)
    print("\n" + "-" * 70)
    print("[UNITY FUNCTION 2] Play Recorded Voice")
    print("-" * 70)
    print("(No server communication needed - handled locally in Unity)")
    
    time.sleep(0.5)
    
    # Unity Function 3: Send text -> Get synthesized voice and play
    print("\n" + "-" * 70)
    print("[UNITY FUNCTION 3] Send Text -> Request Synthesized Voice -> Play")
    print("-" * 70)
    test_synthesize_with_user(
        text=test_text,
        output_file_path=synthesized_output,
        user_id=test_user_id
    )
    
    time.sleep(0.5)
    
    # List all files
    print("\n" + "-" * 70)
    print("[SUMMARY] All files on server")
    print("-" * 70)
    test_list_files()
    
    print("\n" + "=" * 70)
    print("Tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        print("Starting FastAPI Audio Server...")
        print("API Documentation available at: http://localhost:8000/docs")
        print("To run tests, use: python FastAPIdemo.py test")
        uvicorn.run(app, host="0.0.0.0", port=8000)