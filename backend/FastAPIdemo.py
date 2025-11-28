"""
FastAPI Demo - Audio File Communication Backend for Unity Integration

Unity Functions Supported:
1. Record voice -> Send to Python for user embedding generation
2. Play recorded voice (local, no server communication)
3. Send text -> Get synthesized voice (text-to-speech with user embeddings) -> Play synthesized voice
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from pathlib import Path
import shutil
import uvicorn
import os
import json
from datetime import datetime

app = FastAPI(
    title="Audio API Demo",
    description="Backend API for audio file communication with Unity",
    version="3.0.0"
)

# Directories for audio files
UPLOAD_DIR = Path("uploaded_audio")
SYNTHESIZED_DIR = Path("synthesized_audio")
EMBEDDINGS_DIR = Path("user_embeddings")
UPLOAD_DIR.mkdir(exist_ok=True)
SYNTHESIZED_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)


# ============== USER EMBEDDING GENERATION (PLACEHOLDER) ==============
def generate_user_embedding(audio_data: bytes, user_id: str) -> dict:
    """
    Placeholder function for generating user voice embedding.
    
    Args:
        audio_data: Raw bytes of the user's voice recording
        user_id: Unique identifier for the user
    
    Returns:
        Dictionary containing embedding data and metadata
    
    TODO: Replace this placeholder with actual embedding extraction
    (e.g., speaker encoder, voice fingerprint model, etc.)
    """
    # ========================================
    # PLACEHOLDER: Returns dummy embedding data
    # Replace this section with your embedding extraction code:
    # 
    # Example integration points:
    # - Speaker encoder models (e.g., GE2E, ECAPA-TDNN)
    # - Voice fingerprint extraction
    # - Custom voice feature extraction
    # ========================================
    
    # Placeholder: Generate dummy embedding (256-dimensional vector)
    import hashlib
    hash_obj = hashlib.sha256(audio_data)
    hash_hex = hash_obj.hexdigest()
    
    # Create a simple deterministic "embedding" from audio hash
    embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, 64, 2)]
    
    embedding_data = {
        "user_id": user_id,
        "embedding": embedding,
        "embedding_dim": len(embedding),
        "audio_size_bytes": len(audio_data),
        "created_at": datetime.now().isoformat(),
        "model_version": "placeholder_v1"
    }
    
    return embedding_data


# ============== AUDIO SYNTHESIS (PLACEHOLDER) ==============
def synthesize_audio(text: str, user_embedding: dict = None) -> tuple[bytes, str]:
    """
    Placeholder function for text-to-speech synthesis.
    
    Takes text input and reads it aloud using user embeddings for voice cloning.
    
    Args:
        text: The text to be synthesized into speech
        user_embedding: Optional user embedding for voice cloning
    
    Returns:
        Tuple of (synthesized_audio_bytes, output_filename)
    
    TODO: Replace this placeholder with actual TTS synthesis logic
    """
    # ========================================
    # PLACEHOLDER: Currently returns empty audio data
    # Replace this section with your TTS synthesis code:
    # 
    # Example integration points:
    # - Text-to-Speech with voice cloning (e.g., Coqui TTS, VALL-E, XTTS)
    # - Use user_embedding to clone the user's voice
    # - Generate audio from text input
    # ========================================
    
    # Placeholder: Generate empty audio data
    # In real implementation, this would use TTS model with user_embedding
    synthesized_data = b""  # Placeholder: empty audio
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_prefix = user_embedding.get("user_id", "default") if user_embedding else "default"
    output_filename = f"{user_prefix}_synthesized_{timestamp}.wav"
    
    return synthesized_data, output_filename


# ============== ENDPOINT 1: GENERATE USER EMBEDDING ==============
@app.post("/audio/embedding")
async def create_user_embedding(
    audio_file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """
    Receives user's voice recording and generates a voice embedding.
    
    Unity Function: Record user's voice -> Send to Python for embedding
    
    Args:
        audio_file: The user's voice recording (passed via form-data)
        user_id: Unique identifier for the user (query parameter)
    
    Returns:
        JSON response with embedding data and status
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
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Save the voice recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{user_id}_voice_{timestamp}{file_extension}"
        saved_path = UPLOAD_DIR / saved_filename
        with open(saved_path, "wb") as f:
            f.write(audio_data)
        
        # Generate embedding
        embedding_data = generate_user_embedding(
            audio_data=audio_data,
            user_id=user_id
        )
        
        # Save embedding to file
        embedding_filename = f"{user_id}_embedding_{timestamp}.json"
        embedding_path = EMBEDDINGS_DIR / embedding_filename
        with open(embedding_path, "w") as f:
            json.dump(embedding_data, f, indent=2)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "User embedding generated successfully",
                "user_id": user_id,
                "embedding_file": embedding_filename,
                "voice_file": saved_filename,
                "embedding_dim": embedding_data["embedding_dim"],
                "audio_size_bytes": len(audio_data)
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.get("/audio/embedding/{user_id}")
async def get_user_embedding(user_id: str):
    """
    Retrieves the most recent embedding for a user.
    
    Args:
        user_id: The user's unique identifier
    
    Returns:
        JSON response with the user's embedding data
    """
    # Find the most recent embedding file for this user
    embedding_files = list(EMBEDDINGS_DIR.glob(f"{user_id}_embedding_*.json"))
    
    if not embedding_files:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for user '{user_id}'"
        )
    
    # Get the most recent one
    latest_file = max(embedding_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, "r") as f:
        embedding_data = json.load(f)
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "embedding_data": embedding_data
        }
    )


# ============== ENDPOINT 2: SYNTHESIZE AND RETURN ==============
@app.post("/audio/synthesize")
async def synthesize_and_return(
    text: str = Form(...),
    user_id: str = Form(None)
):
    """
    Text-to-speech synthesis endpoint: Receives text, synthesizes speech using user embeddings.
    
    Unity Function: Request synthesized voice -> Play synthesized voice
    
    Args:
        text: The text to be synthesized into speech (form field or query parameter)
        user_id: Optional user ID to use their embedding for voice cloning
    
    Returns:
        The synthesized audio file as binary response
    """
    try:
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text input is required for synthesis"
            )
        
        # Load user embedding if user_id provided
        user_embedding = None
        if user_id:
            embedding_files = list(EMBEDDINGS_DIR.glob(f"{user_id}_embedding_*.json"))
            if embedding_files:
                latest_file = max(embedding_files, key=lambda p: p.stat().st_mtime)
                with open(latest_file, "r") as f:
                    user_embedding = json.load(f)
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No embedding found for user '{user_id}'. Please generate an embedding first."
                )
        
        # Synthesize audio from text
        synthesized_data, output_filename = synthesize_audio(
            text=text,
            user_embedding=user_embedding
        )
        
        # Save synthesized file
        output_path = SYNTHESIZED_DIR / output_filename
        with open(output_path, "wb") as f:
            f.write(synthesized_data)
        
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
                "X-User-Id": user_id or "none"
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
    embedding_files = []
    
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
    
    for file_path in EMBEDDINGS_DIR.iterdir():
        if file_path.is_file():
            embedding_files.append({
                "filename": file_path.name,
                "size_bytes": os.path.getsize(file_path)
            })
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "uploaded_files": uploaded_files,
            "synthesized_files": synthesized_files,
            "embedding_files": embedding_files,
            "total_count": len(uploaded_files) + len(synthesized_files) + len(embedding_files)
        }
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Audio API Demo is active",
        "version": "3.0.0",
        "unity_functions": {
            "1_record_and_embed": "POST /audio/embedding - Record voice, generate embedding",
            "2_playback_local": "No server communication needed",
            "3_synthesize_and_play": "POST /audio/synthesize - Send text, get synthesized audio (TTS with user embeddings)"
        },
        "other_endpoints": {
            "get_embedding": "GET /audio/embedding/{user_id}",
            "upload": "POST /audio/upload",
            "download": "GET /audio/download/{filename}",
            "list": "GET /audio/list"
        }
    }


# ============== TEST CODE ==============
def test_embedding_generation(file_path: Path, user_id: str):
    """
    Test function to generate user embedding from voice recording.
    
    Args:
        file_path: Path to the voice recording file
        user_id: User identifier
    """
    import requests
    
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return None
    
    url = f"http://localhost:8000/audio/embedding?user_id={user_id}"
    
    with open(file_path, "rb") as audio_file:
        files = {"audio_file": (file_path.name, audio_file, "audio/mp4")}
        response = requests.post(url, files=files)
    
    print(f"Embedding Response Status: {response.status_code}")
    print(f"Embedding Response: {response.json()}")
    return response


def test_get_embedding(user_id: str):
    """
    Test function to retrieve user embedding.
    """
    import requests
    
    url = f"http://localhost:8000/audio/embedding/{user_id}"
    response = requests.get(url)
    
    print(f"Get Embedding Response Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"User ID: {data['embedding_data']['user_id']}")
        print(f"Embedding Dim: {data['embedding_data']['embedding_dim']}")
    else:
        print(f"Response: {response.json()}")
    return response


def test_synthesize_with_user(text: str, output_file_path: Path, user_id: str = None):
    """
    Test text-to-speech synthesis with optional user embedding.
    """
    import requests
    
    url = "http://localhost:8000/audio/synthesize"
    data = {"text": text}
    if user_id:
        data["user_id"] = user_id
    
    print(f"Sending text for synthesis (user: {user_id or 'none'}): '{text[:50]}...'")
    
    response = requests.post(url, data=data)
    
    print(f"Synthesis Response Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"User ID used: {response.headers.get('X-User-Id', 'none')}")
        print(f"Input text: {response.headers.get('X-Input-Text', 'N/A')}")
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
    print(f"Embedding files: {len(data.get('embedding_files', []))}")
    return response


def run_tests():
    """
    Run all test functions simulating Unity workflow
    """
    import time
    
    print("=" * 70)
    print("Running API Tests - Unity Workflow Simulation")
    print("=" * 70)
    
    sample_file = Path("sample.m4a")
    test_user_id = "unity_test_user"
    synthesized_output = Path("synthesized_output.wav")
    test_text = "Hello, this is a test of the text-to-speech synthesis system."
    
    # Unity Function 1: Record voice -> Generate embedding
    print("\n" + "-" * 70)
    print("[UNITY FUNCTION 1] Record Voice -> Generate User Embedding")
    print("-" * 70)
    test_embedding_generation(file_path=sample_file, user_id=test_user_id)
    
    time.sleep(0.5)
    
    # Verify embedding was created
    print("\n[VERIFY] Retrieving generated embedding...")
    test_get_embedding(user_id=test_user_id)
    
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