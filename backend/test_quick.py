"""
Quick API Test Script

A simple script to quickly test the API after server startup.
"""

import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"


def quick_test():
    """Quick integration test"""
    print("=" * 50)
    print("WildTTS API Quick Test")
    print("=" * 50)
    
    # 1. Check server status
    print("\n[1] Checking server status...")
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"    Status: {r.json().get('status')}")
        print(f"    Version: {r.json().get('version')}")
    except:
        print("    ❌ Server connection failed!")
        print("    Start the server first: python FastAPIdemo.py")
        return
    
    # 2. Check supported models
    print("\n[2] Checking supported models...")
    r = requests.get(f"{BASE_URL}/audio/models")
    models = r.json().get('supported_models', [])
    print(f"    Supported models: {models}")
    
    # 3. Save reference
    print("\n[3] Saving reference...")
    audio_file = Path("zero_shot_0.wav")
    if not audio_file.exists():
        print(f"    ❌ Test audio file not found: {audio_file}")
        return
    
    with open(audio_file, "rb") as f:
        files = {"audio_file": (audio_file.name, f, "audio/wav")}
        data = {
            "user_id": "quick_test_user",
            "reference_text": "This is a test reference audio."
        }
        r = requests.post(f"{BASE_URL}/audio/reference", files=files, data=data)
    print(f"    Result: {r.json().get('message')}")
    
    # 4. Synthesize speech
    print("\n[4] Synthesizing speech (default model)...")
    data = {
        "text": "Hello, this is a quick test.",
        "user_id": "quick_test_user"
    }
    r = requests.post(f"{BASE_URL}/audio/synthesize", data=data)
    if r.status_code == 200:
        output_file = Path("quick_test_output.wav")
        with open(output_file, "wb") as f:
            f.write(r.content)
        model_used = r.headers.get('X-Model-Used', 'unknown')
        print(f"    ✅ Synthesis successful!")
        print(f"    Model used: {model_used}")
        print(f"    Saved to: {output_file}")
    else:
        print(f"    ❌ Synthesis failed: {r.text}")
    
    # 5. List files
    print("\n[5] Listing server files...")
    r = requests.get(f"{BASE_URL}/audio/list")
    data = r.json()
    print(f"    Reference files: {len(data.get('reference_files', []))}")
    print(f"    Synthesized files: {len(data.get('synthesized_files', []))}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)


def test_different_models():
    """Test synthesis with different models"""
    print("=" * 50)
    print("Model-specific Synthesis Test")
    print("=" * 50)
    
    # Save reference first
    audio_file = Path("zero_shot_0.wav")
    if not audio_file.exists():
        print(f"Test audio file not found: {audio_file}")
        return
    
    with open(audio_file, "rb") as f:
        files = {"audio_file": (audio_file.name, f, "audio/wav")}
        data = {
            "user_id": "model_test_user",
            "reference_text": "This is a reference for multi-model testing."
        }
        requests.post(f"{BASE_URL}/audio/reference", files=files, data=data)
    
    # Get supported models
    r = requests.get(f"{BASE_URL}/audio/models")
    models = r.json().get('supported_models', [])
    
    # Synthesize with each model
    for model in models:
        print(f"\n[{model}] Synthesizing with {model} model...")
        data = {
            "text": f"This is a speech synthesis test using the {model} model.",
            "user_id": "model_test_user",
            "model": model
        }
        
        try:
            r = requests.post(f"{BASE_URL}/audio/synthesize", data=data, timeout=60)
            
            if r.status_code == 200:
                output_file = Path(f"test_output_{model}.wav")
                with open(output_file, "wb") as f:
                    f.write(r.content)
                print(f"    ✅ Success! Saved: {output_file}")
            else:
                print(f"    ❌ Failed: {r.status_code}")
        except Exception as e:
            print(f"    ❌ Error: {e}")


def test_error_cases():
    """Test error cases"""
    print("=" * 50)
    print("Error Cases Test")
    print("=" * 50)
    
    # 1. Non-existent user
    print("\n[1] Attempting synthesis with non-existent user...")
    r = requests.post(f"{BASE_URL}/audio/synthesize", data={
        "text": "test",
        "user_id": "nonexistent_user_99999"
    })
    print(f"    Status code: {r.status_code} (expected: 404)")
    
    # 2. Invalid model name
    print("\n[2] Attempting synthesis with invalid model name...")
    r = requests.post(f"{BASE_URL}/audio/synthesize", data={
        "text": "test",
        "user_id": "quick_test_user",
        "model": "invalid_model"
    })
    print(f"    Status code: {r.status_code} (expected: 400)")
    
    # 3. Empty text
    print("\n[3] Attempting synthesis with empty text...")
    r = requests.post(f"{BASE_URL}/audio/synthesize", data={
        "text": "",
        "user_id": "quick_test_user"
    })
    print(f"    Status code: {r.status_code} (expected: 400)")
    
    # 4. Missing reference text
    print("\n[4] Attempting to save reference without text...")
    audio_file = Path("zero_shot_0.wav")
    if audio_file.exists():
        with open(audio_file, "rb") as f:
            files = {"audio_file": (audio_file.name, f, "audio/wav")}
            data = {"user_id": "test", "reference_text": ""}
            r = requests.post(f"{BASE_URL}/audio/reference", files=files, data=data)
        print(f"    Status code: {r.status_code} (expected: 400)")


def test_long_text():
    """Test synthesis with long text"""
    print("=" * 50)
    print("Long Text Synthesis Test")
    print("=" * 50)
    
    # Check if reference exists
    r = requests.get(f"{BASE_URL}/audio/reference/quick_test_user")
    if r.status_code != 200:
        print("Reference not found. Run quick_test first.")
        return
    
    long_text = """
    The quick brown fox jumps over the lazy dog. 
    This is a test of the text-to-speech synthesis system with a longer piece of text.
    We want to verify that the system can handle multiple sentences and paragraphs correctly.
    The synthesized audio should sound natural and maintain consistent voice characteristics
    throughout the entire passage.
    """
    
    print(f"Input text length: {len(long_text)} characters")
    print("Synthesizing...")
    
    data = {
        "text": long_text,
        "user_id": "quick_test_user"
    }
    
    r = requests.post(f"{BASE_URL}/audio/synthesize", data=data, timeout=120)
    
    if r.status_code == 200:
        output_file = Path("test_long_text_output.wav")
        with open(output_file, "wb") as f:
            f.write(r.content)
        output_size = r.headers.get('X-Output-Size', '0')
        print(f"✅ Success!")
        print(f"Output size: {output_size} bytes")
        print(f"Saved to: {output_file}")
    else:
        print(f"❌ Failed: {r.text}")


def test_special_characters():
    """Test synthesis with special characters"""
    print("=" * 50)
    print("Special Characters Test")
    print("=" * 50)
    
    # Check if reference exists
    r = requests.get(f"{BASE_URL}/audio/reference/quick_test_user")
    if r.status_code != 200:
        print("Reference not found. Run quick_test first.")
        return
    
    test_texts = [
        ("Numbers", "The meeting is scheduled for 3:30 PM on December 25, 2025."),
        ("Punctuation", "Hello! How are you? I'm doing great, thanks!"),
        ("Abbreviations", "Dr. Smith works at NASA and earned his Ph.D. in 2010."),
        ("Symbols", "The price is $99.99, which is a 20% discount."),
    ]
    
    for test_name, text in test_texts:
        print(f"\n[{test_name}] Testing: {text[:40]}...")
        
        data = {
            "text": text,
            "user_id": "quick_test_user"
        }
        
        r = requests.post(f"{BASE_URL}/audio/synthesize", data=data, timeout=60)
        
        if r.status_code == 200:
            print(f"    ✅ Success")
        else:
            print(f"    ❌ Failed: {r.status_code}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "models":
            test_different_models()
        elif arg == "errors":
            test_error_cases()
        elif arg == "long":
            test_long_text()
        elif arg == "special":
            test_special_characters()
        else:
            print("Usage:")
            print("  python test_quick.py          # Quick test")
            print("  python test_quick.py models   # Test all models")
            print("  python test_quick.py errors   # Test error cases")
            print("  python test_quick.py long     # Test long text")
            print("  python test_quick.py special  # Test special characters")
    else:
        quick_test()
