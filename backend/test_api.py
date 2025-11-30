"""
API Test Script - WildTTS Backend API Tests

Usage:
    python test_api.py                    # Run all tests
    python test_api.py --test health      # Health check only
    python test_api.py --test reference   # Reference API tests
    python test_api.py --test synthesize  # Synthesis tests
    python test_api.py --test models      # Model list tests
    python test_api.py --test all         # Full integration tests
"""

import requests
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


# ============== CONFIG ==============
BASE_URL = "http://localhost:8000"
TEST_AUDIO_FILE = Path("zero_shot_0.wav")  # Test audio file
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============== UTILITY FUNCTIONS ==============
def print_header(title: str):
    """Print test section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print test sub-header"""
    print("\n" + "-" * 50)
    print(f"  {title}")
    print("-" * 50)


def print_result(success: bool, message: str):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {message}")


def check_server():
    """Check server connection"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ============== TEST FUNCTIONS ==============

def test_health_check():
    """Health check test"""
    print_header("Health Check Test")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Server Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        print(f"Default TTS Engine: {data.get('default_tts_engine')}")
        print(f"Supported Models: {data.get('supported_models')}")
        
        success = response.status_code == 200 and data.get('status') == 'running'
        print_result(success, "Health check")
        return success
        
    except Exception as e:
        print_result(False, f"Health check failed: {e}")
        return False


def test_get_models():
    """Get supported models test"""
    print_header("Get Models Test")
    
    try:
        response = requests.get(f"{BASE_URL}/audio/models")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Default Model: {data.get('default_model')}")
        print(f"Supported Models: {data.get('supported_models')}")
        
        if 'model_info' in data:
            print("\nModel Info:")
            for model_id, info in data['model_info'].items():
                print(f"  - {model_id}: {info['name']} ({info['type']})")
                print(f"    Description: {info['description']}")
        
        success = response.status_code == 200 and 'supported_models' in data
        print_result(success, "Get models")
        return success
        
    except Exception as e:
        print_result(False, f"Get models failed: {e}")
        return False


def test_save_reference(audio_file: Path, user_id: str, reference_text: str):
    """Save reference test"""
    print_subheader(f"Save Reference: user_id={user_id}")
    
    if not audio_file.exists():
        print_result(False, f"Test audio file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, "rb") as f:
            files = {"audio_file": (audio_file.name, f, "audio/wav")}
            data = {
                "user_id": user_id,
                "reference_text": reference_text
            }
            response = requests.post(f"{BASE_URL}/audio/reference", files=files, data=data)
        
        result = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Message: {result.get('message')}")
        print(f"Audio Filename: {result.get('audio_filename')}")
        print(f"Reference Text: {result.get('reference_text')[:50]}...")
        
        success = response.status_code == 200 and result.get('status') == 'success'
        print_result(success, "Save reference")
        return success
        
    except Exception as e:
        print_result(False, f"Save reference failed: {e}")
        return False


def test_get_reference(user_id: str):
    """Get reference test"""
    print_subheader(f"Get Reference: user_id={user_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/audio/reference/{user_id}")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            ref_data = data.get('reference_data', {})
            print(f"User ID: {ref_data.get('user_id')}")
            print(f"Audio Path: {ref_data.get('audio_path')}")
            print(f"Reference Text: {ref_data.get('reference_text')[:50]}...")
            print(f"Created At: {ref_data.get('created_at')}")
            print_result(True, "Get reference")
            return True
        else:
            print(f"Error: {response.json()}")
            print_result(False, "Get reference")
            return False
            
    except Exception as e:
        print_result(False, f"Get reference failed: {e}")
        return False


def test_get_reference_not_found():
    """Test getting non-existent reference"""
    print_subheader("Get Non-existent Reference")
    
    try:
        response = requests.get(f"{BASE_URL}/audio/reference/nonexistent_user_12345")
        
        print(f"Status Code: {response.status_code}")
        
        success = response.status_code == 404
        print_result(success, "Get non-existent reference returns 404")
        return success
        
    except Exception as e:
        print_result(False, f"Test failed: {e}")
        return False


def test_synthesize(text: str, user_id: str, model: str = None, output_suffix: str = ""):
    """Speech synthesis test"""
    model_str = model or "default"
    print_subheader(f"Synthesize: model={model_str}")
    
    try:
        data = {
            "text": text,
            "user_id": user_id
        }
        if model:
            data["model"] = model
        
        print(f"Input Text: {text[:50]}...")
        print(f"User ID: {user_id}")
        print(f"Model: {model_str}")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/audio/synthesize", data=data)
        elapsed = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            model_used = response.headers.get('X-Model-Used', 'unknown')
            output_size = response.headers.get('X-Output-Size', '0')
            
            print(f"Model Used: {model_used}")
            print(f"Output Size: {output_size} bytes")
            
            # Save file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"synth_{model_used}_{timestamp}{output_suffix}.wav"
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Saved to: {output_file}")
            
            print_result(True, f"Synthesize with {model_str}")
            return True
        else:
            print(f"Error: {response.text}")
            print_result(False, f"Synthesize with {model_str}")
            return False
            
    except Exception as e:
        print_result(False, f"Synthesize failed: {e}")
        return False


def test_synthesize_without_reference():
    """Test synthesis without reference"""
    print_subheader("Synthesize Without Reference")
    
    try:
        data = {
            "text": "This should fail",
            "user_id": "nonexistent_user_99999"
        }
        response = requests.post(f"{BASE_URL}/audio/synthesize", data=data)
        
        print(f"Status Code: {response.status_code}")
        
        success = response.status_code == 404
        print_result(success, "Synthesize without reference returns 404")
        return success
        
    except Exception as e:
        print_result(False, f"Test failed: {e}")
        return False


def test_synthesize_invalid_model():
    """Test synthesis with invalid model name"""
    print_subheader("Synthesize With Invalid Model")
    
    try:
        data = {
            "text": "This should fail",
            "user_id": "test_user",
            "model": "invalid_model_name"
        }
        response = requests.post(f"{BASE_URL}/audio/synthesize", data=data)
        
        print(f"Status Code: {response.status_code}")
        
        success = response.status_code == 400
        print_result(success, "Invalid model returns 400")
        return success
        
    except Exception as e:
        print_result(False, f"Test failed: {e}")
        return False


def test_list_files():
    """List files test"""
    print_subheader("List Files")
    
    try:
        response = requests.get(f"{BASE_URL}/audio/list")
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Uploaded Files: {len(data.get('uploaded_files', []))}")
        print(f"Synthesized Files: {len(data.get('synthesized_files', []))}")
        print(f"Reference Files: {len(data.get('reference_files', []))}")
        print(f"Total Count: {data.get('total_count', 0)}")
        
        success = response.status_code == 200
        print_result(success, "List files")
        return success
        
    except Exception as e:
        print_result(False, f"List files failed: {e}")
        return False


def test_upload_file(audio_file: Path):
    """File upload test"""
    print_subheader("Upload File")
    
    if not audio_file.exists():
        print_result(False, f"Test file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, "rb") as f:
            files = {"audio_file": (audio_file.name, f, "audio/wav")}
            response = requests.post(f"{BASE_URL}/audio/upload", files=files)
        
        data = response.json()
        print(f"Status Code: {response.status_code}")
        print(f"Filename: {data.get('filename')}")
        print(f"Size: {data.get('size_bytes')} bytes")
        
        success = response.status_code == 200
        print_result(success, "Upload file")
        return success
        
    except Exception as e:
        print_result(False, f"Upload failed: {e}")
        return False


# ============== INTEGRATION TESTS ==============

def run_reference_tests():
    """Reference API tests"""
    print_header("Reference API Tests")
    
    results = []
    test_user = f"test_user_{datetime.now().strftime('%H%M%S')}"
    
    # Save reference
    results.append(test_save_reference(
        TEST_AUDIO_FILE,
        test_user,
        "Hello, this is a test reference audio for voice cloning."
    ))
    
    # Get reference
    results.append(test_get_reference(test_user))
    
    # Get non-existent reference
    results.append(test_get_reference_not_found())
    
    return all(results)


def run_synthesize_tests():
    """Speech synthesis tests"""
    print_header("Speech Synthesis API Tests")
    
    results = []
    test_user = f"synth_test_{datetime.now().strftime('%H%M%S')}"
    
    # Save reference first
    test_save_reference(
        TEST_AUDIO_FILE,
        test_user,
        "This is a reference audio for speech synthesis testing."
    )
    
    # Synthesize with default model
    results.append(test_synthesize(
        "Hello, the weather is really nice today.",
        test_user,
        model=None,
        output_suffix="_default"
    ))
    
    # Synthesize with CosyVoice
    results.append(test_synthesize(
        "This is a speech synthesis test using CosyVoice model.",
        test_user,
        model="cosyvoice",
        output_suffix="_cosyvoice"
    ))
    
    # Synthesize without reference
    results.append(test_synthesize_without_reference())
    
    # Synthesize with invalid model
    results.append(test_synthesize_invalid_model())
    
    return all(results)


def run_all_tests():
    """Run all integration tests"""
    print_header("🚀 Starting Full API Integration Tests")
    
    if not check_server():
        print("\n❌ Cannot connect to server.")
        print(f"   Make sure server is running at {BASE_URL}")
        print("   Run: python FastAPIdemo.py")
        return
    
    print(f"\n✅ Server connection verified: {BASE_URL}")
    
    results = {
        "Health Check": test_health_check(),
        "Get Models": test_get_models(),
        "Reference API": run_reference_tests(),
        "Synthesis API": run_synthesize_tests(),
        "List Files": test_list_files(),
        "Upload File": test_upload_file(TEST_AUDIO_FILE),
    }
    
    # Results summary
    print_header("📊 Test Results Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nPassed {passed} of {total} tests")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description="WildTTS API Tests")
    parser.add_argument(
        "--test",
        choices=["health", "models", "reference", "synthesize", "list", "upload", "all"],
        default="all",
        help="Select test to run"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    if not check_server():
        print(f"\n❌ Cannot connect to server: {BASE_URL}")
        print("   Run server: python FastAPIdemo.py")
        return
    
    if args.test == "health":
        test_health_check()
    elif args.test == "models":
        test_get_models()
    elif args.test == "reference":
        run_reference_tests()
    elif args.test == "synthesize":
        run_synthesize_tests()
    elif args.test == "list":
        test_list_files()
    elif args.test == "upload":
        test_upload_file(TEST_AUDIO_FILE)
    else:
        run_all_tests()


if __name__ == "__main__":
    main()
