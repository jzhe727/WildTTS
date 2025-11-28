# TTS Module

### Basic Usage with Factory Function

```python
from tts import create_tts

# Create a CosyVoice TTS instance
tts = create_tts(
    engine='cosyvoice',
    model_path='/path/to/cosyvoice/model'
)

# Synthesize speech
tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path="reference.wav",
    output_wav_path="output.wav",
    style_text="Reference audio transcript"
)

print(f"Sample rate: {tts.sample_rate}")
```

### Switch to FishAudio

```python
from tts import create_tts

# Create a FishAudio TTS instance
tts = create_tts(
    engine='fishaudio',
    api_key='your_api_key'  # Optional if set in environment
)

# Use the exact same interface
tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path="reference.wav",
    output_wav_path="output.wav",
    style_text="Reference audio transcript"  # Optional for FishAudio
)
```

### Direct Class Instantiation

```python
from tts import CosyVoiceTTS, FishAudioTTS

# CosyVoice
cosyvoice = CosyVoiceTTS(
    model_path='/path/to/model',
    load_jit=False,
    load_trt=False,
    fp16=False
)

# FishAudio
fishaudio = FishAudioTTS(api_key='your_api_key')
```

### Advanced CosyVoice Options

```python
tts = create_tts(
    engine='cosyvoice',
    model_path='/path/to/model',
    load_jit=True,      # Use JIT optimization
    load_trt=False,     # Use TensorRT optimization
    load_vllm=False,    # Use VLLM optimization
    fp16=True           # Use FP16 precision
)

# Streaming inference
tts.synthesize(
    text="Long text to synthesize...",
    prompt_wav_path="reference.wav",
    output_wav_path="output_{}.wav",  # {} will be replaced with index
    style_text="Reference transcript",
    stream=True
)
```

## API Reference

### BaseTTS (Abstract Base Class)

All TTS engines inherit from this base class.

#### Methods

- `__init__(**kwargs)`: Initialize the TTS engine
- `synthesize(text, prompt_wav_path, output_wav_path, style_text=None, **kwargs)`: Synthesize speech
- `sample_rate` (property): Get the sample rate of generated audio

### CosyVoiceTTS

#### Constructor Parameters
- `model_path` (str, required): Path to the CosyVoice model
- `load_jit` (bool, default=False): Load JIT optimized model
- `load_trt` (bool, default=False): Load TensorRT optimized model
- `load_vllm` (bool, default=False): Load VLLM optimized model
- `fp16` (bool, default=False): Use FP16 precision

#### Synthesize Parameters
- `text` (str, required): Text to synthesize
- `prompt_wav_path` (str, required): Path to reference audio
- `output_wav_path` (str, required): Output path (use `{}` for streaming)
- `style_text` (str, required): Transcript of reference audio
- `stream` (bool, default=False): Use streaming inference

### FishAudioTTS

#### Constructor Parameters
- `api_key` (str, optional): API key for FishAudio

#### Synthesize Parameters
- `text` (str, required): Text to synthesize
- `prompt_wav_path` (str, required): Path to reference audio
- `output_wav_path` (str, required): Output path
- `style_text` (str, optional): Transcript of reference audio

## Architecture

```
tts/
├── __init__.py           # Module entry point with factory function
├── base.py               # Abstract base class (BaseTTS)
├── cosyvoice_tts.py      # CosyVoice implementation
├── fishaudio_tts.py      # FishAudio implementation
└── README.md             # This file
```

## Adding New TTS Engines

1. Create a new file (e.g., `newtts.py`)
2. Subclass `BaseTTS`
3. Implement all abstract methods
4. Add to `__init__.py` and factory function

```python
from .base import BaseTTS

class NewTTS(BaseTTS):
    def __init__(self, **kwargs):
        # Initialize your TTS engine
        pass
    
    def synthesize(self, text, prompt_wav_path, output_wav_path, 
                   style_text=None, **kwargs):
        # Implement synthesis logic
        pass
    
    @property
    def sample_rate(self):
        return 22050
```

## Error Handling

```python
from tts import create_tts

try:
    tts = create_tts('cosyvoice', model_path='/path/to/model')
    tts.synthesize(
        text="Hello",
        prompt_wav_path="ref.wav",
        output_wav_path="out.wav",
        style_text="Reference"
    )
except ValueError as e:
    print(f"Invalid engine or parameters: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

