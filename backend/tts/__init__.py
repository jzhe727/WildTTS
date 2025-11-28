"""
TTS Module - Unified interface for multiple TTS engines.

This module provides a unified interface for different TTS engines,
allowing easy switching between CosyVoice and FishAudio implementations.

Usage:
    from tts import create_tts
    
    # Create a CosyVoice TTS instance
    tts = create_tts('cosyvoice', model_path='/path/to/model')
    
    # Or create a FishAudio TTS instance
    tts = create_tts('fishaudio', api_key='your_api_key')
    
    # Use the same interface for both
    tts.synthesize(
        text="Hello world",
        prompt_wav_path="reference.wav",
        output_wav_path="output.wav",
        style_text="Reference transcript"
    )
"""

from .base import BaseTTS
from .cosyvoice_tts import CosyVoiceTTS
from .fishaudio_tts import FishAudioTTS


def create_tts(engine: str, **kwargs) -> BaseTTS:
    """
    Factory function to create a TTS engine instance.
    
    Args:
        engine: The TTS engine to use ('cosyvoice' or 'fishaudio')
        **kwargs: Engine-specific initialization parameters
    
    Returns:
        A TTS engine instance implementing BaseTTS interface
    
    Raises:
        ValueError: If an unsupported engine is specified
    
    Examples:
        # CosyVoice
        tts = create_tts('cosyvoice', model_path='/path/to/model')
        
        # FishAudio
        tts = create_tts('fishaudio', api_key='your_api_key')
    """
    engine = engine.lower()
    
    if engine == 'cosyvoice':
        return CosyVoiceTTS(**kwargs)
    elif engine == 'fishaudio':
        return FishAudioTTS(**kwargs)
    else:
        raise ValueError(
            f"Unsupported TTS engine: {engine}. "
            f"Supported engines: 'cosyvoice', 'fishaudio'"
        )


__all__ = [
    'BaseTTS',
    'CosyVoiceTTS',
    'FishAudioTTS',
    'create_tts',
]
