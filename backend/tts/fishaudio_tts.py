import os
from typing import Optional

from .base import BaseTTS

try:
    from fishaudio import FishAudio
    from fishaudio.utils import save
    from fishaudio.types import ReferenceAudio
    FISHAUDIO_AVAILABLE = True
except ImportError:
    FISHAUDIO_AVAILABLE = False


class FishAudioTTS(BaseTTS):
    """FishAudio TTS engine implementation."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize FishAudio TTS engine.
        
        Args:
            api_key: Optional API key for FishAudio (if not set in environment)
        """
        if not FISHAUDIO_AVAILABLE:
            raise ImportError(
                "FishAudio is not installed. "
                "Please install it with: pip install fish-audio-sdk"
            )
        
        self.client = FishAudio(api_key=api_key) if api_key else FishAudio()
        self._sample_rate = 44100  # FishAudio default sample rate
    
    def synthesize(
        self,
        text: str,
        prompt_wav_path: str,
        output_wav_path: str,
        style_text: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Synthesize speech using FishAudio voice cloning.
        
        Args:
            text: The text to synthesize
            prompt_wav_path: Path to the reference audio file
            output_wav_path: Path where the synthesized audio should be saved
            style_text: Optional transcript of the reference audio
        """
        if not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"Reference audio not found: {prompt_wav_path}")
        
        # Read reference audio
        with open(prompt_wav_path, "rb") as f:
            reference_audio = ReferenceAudio(
                audio=f.read(),
                text=style_text or ""  # FishAudio can work without text
            )
        
        # Convert text to speech
        audio = self.client.tts.convert(
            text=text,
            references=[reference_audio]
        )
        
        # Save output
        output_path = output_wav_path.format(0) if '{}' in output_wav_path else output_wav_path
        save(audio, output_path)
    
    @property
    def sample_rate(self) -> int:
        """Return the sample rate of the generated audio."""
        return self._sample_rate
