import os
from typing import Optional, Dict

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
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enhance_audio_quality: bool = False,
        cache_voices: bool = True,
        auto_cleanup_voices: bool = False,
        **kwargs
    ):
        """
        Initialize FishAudio TTS engine.
        
        Args:
            api_key: Optional API key for FishAudio (if not set in environment)
            enhance_audio_quality: If True, creates a persistent voice model with
                audio quality enhancement before synthesis. This cleans up noisy
                reference audio and normalizes levels. Note: This requires creating
                a voice model on FishAudio's servers for each unique reference audio.
            cache_voices: If True and enhance_audio_quality is enabled, caches
                created voice models to avoid recreating them for the same reference
                audio. The cache is keyed by the hash of the audio file content.
            auto_cleanup_voices: If True, automatically deletes created voice models
                after synthesis. Only applies when enhance_audio_quality is True.
                If cache_voices is True, cleanup happens when the instance is destroyed.
        """
        if not FISHAUDIO_AVAILABLE:
            raise ImportError(
                "FishAudio is not installed. "
                "Please install it with: pip install fish-audio-sdk"
            )
        
        self.client = FishAudio(api_key=api_key) if api_key else FishAudio()
        self._sample_rate = 44100  # FishAudio default sample rate
        self.enhance_audio_quality = enhance_audio_quality
        self.cache_voices = cache_voices
        self.auto_cleanup_voices = auto_cleanup_voices
        self._voice_cache: Dict[str, str] = {}  # audio_path -> voice_id
    
    def _get_cache_key(self, audio_path: str) -> str:
        """Generate a cache key from the audio file path."""
        return os.path.basename(audio_path)
    
    def _create_enhanced_voice(
        self,
        audio_path: str,
        audio_data: bytes,
        transcript: Optional[str] = None
    ) -> str:
        """
        Create a persistent voice model with audio quality enhancement.
        
        Args:
            audio_path: Path to the audio file (used as cache key)
            audio_data: Raw audio bytes
            transcript: Optional transcript of the audio
            
        Returns:
            The voice model ID
        """
        cache_key = self._get_cache_key(audio_path)
        
        # Check cache first
        if self.cache_voices and cache_key in self._voice_cache:
            return self._voice_cache[cache_key]
        
        # Create voice model with enhancement
        create_kwargs = {
            "title": f"TTS Voice {cache_key}",
            "voices": [audio_data],
            "description": "Auto-generated voice for TTS synthesis",
            "enhance_audio_quality": True,
            "visibility": "private"
        }
        
        # Add transcript if provided for better quality
        if transcript:
            create_kwargs["texts"] = [transcript]
        
        voice = self.client.voices.create(**create_kwargs)
        voice_id = voice.id
        
        # Cache the voice ID
        if self.cache_voices:
            self._voice_cache[cache_key] = voice_id
        
        return voice_id
    
    def _cleanup_voice(self, voice_id: str) -> None:
        """Delete a voice model from FishAudio."""
        try:
            self.client.voices.delete(voice_id)
        except Exception:
            # Silently ignore cleanup errors
            pass
    
    def cleanup_cached_voices(self) -> None:
        """
        Delete all cached voice models from FishAudio.
        Call this when you're done using the TTS engine to clean up resources.
        """
        for voice_id in self._voice_cache.values():
            self._cleanup_voice(voice_id)
        self._voice_cache.clear()
    
    def __del__(self):
        """Cleanup voice models on destruction if auto_cleanup is enabled."""
        if getattr(self, 'auto_cleanup_voices', False) and getattr(self, 'cache_voices', False):
            self.cleanup_cached_voices()
    
    def synthesize(
        self,
        text: str,
        prompt_wav_path: str,
        output_wav_path: str,
        style_text: Optional[str] = None,
        enhance_audio_quality: Optional[bool] = None,
        **kwargs
    ) -> None:
        """
        Synthesize speech using FishAudio voice cloning.
        
        Args:
            text: The text to synthesize
            prompt_wav_path: Path to the reference audio file
            output_wav_path: Path where the synthesized audio should be saved
            style_text: Optional transcript of the reference audio
            enhance_audio_quality: Override the instance-level enhance_audio_quality
                setting for this synthesis call. If None, uses the instance setting.
        """
        if not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"Reference audio not found: {prompt_wav_path}")
        
        # Determine whether to use enhancement
        use_enhancement = (
            enhance_audio_quality if enhance_audio_quality is not None 
            else self.enhance_audio_quality
        )
        
        # Read reference audio
        with open(prompt_wav_path, "rb") as f:
            audio_data = f.read()
        
        if use_enhancement:
            # Create a persistent voice model with audio quality enhancement
            voice_id = self._create_enhanced_voice(prompt_wav_path, audio_data, style_text)
            
            try:
                # Generate audio using the enhanced voice model
                audio = self.client.tts.convert(
                    text=text,
                    reference_id=voice_id
                )
            finally:
                # Cleanup if not caching and auto_cleanup is enabled
                if self.auto_cleanup_voices and not self.cache_voices:
                    self._cleanup_voice(voice_id)
        else:
            # Use instant voice cloning (original behavior)
            reference_audio = ReferenceAudio(
                audio=audio_data,
                text=style_text or ""  # FishAudio can work without text
            )
            
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
