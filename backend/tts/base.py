from abc import ABC, abstractmethod
from typing import Optional


class BaseTTS(ABC):
    """Abstract base class for TTS engines."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the TTS engine with configuration parameters."""
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        prompt_wav_path: str,
        output_wav_path: str,
        style_text: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Synthesize speech from text using a reference audio prompt.
        
        Args:
            text: The text to synthesize
            prompt_wav_path: Path to the reference/prompt audio file
            output_wav_path: Path where the synthesized audio should be saved
            style_text: Optional text describing the style or transcript of prompt audio
            **kwargs: Additional engine-specific parameters
        """
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the generated audio."""
        pass
