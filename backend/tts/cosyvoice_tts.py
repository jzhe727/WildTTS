import sys
import os
from typing import Optional
import torchaudio

from .base import BaseTTS

# Add CosyVoice to path
cosyvoice_path = os.path.join(os.path.dirname(__file__), '..', 'CosyVoice')
sys.path.append(os.path.abspath(cosyvoice_path))
matcha_tts_path = os.path.join(cosyvoice_path, 'third_party/Matcha-TTS')
sys.path.append(matcha_tts_path)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402


class CosyVoiceTTS(BaseTTS):
    """CosyVoice TTS engine implementation."""
    
    def __init__(
        self,
        model_path: str,
        load_jit: bool = False,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
        **kwargs
    ):
        """
        Initialize CosyVoice TTS engine.
        
        Args:
            model_path: Path to the CosyVoice model
            load_jit: Whether to load JIT optimized model
            load_trt: Whether to load TensorRT optimized model
            load_vllm: Whether to load VLLM optimized model
            fp16: Whether to use FP16 precision
        """
        self.cosyvoice = CosyVoice2(
            model_path,
            load_jit=load_jit,
            load_trt=load_trt,
            load_vllm=load_vllm,
            fp16=fp16
        )
        self._sample_rate = self.cosyvoice.sample_rate
    
    def synthesize(
        self,
        text: str,
        prompt_wav_path: str,
        output_wav_path: str,
        style_text: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> None:
        """
        Synthesize speech using CosyVoice zero-shot inference.
        
        Args:
            text: The text to synthesize
            prompt_wav_path: Path to the reference audio file
            output_wav_path: Path where the synthesized audio should be saved
            style_text: Text transcript or description of the prompt audio
            stream: Whether to use streaming inference
        """
        if style_text is None:
            raise ValueError("CosyVoice requires style_text parameter")
        
        # prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        
        for i, j in enumerate(self.cosyvoice.inference_zero_shot(
            text, style_text, prompt_wav_path, stream=stream
        )):
            torchaudio.save(
                output_wav_path.format(i) if '{}' in output_wav_path else output_wav_path,
                j['tts_speech'],
                self._sample_rate
            )
    
    @property
    def sample_rate(self) -> int:
        """Return the sample rate of the generated audio."""
        return self._sample_rate
