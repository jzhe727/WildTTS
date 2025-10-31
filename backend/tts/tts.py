import sys
import os
import torchaudio

cosyvoice_path = os.path.join(os.path.dirname(__file__), 'CosyVoice')
sys.path.append(os.path.abspath(cosyvoice_path))
matcha_tts_path = os.path.join(cosyvoice_path, 'third_party/Matcha-TTS')
sys.path.append(matcha_tts_path)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402

class TTS:
    def __init__(self, model_path: str):
        self.cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self.sample_rate = self.cosyvoice.sample_rate

    def synthesize(self, text: str, style_text: str, prompt_wav_path: str, output_wav_path: str):
        prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        for i, j in enumerate(self.cosyvoice.inference_zero_shot(text, style_text, prompt_speech_16k, stream=False)):
            torchaudio.save(output_wav_path.format(i), j['tts_speech'], self.sample_rate)
