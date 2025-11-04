import sys
import os
import torchaudio

cosyvoice_path = os.path.join(os.path.dirname(__file__), 'CosyVoice')
sys.path.append(os.path.abspath(cosyvoice_path))
matcha_tts_path = os.path.join(cosyvoice_path, 'third_party/Matcha-TTS')
sys.path.append(matcha_tts_path)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402


def main():
    model_path = cosyvoice_path + '/pretrained_models/CosyVoice2-0.5B'
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    prompt_speech_16k = load_wav('./CosyVoice/asset/zero_shot_prompt.wav', 16000)
    for i, j in enumerate(cosyvoice.inference_zero_shot('Hello world, testing testing 1 2 3', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


if __name__ == "__main__":
    main()
