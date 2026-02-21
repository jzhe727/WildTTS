import sys
import os
os.environ["DS_ACCELERATOR"] = "cpu" # force CPU, comment out to auto-detect GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
import logging
# logging.getLogger('matplotlib').setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
) 

logging.debug(f"Start torchaudio import")
import torchaudio
logging.debug(f"End torchaudio import")

cosyvoice_path = os.path.join(os.path.dirname(__file__), 'CosyVoice')
sys.path.append(os.path.abspath(cosyvoice_path))
matcha_tts_path = os.path.join(cosyvoice_path, 'third_party/Matcha-TTS')
sys.path.append(matcha_tts_path)

logging.debug(f"Start Cosyvoice Imports")
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402
logging.debug(f"End Cosyvoice Imports")

def main():
    model_path = cosyvoice_path + '/pretrained_models/CosyVoice2-0.5B'
    logging.debug(f"Start Cosyvoice constructor")
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    logging.debug(f"End Cosyvoice constructor")
    prompt_speech_16k = load_wav('./CosyVoice/asset/zero_shot_prompt.wav', 16000)
    logging.debug(f"Start Cosyvoice generation")
    for i, j in enumerate(cosyvoice.inference_zero_shot('Hello world, testing testing 1 2 3', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    logging.debug(f"End Cosyvoice generation")

if __name__ == "__main__":
    main()
