import os
os.environ["DS_ACCELERATOR"]="cpu"
from tts import create_tts  


ref_path = './CosyVoice/asset/zero_shot_prompt.wav'
ref_transcript = "希望你以后能够做的比我还好呦。"

tts = create_tts(
    engine='fishaudio',
)


tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path=ref_path,
    output_wav_path="output_interface_fishaudio.wav",
    style_text=ref_transcript  # Optional for FishAudio
)

cosyvoice_tts = create_tts(
    engine='cosyvoice',
    model_path='./CosyVoice/pretrained_models/CosyVoice2-0.5B',
)

cosyvoice_tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path=ref_path,
    output_wav_path="output_interface_cosyvoice.wav",
    style_text=ref_transcript
)
