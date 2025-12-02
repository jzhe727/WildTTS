import os
os.environ["DS_ACCELERATOR"]="cpu"
from tts import create_tts  


ref_path = './CosyVoice/asset/zero_shot_prompt.wav'
ref_transcript = "希望你以后能够做的比我还好呦。"

tts = create_tts(
    engine='fishaudio',
)
print("FishAudio TTS instance created.")

tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path=ref_path,
    output_wav_path="output_interface_fishaudio.wav",
    style_text=ref_transcript  # Optional for FishAudio
)

print("Basic FishAudio synthesis completed.")

tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path=ref_path,
    output_wav_path="output_interface_fishaudio_enhance_audio.wav",
    style_text=ref_transcript,
    enhance_audio_quality=True
)

print("Enhanced audio FishAudio synthesis completed.")

cosyvoice_tts = create_tts(
    engine='cosyvoice',
    model_path='./CosyVoice/pretrained_models/CosyVoice2-0.5B',
)

print("CosyVoice TTS instance created.")
cosyvoice_tts.synthesize(
    text="Hello world, testing testing 1 2 3",
    prompt_wav_path=ref_path,
    output_wav_path="output_interface_cosyvoice.wav",
    style_text=ref_transcript
)

print("CosyVoice synthesis completed.")
