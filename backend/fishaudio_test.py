from fishaudio import FishAudio
from fishaudio.utils import play, save
from fishaudio.types import ReferenceAudio
import os

client = FishAudio()

# Clone voice on-the-fly
ref_path = './CosyVoice/asset/zero_shot_prompt.wav'
with open(ref_path, "rb") as f:
    audio = client.tts.convert(
        text="Hello world, testing testing 1 2 3",
        references=[ReferenceAudio(
            audio=f.read(),
            text="希望你以后能够做的比我还好呦。"
        )]
    )
    save(audio, 'fishaudio_zero_shot.wav')