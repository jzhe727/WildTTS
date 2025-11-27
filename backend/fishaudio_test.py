from fishaudio.types import ReferenceAudio

# Clone voice on-the-fly
with open("reference.wav", "rb") as f:
    audio = client.tts.convert(
        text="Cloned voice speaking",
        references=[ReferenceAudio(
            audio=f.read(),
            text="Text spoken in reference"
        )]
    )