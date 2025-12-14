# WildTTS Frontend - Unity Project

A Unity-based frontend application for WildTTS, a text-to-speech system with voice cloning capabilities. This project provides a 3D avatar interface that can record your voice, clone it, and synthesize speech in your cloned voice.

## Requirements

### Unity Editor
- **Unity Version**: 6000.0.47f1 (Unity 6)
- Download from [Unity Hub](https://unity.com/download)

### System Requirements
- macOS, Windows, or Linux
- Microphone for voice recording
- Internet connection (for Ready Player Me avatar loading)

## Installation

1. **Clone or download this repository**
   ```bash
   cd /path/to/WildTTS/frontend
   ```

2. **Open the project in Unity**
   - Launch Unity Hub
   - Click "Open" and select the `frontend` folder
   - Unity will automatically import all packages and dependencies

3. **Wait for Unity to import assets**
   - First-time import may take several minutes
   - Ensure all packages are imported successfully

4. **Verify package installation**
   - The project uses the following key packages:
     - Ready Player Me Core SDK
     - Unity Input System
     - Universal Render Pipeline (URP)
     - TextMesh Pro
     - Oculus LipSync

## Configuration

## Running the Project

### 1. Open the Scene in Unity

1. In Unity Editor, navigate to `Assets/Scenes/`
2. Double-click `SampleScene.unity` to open it

### 2. Configure the Scene

1. Ensure the `UnityAudioManager` component is attached to a GameObject(AudioManager in SampleScene)
2. Assign all required UI elements in the Inspector
3. Set up the character avatar and animator if using animations

### 3. Play the Scene

1. Click the **Play** button in the Unity Editor

## Usage

### Recording Your Voice

1. **Enter Reference Text**: Type the transcript of what you'll say in the reference text input field
   - Example: "This is my reference audio for voice cloning."
   
2. **Start Recording**: Click the record button
   - The UI will switch to recording state
   - A panel will display the reference text to read
   
3. **Speak the Reference Text**: Read the displayed text clearly into your microphone

4. **Stop Recording**: Click the record button again
   - A success message will appear when the reference is saved

### Synthesizing Speech

1. **Enter Text to Synthesize**: Type the text you want to convert to speech
   - Example: "Hi ENSF classmates this is demo for our project"

2. **Select TTS Model** (optional): Choose a model from the dropdown

3. **Click Synthesize**: The application will:
   - Download the synthesized audio
   - Play it automatically
   - Trigger character animations if configured


## Project Structure

```
frontend/
├── Assets/
│   ├── Animation/          # Character animations
│   ├── Oculus/             # Oculus LipSync integration
│   ├── Ready Player Me/    # Avatar assets
│   ├── Scenes/             # Unity scenes
│   │   └── SampleScene.unity
│   ├── Scripts/            # C# scripts
│   │   └── UnityAudioManager.cs
│   └── Settings/           # Unity settings
├── Packages/               # Package dependencies
├── ProjectSettings/        # Unity project settings
└── README.md              # This file
```

## Key Components

### UnityAudioManager.cs

The main script that handles:
- Voice recording from microphone
- Audio playback (recorded and synthesized)
- Character animation triggers
- UI state management

**Main Methods:**
- `StartRecording()` - Start voice recording
- `StopRecording()` - Stop recording
- `SynthesizeAndPlay()` - Request TTS synthesis and play result
- `PlayRecordedVoice()` - Play locally recorded audio

## Troubleshooting

### Microphone Issues

- **Error**: "No microphone detected!"
  - Check microphone permissions in system settings
  - Ensure a microphone is connected and enabled
  - On macOS: System Preferences → Security & Privacy → Microphone


### Reference Not Found

- **Error**: "No reference found. Please record your voice first!"
  - Record your voice reference before synthesizing
  - Ensure the reference was successfully saved

### Package Import Issues

- If packages fail to import:
  - Open Unity Package Manager (Window → Package Manager)
  - Click "Refresh" to reload packages
  - Manually install missing packages if needed
