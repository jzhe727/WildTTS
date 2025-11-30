using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using TMPro;
using System;
using System.Collections;
using System.IO;

/// <summary>
/// Unity Audio Manager for Python Backend Communication
/// 
/// Handles three main functions:
/// 1. Record user's voice and send to Python for reference saving (voice cloning)
/// 2. Play back user's recorded voice (local, no server communication)
/// 3. Play synthesized voice downloaded from Python (TTS with voice cloning)
/// </summary>
public class UnityAudioManager : MonoBehaviour
{
    [Header("Server Configuration")]
    [SerializeField] private string serverUrl = "http://localhost:8000";
    [SerializeField] private string userId = "unity_user";

    [Header("TTS Model Settings")]
    [SerializeField] private string ttsModel = "cosyvoice";  // "cosyvoice" or "fishaudio"
    [SerializeField] private bool useDefaultModel = true;     // If true, uses server's default model

    [Header("Audio Components")]
    [SerializeField] private AudioSource playbackAudioSource;
    
    [Header("Recording Settings")]
    [SerializeField] private int recordingLengthSeconds = 10;
    [SerializeField] private int sampleRate = 44100;

    [Header("Reference Settings")]
    [SerializeField] private string referenceText = "This is my reference audio for voice cloning.";

    [Header("Recording UI")]
    public Button recordButton;
    public GameObject recordObject1;  // Ready to record state object
    public GameObject recordObject2;  // Recording state object (black)
    public TMP_InputField referenceTextInput;  // Optional: TMP Input field for reference text
    
    [Header("Reference Text Display")]
    public GameObject referenceTextPanel;  // Panel to show during recording (displays reference text)
    public TextMeshProUGUI referenceTextDisplay;  // TMP Text component to display reference text on the panel

    [Header("Synthesize UI")]
    public Button synthesizeButton;  // Synthesize button - press to synthesize and auto-play
    [SerializeField] private string synthesisText = "Hi ENSF classmates this is demo for our project";  // Text to synthesize
    public TMP_InputField synthesisTextInput;  // Optional: TMP Input field for synthesis text
    public TMP_Dropdown modelDropdown;  // Optional: TMP Dropdown for model selection

    [Header("Animation")]
    public Animator characterAnimator;  // Character animator - controls animation when speaking

    // Private variables
    private AudioClip recordedClip;
    private bool isRecording = false;
    private string lastRecordedFilePath;
    private string lastSynthesizedFilePath;
    private string[] supportedModels = { "cosyvoice", "fishaudio" };

    // Events for UI feedback
    public event Action<string> OnStatusUpdate;
    public event Action<string> OnError;
    public event Action OnRecordingStarted;
    public event Action OnRecordingStopped;
    public event Action OnReferenceSaved;
    public event Action OnSynthesisComplete;

    private void Start()
    {
        // Ensure AudioSource exists
        if (playbackAudioSource == null)
        {
            playbackAudioSource = gameObject.AddComponent<AudioSource>();
        }

        // Recording button setup
        if (recordButton != null)
        {
            recordButton.onClick.AddListener(ToggleRecording);
        }

        // Synthesize button setup
        if (synthesizeButton != null)
        {
            synthesizeButton.onClick.AddListener(SynthesizeAndPlay);
        }

        // Reference text input setup
        if (referenceTextInput != null)
        {
            referenceTextInput.text = referenceText;
            referenceTextInput.onEndEdit.AddListener((text) => referenceText = text);
        }

        // Synthesis text input setup
        if (synthesisTextInput != null)
        {
            synthesisTextInput.text = synthesisText;
            synthesisTextInput.onEndEdit.AddListener((text) => synthesisText = text);
        }

        // Model dropdown setup
        if (modelDropdown != null)
        {
            modelDropdown.ClearOptions();
            modelDropdown.AddOptions(new System.Collections.Generic.List<string>(supportedModels));
            modelDropdown.onValueChanged.AddListener((index) => {
                ttsModel = supportedModels[index];
                useDefaultModel = false;
            });
        }

        // Initialize recording UI state
        InitializeRecordingUI();

        // Fetch supported models from server
        StartCoroutine(FetchSupportedModels());
    }

    /// <summary>
    /// Fetch supported TTS models from server.
    /// </summary>
    private IEnumerator FetchSupportedModels()
    {
        string url = $"{serverUrl}/audio/models";

        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string response = www.downloadHandler.text;
                Debug.Log($"Supported models: {response}");
                
                // Parse JSON response to get supported models
                // Simple parsing - in production use JsonUtility or Newtonsoft.Json
                if (response.Contains("supported_models"))
                {
                    OnStatusUpdate?.Invoke("Connected to server. Models loaded.");
                }
            }
            else
            {
                Debug.LogWarning($"Failed to fetch models: {www.error}");
            }
        }
    }

    /// <summary>
    /// Initialize the recording UI to default state (not recording).
    /// </summary>
    private void InitializeRecordingUI()
    {
        if (recordObject1 != null)
        {
            recordObject1.SetActive(true);
        }

        if (recordObject2 != null)
        {
            recordObject2.SetActive(false);
            
            // Set recordObject2 Image component to black if exists
            Image img = recordObject2.GetComponent<Image>();
            if (img != null)
            {
                img.color = Color.black;
            }
        }

        // Hide reference text panel initially
        if (referenceTextPanel != null)
        {
            referenceTextPanel.SetActive(false);
        }

        // Initialize reference text display
        UpdateReferenceTextDisplay();
    }

    /// <summary>
    /// Update the reference text display on the panel.
    /// </summary>
    private void UpdateReferenceTextDisplay()
    {
        if (referenceTextDisplay != null)
        {
            referenceTextDisplay.text = referenceText;
        }
    }

    /// <summary>
    /// Show the reference text panel.
    /// </summary>
    private void ShowReferenceTextPanel()
    {
        if (referenceTextPanel != null)
        {
            UpdateReferenceTextDisplay();
            referenceTextPanel.SetActive(true);
        }
    }

    /// <summary>
    /// Hide the reference text panel.
    /// </summary>
    private void HideReferenceTextPanel()
    {
        if (referenceTextPanel != null)
        {
            referenceTextPanel.SetActive(false);
        }
    }

    /// <summary>
    /// Toggle recording on button click. 
    /// First click starts recording, second click stops recording.
    /// </summary>
    public void ToggleRecording()
    {
        if (isRecording)
        {
            // If recording, stop recording
            StopRecording(true);
            
            // Switch objects: object2 -> object1
            if (recordObject1 != null) recordObject1.SetActive(true);
            if (recordObject2 != null) recordObject2.SetActive(false);
            
            // Hide reference text panel when recording stops
            HideReferenceTextPanel();
        }
        else
        {
            // If not recording, start recording
            StartRecording();
            
            // Switch objects: object1 -> object2
            if (recordObject1 != null) recordObject1.SetActive(false);
            if (recordObject2 != null) recordObject2.SetActive(true);
            
            // Show reference text panel during recording
            ShowReferenceTextPanel();
        }
    }

    #region ===== FUNCTION 1: Record Voice & Save Reference =====

    /// <summary>
    /// Start recording user's voice using the microphone.
    /// </summary>
    public void StartRecording()
    {
        if (isRecording)
        {
            Debug.LogWarning("Already recording!");
            return;
        }

        if (Microphone.devices.Length == 0)
        {
            string error = "No microphone detected!";
            Debug.LogError(error);
            OnError?.Invoke(error);
            return;
        }

        string micDevice = Microphone.devices[0];
        recordedClip = Microphone.Start(micDevice, false, recordingLengthSeconds, sampleRate);
        isRecording = true;
        
        OnStatusUpdate?.Invoke($"Recording started using: {micDevice}");
        OnRecordingStarted?.Invoke();
        Debug.Log($"Recording started using: {micDevice}");
    }

    /// <summary>
    /// Stop recording and optionally send to server for reference saving.
    /// </summary>
    /// <param name="sendToServer">If true, sends recording to Python for reference</param>
    public void StopRecording(bool sendToServer = true)
    {
        if (!isRecording)
        {
            Debug.LogWarning("Not currently recording!");
            return;
        }

        Microphone.End(null);
        isRecording = false;
        
        OnStatusUpdate?.Invoke("Recording stopped");
        OnRecordingStopped?.Invoke();
        Debug.Log("Recording stopped");

        if (sendToServer && recordedClip != null)
        {
            // Get reference text from input field if available
            string refText = referenceTextInput != null ? referenceTextInput.text : referenceText;
            StartCoroutine(SendRecordingForReference(recordedClip, userId, refText));
        }
    }

    /// <summary>
    /// Send recorded audio to Python server to save as reference for voice cloning.
    /// </summary>
    /// <param name="clip">Recorded audio clip</param>
    /// <param name="usrId">User ID</param>
    /// <param name="refText">Transcript of the recorded audio (required for TTS)</param>
    private IEnumerator SendRecordingForReference(AudioClip clip, string usrId, string refText)
    {
        // Validate reference text
        if (string.IsNullOrEmpty(refText) || string.IsNullOrWhiteSpace(refText))
        {
            string error = "Reference text is required! Please enter the transcript of your recording.";
            OnError?.Invoke(error);
            Debug.LogError(error);
            yield break;
        }

        OnStatusUpdate?.Invoke("Converting audio...");

        // Convert AudioClip to WAV bytes
        byte[] audioData = AudioClipToWav(clip);
        
        if (audioData == null || audioData.Length == 0)
        {
            string error = "Failed to convert audio clip";
            OnError?.Invoke(error);
            yield break;
        }

        // Save locally for playback
        lastRecordedFilePath = Path.Combine(Application.temporaryCachePath, $"{usrId}_recording.wav");
        File.WriteAllBytes(lastRecordedFilePath, audioData);
        Debug.Log($"Recording saved locally: {lastRecordedFilePath}");

        OnStatusUpdate?.Invoke("Sending to server for reference...");

        // Create form with audio file, user_id, and reference_text
        WWWForm form = new WWWForm();
        form.AddBinaryData("audio_file", audioData, $"{usrId}_recording.wav", "audio/wav");
        form.AddField("user_id", usrId);
        form.AddField("reference_text", refText);

        string url = $"{serverUrl}/audio/reference";

        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string response = www.downloadHandler.text;
                Debug.Log($"Reference saved: {response}");
                OnStatusUpdate?.Invoke("Reference audio saved successfully!");
                OnReferenceSaved?.Invoke();
            }
            else
            {
                string error = $"Reference saving failed: {www.error}\n{www.downloadHandler.text}";
                Debug.LogError(error);
                OnError?.Invoke(error);
            }
        }
    }

    /// <summary>
    /// Send an existing audio file to save as reference.
    /// </summary>
    /// <param name="audioData">Raw audio bytes</param>
    /// <param name="filename">Filename for the audio</param>
    /// <param name="refText">Transcript of the audio</param>
    public void SendAudioForReference(byte[] audioData, string filename, string refText)
    {
        StartCoroutine(SendAudioForReferenceCoroutine(audioData, filename, userId, refText));
    }

    private IEnumerator SendAudioForReferenceCoroutine(byte[] audioData, string filename, string usrId, string refText)
    {
        if (string.IsNullOrEmpty(refText))
        {
            OnError?.Invoke("Reference text is required!");
            yield break;
        }

        OnStatusUpdate?.Invoke("Sending audio for reference...");

        WWWForm form = new WWWForm();
        form.AddBinaryData("audio_file", audioData, filename, "audio/wav");
        form.AddField("user_id", usrId);
        form.AddField("reference_text", refText);

        string url = $"{serverUrl}/audio/reference";

        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"Reference saved: {www.downloadHandler.text}");
                OnStatusUpdate?.Invoke("Reference audio saved!");
                OnReferenceSaved?.Invoke();
            }
            else
            {
                OnError?.Invoke($"Failed: {www.error}");
            }
        }
    }

    /// <summary>
    /// Get the saved reference for current user.
    /// </summary>
    public void GetUserReference()
    {
        StartCoroutine(GetUserReferenceCoroutine());
    }

    private IEnumerator GetUserReferenceCoroutine()
    {
        string url = $"{serverUrl}/audio/reference/{userId}";

        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string response = www.downloadHandler.text;
                Debug.Log($"User reference: {response}");
                OnStatusUpdate?.Invoke("Reference found!");
            }
            else
            {
                if (www.responseCode == 404)
                {
                    OnStatusUpdate?.Invoke("No reference found. Please record your voice first.");
                }
                else
                {
                    OnError?.Invoke($"Failed to get reference: {www.error}");
                }
            }
        }
    }

    #endregion

    #region ===== FUNCTION 2: Play Recorded Voice (Local) =====

    /// <summary>
    /// Play the last recorded voice clip locally.
    /// No server communication required.
    /// </summary>
    public void PlayRecordedVoice()
    {
        if (recordedClip != null)
        {
            playbackAudioSource.clip = recordedClip;
            playbackAudioSource.Play();
            OnStatusUpdate?.Invoke("Playing recorded voice...");
            Debug.Log("Playing recorded voice from memory");
        }
        else if (!string.IsNullOrEmpty(lastRecordedFilePath) && File.Exists(lastRecordedFilePath))
        {
            StartCoroutine(LoadAndPlayAudio(lastRecordedFilePath));
        }
        else
        {
            string error = "No recorded voice available to play";
            Debug.LogWarning(error);
            OnError?.Invoke(error);
        }
    }

    /// <summary>
    /// Play a specific audio file from disk.
    /// </summary>
    /// <param name="filePath">Path to the audio file</param>
    public void PlayAudioFile(string filePath)
    {
        if (File.Exists(filePath))
        {
            StartCoroutine(LoadAndPlayAudio(filePath));
        }
        else
        {
            OnError?.Invoke($"File not found: {filePath}");
        }
    }

    private IEnumerator LoadAndPlayAudio(string filePath)
    {
        OnStatusUpdate?.Invoke("Loading audio...");

        using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip("file://" + filePath, AudioType.WAV))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                playbackAudioSource.clip = clip;
                playbackAudioSource.Play();
                OnStatusUpdate?.Invoke("Playing audio...");
                Debug.Log($"Playing audio from: {filePath}");
            }
            else
            {
                OnError?.Invoke($"Failed to load audio: {www.error}");
            }
        }
    }

    #endregion

    #region ===== FUNCTION 3: Request & Play Synthesized Voice =====

    /// <summary>
    /// Request synthesized voice from text and play the result.
    /// </summary>
    /// <param name="text">The text to be synthesized into speech</param>
    /// <param name="model">Optional: TTS model to use ("cosyvoice" or "fishaudio")</param>
    public void RequestSynthesizedVoice(string text, string model = null)
    {
        if (string.IsNullOrEmpty(text))
        {
            OnError?.Invoke("Text cannot be empty");
            return;
        }
        StartCoroutine(SynthesizeAndPlayCoroutine(text, model));
    }

    /// <summary>
    /// Synthesize text using default text and play the result.
    /// </summary>
    public void SynthesizeAndPlay()
    {
        // Get text from input field if available
        string text = synthesisTextInput != null ? synthesisTextInput.text : synthesisText;
        
        if (string.IsNullOrEmpty(text))
        {
            OnError?.Invoke("Synthesis text is not set");
            return;
        }

        // Determine which model to use
        string model = useDefaultModel ? null : ttsModel;
        
        StartCoroutine(SynthesizeAndPlayCoroutine(text, model));
    }

    /// <summary>
    /// Synthesize text to speech and play the result.
    /// </summary>
    /// <param name="text">The text to be synthesized into speech</param>
    /// <param name="model">Optional: TTS model to use</param>
    private IEnumerator SynthesizeAndPlayCoroutine(string text, string model = null)
    {
        OnStatusUpdate?.Invoke("Sending text for synthesis...");

        // Create form with text, user_id, and optional model
        WWWForm form = new WWWForm();
        form.AddField("text", text);
        form.AddField("user_id", userId);
        
        // Add model if specified
        if (!string.IsNullOrEmpty(model))
        {
            form.AddField("model", model);
            Debug.Log($"Using TTS model: {model}");
        }

        string url = $"{serverUrl}/audio/synthesize";

        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                // Get model used from response header
                string modelUsed = www.GetResponseHeader("X-Model-Used") ?? "unknown";
                OnStatusUpdate?.Invoke($"Synthesis complete (model: {modelUsed})! Playing...");
                
                // Get synthesized audio as binary response
                byte[] synthesizedAudio = www.downloadHandler.data;
                
                if (synthesizedAudio == null || synthesizedAudio.Length == 0)
                {
                    OnError?.Invoke("Received empty audio data");
                    Debug.LogError("Received empty audio data");
                    yield break;
                }

                // Get filename from header or use default
                string synthesizedFilename = www.GetResponseHeader("X-Synthesized-Filename") ?? "synthesized.wav";
                Debug.Log($"Received synthesized audio: {synthesizedFilename} ({synthesizedAudio.Length} bytes), model: {modelUsed}");

                // Save locally
                lastSynthesizedFilePath = Path.Combine(Application.temporaryCachePath, synthesizedFilename);
                File.WriteAllBytes(lastSynthesizedFilePath, synthesizedAudio);

                // Play the synthesized audio
                yield return PlaySynthesizedAudio(lastSynthesizedFilePath);
                
                OnSynthesisComplete?.Invoke();
            }
            else
            {
                string error = $"Synthesis failed: {www.error}";
                
                // Check for specific error codes
                if (www.responseCode == 404)
                {
                    error = "No reference found. Please record your voice first!";
                }
                else if (www.responseCode == 400)
                {
                    error = $"Bad request: {www.downloadHandler.text}";
                }
                
                Debug.LogError(error);
                OnError?.Invoke(error);
            }
        }
    }

    private IEnumerator PlaySynthesizedAudio(string filePath)
    {
        // Try loading as different audio types
        AudioType[] audioTypes = { AudioType.WAV, AudioType.MPEG, AudioType.OGGVORBIS, AudioType.UNKNOWN };
        
        foreach (AudioType audioType in audioTypes)
        {
            using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip("file://" + filePath, audioType))
            {
                yield return www.SendWebRequest();

                if (www.result == UnityWebRequest.Result.Success)
                {
                    AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                    if (clip != null)
                    {
                        playbackAudioSource.clip = clip;
                        playbackAudioSource.Play();
                        OnStatusUpdate?.Invoke("Playing synthesized voice!");
                        Debug.Log("Playing synthesized audio");

                        // Start talking animation
                        if (characterAnimator != null)
                        {
                            characterAnimator.SetTrigger("Talking");
                        }

                        // Wait for audio clip duration (more stable)
                        float clipLength = clip.length;
                        Debug.Log($"Audio clip length: {clipLength} seconds");
                        yield return new WaitForSeconds(clipLength);

                        Debug.Log("Audio playback completed");

                        // Stop talking animation
                        if (characterAnimator != null)
                        {
                            characterAnimator.SetTrigger("Stop_Talk");
                        }

                        yield break;
                    }
                }
            }
        }

        OnError?.Invoke("Failed to play synthesized audio");
    }

    /// <summary>
    /// Play the last synthesized voice again.
    /// </summary>
    public void ReplaySynthesizedVoice()
    {
        if (!string.IsNullOrEmpty(lastSynthesizedFilePath) && File.Exists(lastSynthesizedFilePath))
        {
            StartCoroutine(PlaySynthesizedAudio(lastSynthesizedFilePath));
        }
        else
        {
            OnError?.Invoke("No synthesized voice available to replay");
        }
    }

    #endregion

    #region ===== Utility Methods =====

    /// <summary>
    /// Stop any currently playing audio.
    /// </summary>
    public void StopPlayback()
    {
        if (playbackAudioSource.isPlaying)
        {
            playbackAudioSource.Stop();
            
            // Stop talking animation if playing
            if (characterAnimator != null)
            {
                characterAnimator.SetTrigger("Stop_Talk");
            }
            
            OnStatusUpdate?.Invoke("Playback stopped");
        }
    }

    /// <summary>
    /// Check if audio is currently playing.
    /// </summary>
    public bool IsPlaying => playbackAudioSource != null && playbackAudioSource.isPlaying;

    /// <summary>
    /// Check if currently recording.
    /// </summary>
    public bool IsRecording => isRecording;

    /// <summary>
    /// Set the user ID for reference association.
    /// </summary>
    public void SetUserId(string newUserId)
    {
        userId = newUserId;
        Debug.Log($"User ID set to: {userId}");
    }

    /// <summary>
    /// Set the TTS model to use.
    /// </summary>
    /// <param name="model">Model name ("cosyvoice" or "fishaudio")</param>
    public void SetTTSModel(string model)
    {
        ttsModel = model;
        useDefaultModel = false;
        Debug.Log($"TTS model set to: {ttsModel}");
    }

    /// <summary>
    /// Use the server's default TTS model.
    /// </summary>
    public void UseDefaultTTSModel()
    {
        useDefaultModel = true;
        Debug.Log("Using server's default TTS model");
    }

    /// <summary>
    /// Set the reference text (transcript of recorded audio).
    /// </summary>
    public void SetReferenceText(string text)
    {
        referenceText = text;
        if (referenceTextInput != null)
        {
            referenceTextInput.text = text;
        }
        // Update the display panel text as well
        UpdateReferenceTextDisplay();
    }

    /// <summary>
    /// Set the synthesis text.
    /// </summary>
    public void SetSynthesisText(string text)
    {
        synthesisText = text;
        if (synthesisTextInput != null)
        {
            synthesisTextInput.text = text;
        }
    }

    /// <summary>
    /// Convert AudioClip to WAV byte array.
    /// </summary>
    private byte[] AudioClipToWav(AudioClip clip)
    {
        if (clip == null) return null;

        float[] samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);

        // Convert to 16-bit PCM
        short[] intData = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            intData[i] = (short)(samples[i] * 32767f);
        }

        byte[] bytesData = new byte[intData.Length * 2];
        Buffer.BlockCopy(intData, 0, bytesData, 0, bytesData.Length);

        // Create WAV file
        using (MemoryStream stream = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(stream))
        {
            // RIFF header
            writer.Write(new char[] { 'R', 'I', 'F', 'F' });
            writer.Write(36 + bytesData.Length);
            writer.Write(new char[] { 'W', 'A', 'V', 'E' });

            // fmt chunk
            writer.Write(new char[] { 'f', 'm', 't', ' ' });
            writer.Write(16); // Subchunk1Size
            writer.Write((short)1); // AudioFormat (PCM)
            writer.Write((short)clip.channels);
            writer.Write(clip.frequency);
            writer.Write(clip.frequency * clip.channels * 2); // ByteRate
            writer.Write((short)(clip.channels * 2)); // BlockAlign
            writer.Write((short)16); // BitsPerSample

            // data chunk
            writer.Write(new char[] { 'd', 'a', 't', 'a' });
            writer.Write(bytesData.Length);
            writer.Write(bytesData);

            return stream.ToArray();
        }
    }

    /// <summary>
    /// Check server connection and get server info.
    /// </summary>
    public void CheckServerConnection()
    {
        StartCoroutine(CheckServerCoroutine());
    }

    private IEnumerator CheckServerCoroutine()
    {
        using (UnityWebRequest www = UnityWebRequest.Get(serverUrl))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                OnStatusUpdate?.Invoke("Server connected!");
                Debug.Log($"Server response: {www.downloadHandler.text}");
            }
            else
            {
                OnError?.Invoke($"Server connection failed: {www.error}");
            }
        }
    }

    /// <summary>
    /// Get list of supported TTS models from server.
    /// </summary>
    public void GetSupportedModels(Action<string[]> callback)
    {
        StartCoroutine(GetSupportedModelsCoroutine(callback));
    }

    private IEnumerator GetSupportedModelsCoroutine(Action<string[]> callback)
    {
        string url = $"{serverUrl}/audio/models";

        using (UnityWebRequest www = UnityWebRequest.Get(url))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                // Parse response - in production use proper JSON parsing
                string response = www.downloadHandler.text;
                Debug.Log($"Models response: {response}");
                
                // Return supported models
                callback?.Invoke(supportedModels);
            }
            else
            {
                Debug.LogError($"Failed to get models: {www.error}");
                callback?.Invoke(null);
            }
        }
    }

    #endregion
}
