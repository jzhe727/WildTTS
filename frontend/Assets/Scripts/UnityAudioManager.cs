using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using System;
using System.Collections;
using System.IO;

/// <summary>
/// Unity Audio Manager for Python Backend Communication
/// 
/// Handles three main functions:
/// 1. Record user's voice and send to Python for embedding generation
/// 2. Play back user's recorded voice (local, no server communication)
/// 3. Play synthesized voice downloaded from Python
/// </summary>
public class UnityAudioManager : MonoBehaviour
{
    [Header("Server Configuration")]
    [SerializeField] private string serverUrl = "http://localhost:8000";
    [SerializeField] private string userId = "unity_user";

    [Header("Audio Components")]
    [SerializeField] private AudioSource playbackAudioSource;
    
    [Header("Recording Settings")]
    [SerializeField] private int recordingLengthSeconds = 10;
    [SerializeField] private int sampleRate = 44100;

    [Header("Recording UI")]
    public Button recordButton;
    public GameObject recordObject1;  // 녹음 대기 상태 오브젝트
    public GameObject recordObject2;  // 녹음 중 상태 오브젝트 (검정색)

    [Header("Synthesize UI")]
    public Button synthesizeButton;  // 합성 버튼 - 누르면 녹음된 음성을 합성하여 자동 재생

    [Header("Animation")]
    public Animator characterAnimator;  // 캐릭터 애니메이터 - 말할 때 애니메이션 제어

    // Private variables
    private AudioClip recordedClip;
    private bool isRecording = false;
    private string lastRecordedFilePath;
    private string lastSynthesizedFilePath;

    // Events for UI feedback
    public event Action<string> OnStatusUpdate;
    public event Action<string> OnError;
    public event Action OnRecordingStarted;
    public event Action OnRecordingStopped;
    public event Action OnEmbeddingGenerated;
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
            synthesizeButton.onClick.AddListener(SynthesizeRecordedVoice);
        }

        // Initialize recording UI state
        InitializeRecordingUI();
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
            
            // recordObject2에 Image 컴포넌트가 있으면 검정색으로 설정
            Image img = recordObject2.GetComponent<Image>();
            if (img != null)
            {
                img.color = Color.black;
            }
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
            // 녹음 중이면 녹음 종료
            StopRecording(true);
            
            // 오브젝트 전환: 오브젝트2 -> 오브젝트1
            if (recordObject1 != null) recordObject1.SetActive(true);
            if (recordObject2 != null) recordObject2.SetActive(false);
        }
        else
        {
            // 녹음 중이 아니면 녹음 시작
            StartRecording();
            
            // 오브젝트 전환: 오브젝트1 -> 오브젝트2
            if (recordObject1 != null) recordObject1.SetActive(false);
            if (recordObject2 != null) recordObject2.SetActive(true);
        }
    }

    #region ===== FUNCTION 1: Record Voice & Generate Embedding =====

    /// <summary>
    /// Start recording user's voice using the microphone.
    /// </summary>
    /// 
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
    /// Stop recording and optionally send to server for embedding generation.
    /// </summary>
    /// <param name="sendToServer">If true, sends recording to Python for embedding</param>
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
            StartCoroutine(SendRecordingForEmbedding(recordedClip, userId));
        }
    }

    /// <summary>
    /// Send recorded audio to Python server to generate user embedding.
    /// </summary>
    private IEnumerator SendRecordingForEmbedding(AudioClip clip, string usrId)
    {
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

        OnStatusUpdate?.Invoke("Sending to server for embedding...");

        // Send to server
        WWWForm form = new WWWForm();
        form.AddBinaryData("audio_file", audioData, $"{usrId}_recording.wav", "audio/wav");

        string url = $"{serverUrl}/audio/embedding?user_id={usrId}";

        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string response = www.downloadHandler.text;
                Debug.Log($"Embedding generated: {response}");
                OnStatusUpdate?.Invoke("User embedding generated successfully!");
                OnEmbeddingGenerated?.Invoke();
            }
            else
            {
                string error = $"Embedding generation failed: {www.error}";
                Debug.LogError(error);
                OnError?.Invoke(error);
            }
        }
    }

    /// <summary>
    /// Send an existing audio file to generate embedding.
    /// </summary>
    /// <param name="audioData">Raw audio bytes</param>
    /// <param name="filename">Filename for the audio</param>
    public void SendAudioForEmbedding(byte[] audioData, string filename)
    {
        StartCoroutine(SendAudioForEmbeddingCoroutine(audioData, filename, userId));
    }

    private IEnumerator SendAudioForEmbeddingCoroutine(byte[] audioData, string filename, string usrId)
    {
        OnStatusUpdate?.Invoke("Sending audio for embedding...");

        WWWForm form = new WWWForm();
        form.AddBinaryData("audio_file", audioData, filename, "audio/wav");

        string url = $"{serverUrl}/audio/embedding?user_id={usrId}";

        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"Embedding generated: {www.downloadHandler.text}");
                OnStatusUpdate?.Invoke("User embedding generated!");
                OnEmbeddingGenerated?.Invoke();
            }
            else
            {
                OnError?.Invoke($"Failed: {www.error}");
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
    /// Send audio to Python for synthesis and play the result.
    /// </summary>
    /// <param name="audioData">Input audio bytes to synthesize</param>
    /// <param name="filename">Filename for the input audio</param>
    public void RequestSynthesizedVoice(byte[] audioData, string filename)
    {
        StartCoroutine(SynthesizeAndPlayCoroutine(audioData, filename));
    }

    /// <summary>
    /// Send the last recorded voice for synthesis and play the result.
    /// </summary>
    public void SynthesizeRecordedVoice()
    {
        if (recordedClip != null)
        {
            byte[] audioData = AudioClipToWav(recordedClip);
            StartCoroutine(SynthesizeAndPlayCoroutine(audioData, $"{userId}_recording.wav"));
        }
        else if (!string.IsNullOrEmpty(lastRecordedFilePath) && File.Exists(lastRecordedFilePath))
        {
            byte[] audioData = File.ReadAllBytes(lastRecordedFilePath);
            StartCoroutine(SynthesizeAndPlayCoroutine(audioData, Path.GetFileName(lastRecordedFilePath)));
        }
        else
        {
            OnError?.Invoke("No recorded voice available to synthesize");
        }
    }

    private IEnumerator SynthesizeAndPlayCoroutine(byte[] audioData, string filename)
    {
        OnStatusUpdate?.Invoke("Sending audio for synthesis...");

        WWWForm form = new WWWForm();
        form.AddBinaryData("audio_file", audioData, filename, "audio/wav");

        // Include user_id to use their embedding for synthesis
        string url = $"{serverUrl}/audio/synthesize?user_id={userId}";

        using (UnityWebRequest www = UnityWebRequest.Post(url, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                OnStatusUpdate?.Invoke("Synthesis complete! Playing...");
                
                // Get synthesized audio
                byte[] synthesizedAudio = www.downloadHandler.data;
                
                // Get filename from header
                string synthesizedFilename = www.GetResponseHeader("X-Synthesized-Filename") ?? "synthesized.wav";
                Debug.Log($"Received synthesized audio: {synthesizedFilename} ({synthesizedAudio.Length} bytes)");

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

                        // 말하기 애니메이션 시작
                        if (characterAnimator != null)
                        {
                            characterAnimator.SetTrigger("Talking");
                        }

                        // 오디오 클립 길이만큼 대기 (더 안정적)
                        float clipLength = clip.length;
                        Debug.Log($"Audio clip length: {clipLength} seconds");
                        yield return new WaitForSeconds(clipLength);

                        Debug.Log("Audio playback completed");

                        // 말하기 애니메이션 종료
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
    /// Set the user ID for embedding association.
    /// </summary>
    public void SetUserId(string newUserId)
    {
        userId = newUserId;
        Debug.Log($"User ID set to: {userId}");
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
    /// Check server connection.
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

    #endregion
}

