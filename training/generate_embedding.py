import argparse
import os
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import numpy as np

def generate_speaker_embedding(audio_path, campplus_model_path, output_path):
    """
    Generates a speaker embedding from an audio file using the Campplus ONNX model.
    """
    
    # 1. Load Audio
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    speech, sample_rate = torchaudio.load(audio_path)
    
    # 2. Resample to 16000 Hz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        speech = resampler(speech)
    
    # Ensure it's mono
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
        
    # 3. Extract Features (Fbank)
    # Matches CosyVoiceFrontEnd._extract_spk_embedding logic
    feat = kaldi.fbank(speech,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    # 4. Load Campplus Model
    if not os.path.exists(campplus_model_path):
        raise FileNotFoundError(f"Campplus model not found: {campplus_model_path}")
        
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    
    # Use CPU provider for simplicity as requested (minimal)
    session = onnxruntime.InferenceSession(campplus_model_path, sess_options=option, providers=["CPUExecutionProvider"])
    
    # 5. Run Inference
    embedding = session.run(None,
                            {session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    
    embedding_tensor = torch.tensor([embedding])
    
    # 6. Save Embedding
    torch.save(embedding_tensor, output_path)
    print(f"Embedding saved to {output_path}")
    print(f"Embedding shape: {embedding_tensor.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speaker embedding using Campplus model.")
    parser.add_argument("--audio", required=True, help="Path to the input audio file.")
    parser.add_argument("--model", required=True, help="Path to the campplus.onnx model.")
    parser.add_argument("--output", required=True, help="Path to save the output embedding (.pt file).")
    
    args = parser.parse_args()
    
    generate_speaker_embedding(args.audio, args.model, args.output)
