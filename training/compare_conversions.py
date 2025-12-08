"""
Simple comparison script for CAM++ model conversion methods.
Tests: ONNX Runtime, onnx2torch, and onnx2pytorch
"""

import sys
import torch
import numpy as np
import onnxruntime as ort
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from pathlib import Path

def preprocess_audio(audio_path: str):
    """Preprocess audio exactly like generate_embedding.py does."""
    # 1. Load Audio
    speech, sample_rate = torchaudio.load(audio_path)
    
    # 2. Resample to 16000 Hz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        speech = resampler(speech)
    
    # Ensure it's mono
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
        
    # 3. Extract Features (Fbank)
    feat = kaldi.fbank(speech,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    
    return feat


def test_onnx_runtime(onnx_path: str, input_data: np.ndarray):
    """Test with ONNX Runtime (baseline)."""
    print("\n=== Testing ONNX Runtime ===")
    
    option = ort.SessionOptions()
    option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    
    session = ort.InferenceSession(onnx_path, sess_options=option, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"Output: {output_name}, shape: {session.get_outputs()[0].shape}")
    
    output = session.run([output_name], {input_name: input_data})[0]
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}")
    
    return output

def test_onnx2torch(onnx_path: str, input_tensor: torch.Tensor):
    """Test with onnx2torch."""
    print("\n=== Testing onnx2torch ===")
    try:
        from onnx2torch import convert
        
        model = convert(onnx_path)
        
        # Introspect the model structure
        print("\n--- Model Structure Introspection ---")
        print(f"Model type: {type(model)}")
        print(f"Model training mode: {model.training}")
        
        # Find all BatchNorm layers
        bn_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                bn_layers.append((name, module))
        
        if bn_layers:
            print(f"\nFound {len(bn_layers)} BatchNorm layers:")
            for name, module in bn_layers[:5]:  # Show first 5
                print(f"  {name}: {module}")
        
        # Try to inspect structure depth
        total_modules = sum(1 for _ in model.modules())
        print(f"Total modules in model: {total_modules}")
        
        # CRITICAL: Must use eval() mode with batch_size=1, otherwise BatchNorm fails
        # In training mode with batch_size=1, BatchNorm can't compute statistics
        print("\n⚠ Setting model to eval() mode (required for batch_size=1)")
        model.eval()
        
        print("\n--- Running Forward Pass ---")
        print(f"Input shape: {input_tensor.shape}")
        
        with torch.no_grad():
            output = model(input_tensor)
        
        output_np = output.cpu().numpy()
        print(f"✓ Conversion successful")
        print(f"Output shape: {output_np.shape}")
        print(f"Output sample: {output_np[0, :5]}")
        
        return output_np, model
        
    except Exception as e:
        import traceback
        print(f"✗ Failed: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        return None, None

def test_model_py(onnx_path: str, input_tensor: torch.Tensor):
    """Test using custom model wrapper."""
    print("\n=== Testing custom model ===")
    try:
        sys.path.append(str(Path(__file__).parent / "campplus_finetuning"))
        from model import CampplusFinetunableModel
        model = CampplusFinetunableModel(onnx_path)
        model.train()
        model.unfreeze_all()
        model.freeze_batchnorm()  # Freeze BatchNorm for batch_size=1 training

        print("\n--- Running Forward Pass ---")
        print(f"Input shape: {input_tensor.shape}")
        with torch.no_grad():
            output = model(input_tensor)
        output_np = output.cpu().numpy()
        print(f"✓ Conversion successful")
        print(f"Output shape: {output_np.shape}")
        print(f"Output sample: {output_np[0, :5]}")
        return output_np, model
    except Exception as e:
        import traceback
        print(f"✗ Failed: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        return None, None



def compare_outputs(onnx_out, torch_out, pytorch_out, exported_out=None):
    """Compare outputs from all three methods."""
    print("\n=== Comparison Results ===")
    
    if torch_out is not None:
        diff_torch = np.abs(onnx_out - torch_out).max()
        print(f"ONNX vs onnx2torch max diff: {diff_torch:.6e}")
        print(f"  Are they close? {np.allclose(onnx_out, torch_out, atol=1e-5)}")
        # Calculate cosine similarity as well
        cos_sim = np.dot(onnx_out.flatten(), torch_out.flatten()) / (np.linalg.norm(onnx_out.flatten()) * np.linalg.norm(torch_out.flatten()))
        print(f"  Cosine similarity: {cos_sim:.6e}")
    
    if pytorch_out is not None:
        diff_pytorch = np.abs(onnx_out - pytorch_out).max()
        print(f"ONNX vs onnx2pytorch max diff: {diff_pytorch:.6e}")
        print(f"  Are they close? {np.allclose(onnx_out, pytorch_out, atol=1e-5)}")
    
    if torch_out is not None and pytorch_out is not None:
        diff_both = np.abs(torch_out - pytorch_out).max()
        print(f"onnx2torch vs onnx2pytorch max diff: {diff_both:.6e}")
    
    if exported_out is not None:
        diff_exported = np.abs(onnx_out - exported_out).max()
        print(f"ONNX vs exported ONNX max diff: {diff_exported:.6e}")
        print(f"  Are they close? {np.allclose(onnx_out, exported_out, atol=1e-5)}")

def test_gradient_flow(model, input_tensor: torch.Tensor):
    """Test if gradients flow through the model."""
    print("\n=== Testing Gradient Flow ===")
    
    if model is None:
        print("Model is None, skipping")
        return False
    

    input_tensor.requires_grad = True
    try:
        output = model(input_tensor)
    except Exception as e:
        import traceback
        print(f"✗ Failed: {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        return None, None
    
    # Simple backward pass
    loss = output.sum()
    loss.backward()
    
    has_grad = input_tensor.grad is not None and input_tensor.grad.abs().sum() > 0
    print(f"Gradient flow: {'✓ Working' if has_grad else '✗ Not working'}")
    
    return has_grad

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare CAM++ model conversion methods")
    parser.add_argument("--audio", default="./zero_shot_prompt.wav", help="Path to input audio file")
    parser.add_argument("--model", default="./campplus.onnx", help="Path to campplus ONNX model")
    
    args = parser.parse_args()
    
    onnx_path = args.model
    audio_path = args.audio

    
    print("=== CAM++ Model Conversion Comparison ===")
    print(f"Model: {onnx_path}")
    print(f"Audio: {audio_path}")

    
    # Preprocess audio using same logic as generate_embedding.py
    print("\n=== Preprocessing Audio ===")
    feat = preprocess_audio(audio_path)
    print(f"Feature shape: {feat.shape}")
    
    # Prepare inputs for all three methods
    # ONNX expects batch dimension
    input_np = feat.unsqueeze(0).cpu().numpy()
    input_torch = feat.unsqueeze(0)
    
    print(f"Input shape for models: {input_np.shape}")
    
    # Test all three methods
    onnx_output = test_onnx_runtime(onnx_path, input_np)
    torch_output, torch_model = test_onnx2torch(onnx_path, input_torch)
    custom_model_output, custom_model = test_model_py(onnx_path, input_torch)

    from train_simple import export_onnx_model
    # test exporting the original model
    exported_path = "./test_exported.onnx"
    export_onnx_model(custom_model, exported_path)

    exported_output = test_onnx_runtime(exported_path, input_np)
    
    # Compare outputs
    compare_outputs(onnx_output, torch_output, custom_model_output, exported_output)
    
    # Test gradients
    if torch_model is not None:
        print("\n--- onnx2torch gradient test ---")
        test_gradient_flow(torch_model, input_torch.clone())
    
    if custom_model is not None:
        print("\n--- onnx2pytorch gradient test ---")
        test_gradient_flow(custom_model, input_torch.clone())
    
    # Recommendation
    print("\n=== Recommendation ===")
    if torch_output is not None and np.allclose(onnx_output, torch_output, atol=1e-5):
        print("✓ onnx2torch: Working correctly")
    if custom_model_output is not None and np.allclose(onnx_output, custom_model_output, atol=1e-5):
        print("✓ onnx2pytorch: Working correctly")



if __name__ == "__main__":
    # Install dependencies first:
    # pip install onnx onnxruntime onnx2torch onnx-pytorch torchaudio
    main()
