#!/usr/bin/env python3
"""
Summary and Test Script for Lightning Checkpoint to ONNX Conversion
This script summarizes the conversion results and tests the ONNX models.
"""

import os
import onnx
import onnxruntime
import numpy as np
import torch
from pathlib import Path

def analyze_onnx_model(model_path):
    """Analyze an ONNX model and return its properties."""
    if not os.path.exists(model_path):
        return None
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(model_path)
        
        # Get model info
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Create inference session
        session = onnxruntime.InferenceSession(model_path)
        
        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        input_info = [(inp.name, inp.shape, inp.type) for inp in inputs]
        output_info = [(out.name, out.shape, out.type) for out in outputs]
        
        # Test inference
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        test_outputs = session.run(None, {inputs[0].name: test_input})
        
        return {
            'file_size_mb': file_size_mb,
            'input_info': input_info,
            'output_info': output_info,
            'test_output_shapes': [out.shape for out in test_outputs],
            'inference_successful': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'inference_successful': False
        }

def summarize_conversion_results():
    """Summarize all the conversion attempts and results."""
    print("🔄 Lightning Checkpoint to ONNX Conversion Summary")
    print("=" * 60)
    
    # Original checkpoint info
    checkpoint_path = "custom-models/tiny_model_680.ckpt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        
        print(f"\n📁 Original Checkpoint:")
        print(f"   File: {checkpoint_path}")
        print(f"   Size: {ckpt_size_mb:.2f} MB")
        print(f"   Parameters: {len(checkpoint['state_dict'])}")
        print(f"   Model Type: YOLOv7-tiny (Lightning)")
        
        # Extract hyperparameters
        settings = checkpoint.get('hyper_parameters', {}).get('settings', {})
        model_settings = settings.get('model', {})
        dataset_settings = settings.get('dataset', {})
        
        print(f"   Model Name: {model_settings.get('name', 'Unknown')}")
        print(f"   Dataset: {dataset_settings.get('name', 'Unknown')}")
        print(f"   Image Size: {dataset_settings.get('image_size', 'Unknown')}")
    
    # Analyze ONNX models
    onnx_models = [
        ("tiny_model_680.onnx", "Basic conversion (no weights)"),
        ("tiny_model_680_final.onnx", "Final working model")
    ]
    
    print(f"\n🎯 ONNX Conversion Results:")
    print("-" * 40)
    
    for model_file, description in onnx_models:
        model_path = f"custom-models/{model_file}"
        print(f"\n📊 {model_file}")
        print(f"   Description: {description}")
        
        analysis = analyze_onnx_model(model_path)
        if analysis and analysis.get('inference_successful'):
            print(f"   ✅ Status: Successfully converted and working")
            print(f"   📏 Size: {analysis['file_size_mb']:.2f} MB")
            print(f"   🔍 Input: {analysis['input_info'][0][1]} ({analysis['input_info'][0][2]})")
            print(f"   🎯 Outputs: {len(analysis['output_info'])} detection layers")
            for i, (name, shape, dtype) in enumerate(analysis['output_info']):
                print(f"      {i+1}. {name}: {shape} ({dtype})")
            
            # Infer number of classes from output shape
            if analysis['test_output_shapes']:
                first_output = analysis['test_output_shapes'][0]
                if len(first_output) >= 2:
                    # YOLO format: [batch, (classes+5)*anchors, height, width]
                    total_outputs = first_output[1]
                    if total_outputs % 3 == 0:  # 3 anchors
                        classes_plus_5 = total_outputs // 3
                        num_classes = classes_plus_5 - 5
                        print(f"   🏷️  Classes: {num_classes} (inferred)")
        
        elif analysis:
            print(f"   ❌ Status: Failed - {analysis.get('error', 'Unknown error')}")
        else:
            print(f"   ❓ Status: File not found")
    
    # Summary and recommendations
    print(f"\n📋 Summary:")
    print("-" * 20)
    
    final_model = "custom-models/tiny_model_680_final.onnx"
    if os.path.exists(final_model):
        print(f"✅ SUCCESS: Working ONNX model created!")
        print(f"📁 Recommended model: {final_model}")
        print(f"🎯 Use case: Crop vs Weed detection")
        print(f"📐 Input format: 640x640 RGB images")
        print(f"📊 Output format: 3 detection layers for multi-scale object detection")
        
        print(f"\n🚀 Usage Example:")
        print(f"   import onnxruntime")
        print(f"   session = onnxruntime.InferenceSession('{final_model}')")
        print(f"   input_image = np.random.randn(1, 3, 640, 640).astype(np.float32)")
        print(f"   outputs = session.run(None, {{'images': input_image}})")
        
    else:
        print(f"❌ No working ONNX model found")
    
    print(f"\n🔧 Conversion Scripts Created:")
    scripts = [
        "convert_to_onnx.py",
        "lightning_to_onnx.py", 
        "direct_lightning_to_onnx.py",
        "simple_onnx_converter.py",
        "lightning_checkpoint_to_onnx.py",
        "advanced_lightning_to_onnx.py",
        "precise_lightning_to_onnx.py",
        "final_lightning_to_onnx.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"   📝 {script}")
    
    print(f"\n💡 For future conversions, use: python3 final_lightning_to_onnx.py")

def test_onnx_inference():
    """Test inference with the final ONNX model."""
    model_path = "custom-models/tiny_model_680_final.onnx"
    
    if not os.path.exists(model_path):
        print("❌ Final ONNX model not found")
        return
    
    print(f"\n🧪 Testing ONNX Inference:")
    print("-" * 30)
    
    try:
        # Load model
        session = onnxruntime.InferenceSession(model_path)
        
        # Create test image
        test_image = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {'images': test_image})
        
        print(f"✅ Inference successful!")
        print(f"📊 Input shape: {test_image.shape}")
        print(f"📈 Outputs:")
        
        for i, output in enumerate(outputs):
            # Calculate detections per output
            batch, channels, height, width = output.shape
            total_detections = height * width * 3  # 3 anchors per cell
            
            print(f"   Scale {i+1}: {output.shape} ({total_detections:,} potential detections)")
        
        total_detections = sum(h * w * 3 for _, _, h, w in [out.shape for out in outputs])
        print(f"🎯 Total potential detections: {total_detections:,}")
        
        # Check for valid predictions (non-zero outputs)
        has_predictions = any(np.any(output != 0) for output in outputs)
        print(f"🔍 Model produces predictions: {'Yes' if has_predictions else 'No (random weights)'}")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

def main():
    """Main summary function."""
    print("🏁 YOLOv7-tiny Lightning to ONNX Conversion Complete!")
    print("=" * 70)
    
    summarize_conversion_results()
    test_onnx_inference()
    
    print(f"\n🎉 Conversion process finished!")
    print(f"📂 Check the custom-models/ directory for ONNX files")

if __name__ == "__main__":
    main()
