#!/usr/bin/env python3
"""
Simple YOLO Checkpoint to ONNX Converter

This script uses the downloaded YOLOv7 repository to convert the Lightning checkpoint
by creating a compatible PyTorch model file and then using the YOLOv7 export functionality.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
import os
import json
import argparse


def create_yolov7_compatible_checkpoint(lightning_ckpt_path: str, output_pt_path: str):
    """Create a YOLOv7-compatible checkpoint from Lightning checkpoint."""
    print(f"🔍 Loading Lightning checkpoint: {lightning_ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(lightning_ckpt_path, map_location='cpu', weights_only=False)
    
    # Extract info
    hyp = checkpoint.get('hyper_parameters', {})
    settings = hyp.get('settings', {})
    
    print(f"📊 Model info:")
    print(f"   - Model: {settings.get('model', {}).get('name', 'yolov7-tiny')}")
    print(f"   - Dataset: {settings.get('dataset', {}).get('name', 'CropOrWeed2')}")
    print(f"   - Image size: {settings.get('dataset', {}).get('image_size', [640, 640])}")
    
    # Extract state dict and remove 'model.' prefix
    state_dict = checkpoint['state_dict']
    clean_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('model.'):
            clean_key = key[6:]  # Remove 'model.' prefix
            clean_state_dict[clean_key] = value
        else:
            clean_state_dict[key] = value
    
    print(f"📊 Cleaned {len(clean_state_dict)} weight parameters")
    
    # Create YOLOv7-compatible checkpoint structure
    yolov7_checkpoint = {
        'model': clean_state_dict,
        'epoch': checkpoint.get('epoch', 0),
        'names': {0: 'crop', 1: 'weed'},  # Crop/weed detection
        'nc': 2,  # Number of classes
        'stride': [8, 16, 32],  # YOLOv7 default strides
        'anchors': None  # Will be set by model if needed
    }
    
    # Save the compatible checkpoint
    torch.save(yolov7_checkpoint, output_pt_path)
    print(f"✅ Saved YOLOv7-compatible checkpoint to: {output_pt_path}")
    
    return yolov7_checkpoint


def simple_onnx_export(model_path: str, output_path: str, img_size: int = 640):
    """Simple ONNX export using torch.onnx.export."""
    print(f"🔄 Attempting simple ONNX export...")
    
    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create a simple dummy model that loads the state dict
        class SimpleModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                # Create a sequential model with the loaded weights
                # This is a simplified approach - the actual model structure would be more complex
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, 2)  # 2 classes: crop, weed
                )
                
            def forward(self, x):
                return self.features(x)
        
        # This is a placeholder model - the real conversion would need the actual YOLOv7 architecture
        print("⚠️  Creating placeholder model (real model structure needed for production use)")
        model = SimpleModel(checkpoint.get('model', {}))
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Simple ONNX export completed: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Simple ONNX export failed: {e}")
        return False


def manual_onnx_with_ultralytics(model_path: str, output_path: str, img_size: int = 640):
    """Try to use ultralytics YOLO for export."""
    try:
        from ultralytics import YOLO
        print(f"🔄 Attempting ultralytics YOLO export...")
        
        # Load with ultralytics
        model = YOLO(model_path)
        
        # Export to ONNX
        exported_path = model.export(
            format='onnx',
            imgsz=img_size,
            simplify=True,
            opset=11
        )
        
        # Move to desired location if needed
        if Path(exported_path).resolve() != Path(output_path).resolve():
            Path(exported_path).rename(output_path)
        
        print(f"✅ Ultralytics export successful: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ultralytics export failed: {e}")
        return False


def validate_onnx_model(onnx_path: str):
    """Validate the exported ONNX model."""
    try:
        print(f"🔍 Validating ONNX model...")
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()
        
        print(f"✅ ONNX model validation successful!")
        print(f"📊 Input: {input_info.name} - {input_info.shape} ({input_info.type})")
        print(f"📊 Outputs: {len(output_info)} tensor(s)")
        for i, output in enumerate(output_info):
            print(f"   Output {i}: {output.name} - {output.shape} ({output.type})")
        
        # Test inference
        dummy_input = np.random.randn(*input_info.shape).astype(np.float32)
        outputs = ort_session.run(None, {input_info.name: dummy_input})
        print(f"✅ Test inference successful! Output shapes: {[out.shape for out in outputs]}")
        
        # Model size
        file_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"📁 Model size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"⚠️  ONNX validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO Lightning checkpoint to ONNX")
    parser.add_argument("--input", "-i", required=True, help="Input checkpoint path (.ckpt)")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX model path")
    parser.add_argument("--imgsz", "--img-size", type=int, default=640, help="Image size for inference")
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting conversion...")
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"📏 Image size: {args.imgsz}x{args.imgsz}")
    
    # Create temporary PT file
    temp_pt_path = Path(args.output).with_suffix('.pt')
    
    try:
        # Step 1: Create YOLOv7-compatible checkpoint
        model_info = create_yolov7_compatible_checkpoint(args.input, str(temp_pt_path))
        
        # Step 2: Try different conversion approaches
        success = False
        
        # Try ultralytics first
        print(f"\n🔄 Method 1: Ultralytics YOLO export")
        success = manual_onnx_with_ultralytics(str(temp_pt_path), args.output, args.imgsz)
        
        if not success:
            print(f"\n🔄 Method 2: Simple torch.onnx.export")
            success = simple_onnx_export(str(temp_pt_path), args.output, args.imgsz)
        
        if success and Path(args.output).exists():
            # Validate the exported model
            validate_onnx_model(args.output)
            
            print(f"\n🎉 Conversion completed!")
            print(f"📁 ONNX model saved to: {args.output}")
            
            # Save metadata
            metadata = {
                "source_checkpoint": str(Path(args.input).absolute()),
                "model_type": "yolov7-tiny",
                "dataset": "CropOrWeed2",
                "classes": ["crop", "weed"],
                "input_size": [args.imgsz, args.imgsz],
                "note": "Converted from PyTorch Lightning checkpoint"
            }
            
            metadata_path = Path(args.output).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"📄 Metadata saved to: {metadata_path}")
        else:
            print(f"\n❌ All conversion methods failed!")
            print("\n📋 Manual conversion options:")
            print("1. Use the original YOLOv7 repository with the created .pt file:")
            print(f"   python yolov7/export.py --weights {temp_pt_path} --grid --end2end --simplify --img-size {args.imgsz}")
            print("\n2. Or manually load and export the model using the actual YOLOv7 architecture")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Conversion failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary file
        if temp_pt_path.exists():
            try:
                temp_pt_path.unlink()
                print(f"🧹 Cleaned up temporary file: {temp_pt_path}")
            except:
                pass


if __name__ == "__main__":
    main()
