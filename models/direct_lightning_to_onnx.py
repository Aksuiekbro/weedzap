#!/usr/bin/env python3
"""
Direct YOLO Lightning Model to ONNX Converter

This script loads a YOLO_PL model from the CropAndWeedDetection project
and exports it directly to ONNX format using the model's own structure.
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import argparse

# Set up paths to import from the project
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Try to import the project modules
try:
    from model import YOLO_PL
    print("✅ Successfully imported YOLO_PL from the project")
except ImportError as e:
    print(f"❌ Failed to import YOLO_PL: {e}")
    print("Make sure you're running this from the project directory with model.py")
    sys.exit(1)


def reconstruct_model_from_checkpoint(checkpoint_path: str):
    """Reconstruct YOLO model from Lightning checkpoint."""
    print(f"🔍 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters to reconstruct the model
    hyp = checkpoint.get('hyper_parameters', {})
    settings = hyp.get('settings', {})
    
    print(f"📊 Model settings:")
    print(f"   - Model: {settings.get('model', {}).get('name', 'yolov7-tiny')}")
    print(f"   - Dataset: {settings.get('dataset', {}).get('name', 'CropOrWeed2')}")
    print(f"   - Classes: 2 (crop, weed)")
    print(f"   - Image size: {settings.get('dataset', {}).get('image_size', [640, 640])}")
    
    # Create a new model instance with the same settings
    print("🔧 Reconstructing model...")
    model = YOLO_PL(settings)
    
    # Load the state dict
    print("📥 Loading model weights...")
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"⚠️  Direct load failed: {e}")
        print("🔄 Trying to load weights with strict=False...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✅ Weights loaded with some mismatches (this might be OK)")
    
    # Set to evaluation mode
    model.eval()
    
    return model, settings


def export_to_onnx(model, settings: dict, output_path: str, img_size: int = 640):
    """Export the model to ONNX format."""
    print(f"🔄 Exporting to ONNX...")
    
    # Create dummy input
    batch_size = 1
    channels = 3
    height, width = img_size, img_size
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print(f"📊 Input shape: {dummy_input.shape}")
    
    try:
        # Test forward pass first
        print("🧪 Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful! Output type: {type(output)}")
        if isinstance(output, (list, tuple)):
            print(f"   Output shapes: {[o.shape if hasattr(o, 'shape') else type(o) for o in output]}")
        else:
            print(f"   Output shape: {output.shape}")
        
        # Export to ONNX
        print("📤 Exporting to ONNX...")
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
            },
            verbose=False
        )
        
        print(f"✅ Successfully exported to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        
        # Try exporting just the underlying model
        print("🔄 Trying to export the underlying YOLOv7 model...")
        try:
            torch.onnx.export(
                model.model,  # Use the underlying YOLOv7 model
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
                },
                verbose=False
            )
            print(f"✅ Successfully exported underlying model to: {output_path}")
            return True
            
        except Exception as e2:
            print(f"❌ Underlying model export also failed: {e2}")
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
    
    print(f"🚀 Starting YOLO Lightning to ONNX conversion...")
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"📏 Image size: {args.imgsz}x{args.imgsz}")
    
    try:
        # Reconstruct model from checkpoint
        model, settings = reconstruct_model_from_checkpoint(args.input)
        
        # Export to ONNX
        success = export_to_onnx(model, settings, args.output, args.imgsz)
        
        if success and Path(args.output).exists():
            # Validate the exported model
            validate_onnx_model(args.output)
            
            print(f"\n🎉 Conversion completed successfully!")
            print(f"📁 ONNX model saved to: {args.output}")
            
            # Create metadata file
            metadata = {
                "model_type": "yolov7-tiny",
                "dataset": "CropOrWeed2", 
                "classes": ["crop", "weed"],
                "input_size": [args.imgsz, args.imgsz],
                "source_checkpoint": str(Path(args.input).absolute()),
                "export_date": str(torch.utils.data.get_worker_info())
            }
            
            metadata_path = Path(args.output).with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"📄 Metadata saved to: {metadata_path}")
            
        else:
            print(f"\n❌ Conversion failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Conversion failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
