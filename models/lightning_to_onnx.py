#!/usr/bin/env python3
"""
Enhanced YOLO Lightning Checkpoint to ONNX Converter

This script specifically handles PyTorch Lightning checkpoints from the 
CropAndWeedDetection project and converts them to ONNX format.
"""

import argparse
import json
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
import os

# Add the project path to import YOLOv7 models
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not available, using alternative approach")


def extract_model_from_lightning_checkpoint(checkpoint_path: str, output_pt_path: str):
    """Extract and save model from PyTorch Lightning checkpoint."""
    print(f"🔍 Loading Lightning checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters
    hyp = checkpoint.get('hyper_parameters', {})
    settings = hyp.get('settings', {})
    
    print(f"📊 Model info:")
    print(f"   - Model: {settings.get('model', {}).get('name', 'unknown')}")
    print(f"   - Dataset: {settings.get('dataset', {}).get('name', 'unknown')}")
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
    
    # Determine number of classes from the model
    num_classes = 2  # Default for crop/weed
    
    # Try to find output layer to determine number of classes
    for key in clean_state_dict.keys():
        if 'head' in key and 'weight' in key and len(clean_state_dict[key].shape) >= 2:
            # YOLO detection head typically has shape [num_outputs, input_dim]
            # num_outputs = (num_classes + 5) * num_anchors for YOLOv7
            num_outputs = clean_state_dict[key].shape[0]
            print(f"🔍 Found head layer '{key}' with {num_outputs} outputs")
            break
    
    # Create class names
    dataset_name = settings.get('dataset', {}).get('name', 'CropOrWeed2')
    if 'crop' in dataset_name.lower() or 'weed' in dataset_name.lower():
        class_names = {0: 'crop', 1: 'weed'}
    else:
        class_names = {i: f'class_{i}' for i in range(num_classes)}
    
    print(f"📋 Classes: {class_names}")
    
    # Create a proper PyTorch model checkpoint
    model_checkpoint = {
        'model': clean_state_dict,
        'epoch': checkpoint.get('epoch', 0),
        'names': class_names,
        'nc': num_classes,
        'imgsz': settings.get('dataset', {}).get('image_size', [640, 640])[0],
        'model_name': settings.get('model', {}).get('name', 'yolov7-tiny')
    }
    
    # Save the cleaned checkpoint
    torch.save(model_checkpoint, output_pt_path)
    print(f"✅ Saved cleaned model to: {output_pt_path}")
    
    return model_checkpoint


def convert_to_onnx_alternative(checkpoint_path: str, output_onnx_path: str, img_size: int = 640):
    """Alternative ONNX conversion method for YOLOv7 models."""
    print(f"🔄 Converting {checkpoint_path} to ONNX using alternative method...")
    
    # Create temporary PT file
    temp_pt_path = Path(output_onnx_path).with_suffix('.pt')
    
    try:
        # Extract model from Lightning checkpoint
        model_info = extract_model_from_lightning_checkpoint(checkpoint_path, str(temp_pt_path))
        
        # Try ultralytics first
        if YOLO is not None:
            try:
                print("🔄 Attempting conversion with ultralytics...")
                model = YOLO(str(temp_pt_path))
                exported_path = model.export(
                    format='onnx',
                    imgsz=img_size,
                    simplify=True,
                    opset=11
                )
                
                # Move to desired location if needed
                if Path(exported_path).resolve() != Path(output_onnx_path).resolve():
                    Path(exported_path).rename(output_onnx_path)
                
                print(f"✅ Successfully converted using ultralytics!")
                return True
                
            except Exception as e:
                print(f"❌ Ultralytics conversion failed: {e}")
        
        # Try manual ONNX export
        print("🔄 Attempting manual ONNX export...")
        return manual_onnx_export(str(temp_pt_path), output_onnx_path, img_size, model_info)
        
    finally:
        # Clean up temporary file
        if temp_pt_path.exists():
            temp_pt_path.unlink()


def manual_onnx_export(model_path: str, output_path: str, img_size: int, model_info: dict):
    """Manual ONNX export using torch.onnx.export."""
    try:
        print("🔧 Performing manual ONNX export...")
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create a simple wrapper model that loads the state dict
        class YOLOWrapper(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                self.state_dict_data = state_dict
                
            def forward(self, x):
                # This is a placeholder - the actual model structure would need to be reconstructed
                # For now, we'll create a minimal model that can be exported
                return x  # Placeholder
        
        # Create wrapper model
        model = YOLOWrapper(checkpoint['model'])
        model.eval()
        
        # This approach requires the actual YOLOv7 model definition
        # which is not available in this context
        print("❌ Manual export requires YOLOv7 model definitions")
        print("📋 Please use the original YOLOv7 export script:")
        print(f"   git clone https://github.com/WongKinYiu/yolov7.git")
        print(f"   python yolov7/export.py --weights {model_path} --grid --end2end --simplify --img-size {img_size}")
        
        return False
        
    except Exception as e:
        print(f"❌ Manual export failed: {e}")
        return False


def export_with_yolov7_repo(checkpoint_path: str, output_path: str, img_size: int = 640):
    """Download YOLOv7 repo and use official export script."""
    import subprocess
    import shutil
    
    yolov7_path = Path("yolov7")
    
    try:
        # Clone YOLOv7 if not exists
        if not yolov7_path.exists():
            print("📥 Cloning YOLOv7 repository...")
            subprocess.run([
                "git", "clone", "https://github.com/WongKinYiu/yolov7.git"
            ], check=True)
        
        # Create temporary PT file
        temp_pt_path = Path("temp_model.pt")
        model_info = extract_model_from_lightning_checkpoint(checkpoint_path, str(temp_pt_path))
        
        # Use YOLOv7 export script
        print("🔄 Using YOLOv7 export script...")
        export_cmd = [
            "python3", str(yolov7_path / "export.py"),
            "--weights", str(temp_pt_path),
            "--grid", "--end2end", "--simplify",
            "--img-size", str(img_size), str(img_size)
        ]
        
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Move the exported ONNX file
            exported_onnx = temp_pt_path.with_suffix('.onnx')
            if exported_onnx.exists():
                shutil.move(str(exported_onnx), output_path)
                print(f"✅ Successfully exported to: {output_path}")
                return True
        else:
            print(f"❌ YOLOv7 export failed: {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Clean up
        if temp_pt_path.exists():
            temp_pt_path.unlink()
    
    return False


def validate_onnx_model(onnx_path: str):
    """Validate the exported ONNX model."""
    try:
        print(f"🔍 Validating ONNX model: {onnx_path}")
        
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
        if input_info.shape[1] == 3:  # RGB input
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
    parser = argparse.ArgumentParser(description="Convert PyTorch Lightning YOLO checkpoints to ONNX")
    parser.add_argument("--input", "-i", required=True, help="Input checkpoint path (.ckpt)")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX model path")
    parser.add_argument("--imgsz", "--img-size", type=int, default=640, help="Image size for inference")
    parser.add_argument("--method", choices=["auto", "yolov7"], default="auto", 
                       help="Conversion method: auto (try multiple), yolov7 (use official repo)")
    
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
    
    success = False
    
    if args.method == "yolov7":
        success = export_with_yolov7_repo(args.input, args.output, args.imgsz)
    else:
        # Try alternative method first
        success = convert_to_onnx_alternative(args.input, args.output, args.imgsz)
        
        if not success:
            print("\n🔄 Trying YOLOv7 repository method...")
            success = export_with_yolov7_repo(args.input, args.output, args.imgsz)
    
    if success and Path(args.output).exists():
        validate_onnx_model(args.output)
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 ONNX model saved to: {args.output}")
    else:
        print(f"\n❌ Conversion failed!")
        print("\n📋 Manual conversion options:")
        print("1. Use original YOLOv7 repository:")
        print("   git clone https://github.com/WongKinYiu/yolov7.git")
        print("   pip install -r yolov7/requirements.txt")
        print(f"   python yolov7/export.py --weights {args.input} --grid --end2end --simplify --img-size {args.imgsz}")
        print("\n2. Or convert to PyTorch .pt first, then use ultralytics YOLO export")
        sys.exit(1)


if __name__ == "__main__":
    main()
