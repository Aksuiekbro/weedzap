#!/usr/bin/env python3
"""
YOLO to ONNX Conversion Script

This script converts YOLO models (PyTorch .pt, YOLOv7 .ckpt) to ONNX format.
Based on the CropAndWeedDetection repository structure.

Usage:
    python convert_to_onnx.py --input tiny_model_680.ckpt --output yolo_crop_weed.onnx
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install torch ultralytics onnx onnxruntime numpy")
    sys.exit(1)


class YOLOToONNXConverter:
    """Convert YOLO models to ONNX format."""
    
    def __init__(self, input_path: str, output_path: str, imgsz: int = 640, simplify: bool = True):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.imgsz = imgsz
        self.simplify = simplify
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input model not found: {self.input_path}")
    
    def convert(self) -> bool:
        """Convert model to ONNX format."""
        try:
            print(f"🔄 Converting {self.input_path} to ONNX format...")
            
            # Create output directory
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine conversion strategy based on file extension
            if self.input_path.suffix == '.pt':
                return self._convert_pytorch_model()
            elif self.input_path.suffix == '.ckpt':
                return self._convert_checkpoint_model()
            else:
                print(f"❌ Unsupported format: {self.input_path.suffix}")
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def _convert_pytorch_model(self) -> bool:
        """Convert PyTorch .pt model to ONNX."""
        try:
            print(f"📦 Loading PyTorch model: {self.input_path}")
            
            # Load with ultralytics
            model = YOLO(str(self.input_path))
            
            # Export to ONNX
            print(f"🔄 Exporting to ONNX (image size: {self.imgsz}x{self.imgsz})...")
            exported_path = model.export(
                format='onnx',
                imgsz=self.imgsz,
                simplify=self.simplify,
                opset=11  # Use opset 11 for better compatibility
            )
            
            # Move to desired output location
            if Path(exported_path).resolve() != self.output_path.resolve():
                Path(exported_path).rename(self.output_path)
            
            print(f"✅ Successfully exported to: {self.output_path}")
            
            # Validate the ONNX model
            self._validate_onnx_model()
            
            return True
            
        except Exception as e:
            print(f"❌ PyTorch model conversion failed: {e}")
            return False
    
    def _convert_checkpoint_model(self) -> bool:
        """Convert checkpoint (.ckpt) model to ONNX."""
        try:
            print(f"🔍 Loading checkpoint: {self.input_path}")
            
            # Load checkpoint
            checkpoint = torch.load(str(self.input_path), map_location='cpu', weights_only=False)
            print(f"✅ Checkpoint loaded successfully")
            
            # Inspect checkpoint structure
            print(f"🔍 Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"🔑 Available keys: {list(checkpoint.keys())}")
            
            # Try multiple conversion strategies
            strategies = [
                self._strategy_lightning_checkpoint,
                self._strategy_ultralytics_compatible,
                self._strategy_direct_model_export,
                self._strategy_manual_onnx_export
            ]
            
            for i, strategy in enumerate(strategies, 1):
                print(f"\n🔄 Trying strategy {i}/{len(strategies)}: {strategy.__name__}")
                try:
                    if strategy(checkpoint):
                        print(f"✅ Strategy {i} succeeded!")
                        self._validate_onnx_model()
                        return True
                except Exception as e:
                    print(f"❌ Strategy {i} failed: {e}")
                    continue
            
            print(f"❌ All conversion strategies failed")
            return False
            
        except Exception as e:
            print(f"❌ Checkpoint loading failed: {e}")
            return False
    
    def _strategy_lightning_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Strategy 1: Handle PyTorch Lightning checkpoint."""
        if 'state_dict' not in checkpoint:
            raise ValueError("Not a Lightning checkpoint")
        
        print("🔍 Detected PyTorch Lightning checkpoint")
        
        # Extract model state dict
        state_dict = checkpoint['state_dict']
        
        # Filter model weights (remove 'model.' prefix if present)
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                clean_key = key[6:]  # Remove 'model.' prefix
                filtered_state_dict[clean_key] = value
            else:
                filtered_state_dict[key] = value
        
        # Create temporary .pt file
        temp_pt_path = self.output_path.with_suffix('.pt')
        
        # Get model info
        model_info = self._extract_model_info(checkpoint)
        
        # Create compatible checkpoint
        compatible_checkpoint = {
            'model': filtered_state_dict,
            'epoch': checkpoint.get('epoch', 0),
            'names': model_info.get('class_names', {i: f'class_{i}' for i in range(model_info.get('num_classes', 2))})
        }
        
        # Save temporary model
        torch.save(compatible_checkpoint, temp_pt_path)
        
        try:
            # Load with ultralytics and export
            model = YOLO(str(temp_pt_path))
            exported_path = model.export(
                format='onnx',
                imgsz=self.imgsz,
                simplify=self.simplify,
                opset=11
            )
            
            # Move to desired location
            if Path(exported_path).resolve() != self.output_path.resolve():
                Path(exported_path).rename(self.output_path)
            
            return True
            
        finally:
            # Clean up temporary file
            if temp_pt_path.exists():
                temp_pt_path.unlink()
    
    def _strategy_ultralytics_compatible(self, checkpoint: Dict[str, Any]) -> bool:
        """Strategy 2: Try to make checkpoint ultralytics-compatible."""
        if 'model' not in checkpoint and 'ema' not in checkpoint:
            raise ValueError("No model data found")
        
        print("🔍 Attempting ultralytics compatibility conversion")
        
        # Get model state dict
        model_state = checkpoint.get('model') or checkpoint.get('ema')
        if hasattr(model_state, 'state_dict'):
            model_state = model_state.state_dict()
        
        # Create temporary .pt file
        temp_pt_path = self.output_path.with_suffix('.pt')
        
        # Create ultralytics-compatible structure
        model_info = self._extract_model_info(checkpoint)
        ultralytics_checkpoint = {
            'model': model_state,
            'epoch': checkpoint.get('epoch', 0),
            'date': checkpoint.get('date', None),
            'names': model_info.get('class_names', {0: 'crop', 1: 'weed'}),
            'nc': model_info.get('num_classes', 2)
        }
        
        # Save and convert
        torch.save(ultralytics_checkpoint, temp_pt_path)
        
        try:
            model = YOLO(str(temp_pt_path))
            exported_path = model.export(
                format='onnx',
                imgsz=self.imgsz,
                simplify=self.simplify,
                opset=11
            )
            
            if Path(exported_path).resolve() != self.output_path.resolve():
                Path(exported_path).rename(self.output_path)
            
            return True
            
        finally:
            if temp_pt_path.exists():
                temp_pt_path.unlink()
    
    def _strategy_direct_model_export(self, checkpoint: Dict[str, Any]) -> bool:
        """Strategy 3: Direct model export if model object is available."""
        model_obj = None
        
        # Look for model object
        for key in ['model', 'ema']:
            if key in checkpoint:
                model_obj = checkpoint[key]
                break
        
        if model_obj is None or not hasattr(model_obj, 'eval'):
            raise ValueError("No model object found for direct export")
        
        print("🔍 Attempting direct model export")
        
        # Set model to evaluation mode
        model_obj.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.imgsz, self.imgsz)
        
        # Export to ONNX
        torch.onnx.export(
            model_obj,
            dummy_input,
            str(self.output_path),
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
        
        return True
    
    def _strategy_manual_onnx_export(self, checkpoint: Dict[str, Any]) -> bool:
        """Strategy 4: Manual ONNX export for YOLOv7 models."""
        print("🔍 Attempting manual ONNX export")
        
        # This strategy provides instructions for manual conversion
        print("\n📋 Manual conversion required:")
        print("This appears to be a YOLOv7 checkpoint that requires the original YOLOv7 export script.")
        print("\nRecommended steps:")
        print("1. Clone YOLOv7 repository: git clone https://github.com/WongKinYiu/yolov7.git")
        print("2. Install requirements: pip install -r yolov7/requirements.txt")
        print(f"3. Export ONNX: python yolov7/export.py --weights {self.input_path} --grid --end2end --simplify --img-size {self.imgsz}")
        print(f"4. The output will be: {self.input_path.with_suffix('.onnx')}")
        
        # Create a placeholder file with instructions
        instructions_path = self.output_path.with_suffix('.txt')
        with open(instructions_path, 'w') as f:
            f.write(f"Manual ONNX conversion required for: {self.input_path}\n")
            f.write(f"Use YOLOv7 export script with these parameters:\n")
            f.write(f"python yolov7/export.py --weights {self.input_path} --grid --end2end --simplify --img-size {self.imgsz}\n")
        
        print(f"📝 Instructions saved to: {instructions_path}")
        
        return False  # This strategy doesn't actually convert
    
    def _extract_model_info(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from checkpoint."""
        info = {
            'num_classes': 2,  # Default for crop/weed detection
            'class_names': {0: 'crop', 1: 'weed'},
            'input_size': self.imgsz,
            'model_type': 'yolo'
        }
        
        # Try to extract from hyperparameters
        if 'hyper_parameters' in checkpoint:
            hyp = checkpoint['hyper_parameters']
            if 'nc' in hyp:
                info['num_classes'] = hyp['nc']
            if 'names' in hyp:
                info['class_names'] = hyp['names']
        
        # Try to extract from names field
        if 'names' in checkpoint:
            names = checkpoint['names']
            if isinstance(names, dict):
                info['class_names'] = names
                info['num_classes'] = len(names)
            elif isinstance(names, list):
                info['class_names'] = {i: name for i, name in enumerate(names)}
                info['num_classes'] = len(names)
        
        # Determine model variant from filename
        filename = self.input_path.name.lower()
        if 'tiny' in filename:
            info['variant'] = 'tiny'
        elif 'nano' in filename or 'n' in filename:
            info['variant'] = 'nano'
        elif 'small' in filename or 's' in filename:
            info['variant'] = 'small'
        elif 'medium' in filename or 'm' in filename:
            info['variant'] = 'medium'
        elif 'large' in filename or 'l' in filename:
            info['variant'] = 'large'
        elif 'extra' in filename or 'x' in filename:
            info['variant'] = 'xlarge'
        
        return info
    
    def _validate_onnx_model(self) -> bool:
        """Validate the exported ONNX model."""
        try:
            print(f"🔍 Validating ONNX model: {self.output_path}")
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(self.output_path))
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(str(self.output_path))
            
            # Get input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()
            
            print(f"✅ ONNX model validation successful!")
            print(f"📊 Input shape: {input_info.shape}")
            print(f"📊 Input type: {input_info.type}")
            print(f"📊 Output(s): {len(output_info)} tensor(s)")
            for i, output in enumerate(output_info):
                print(f"   Output {i}: {output.name} - {output.shape}")
            
            # Test inference with dummy data
            dummy_input = np.random.randn(*input_info.shape).astype(np.float32)
            outputs = ort_session.run(None, {input_info.name: dummy_input})
            print(f"✅ Test inference successful! Output shapes: {[out.shape for out in outputs]}")
            
            # Print model info
            file_size = self.output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"📁 Model size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"⚠️  ONNX validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO models to ONNX format")
    parser.add_argument("--input", "-i", required=True, help="Input model path (.pt or .ckpt)")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX model path")
    parser.add_argument("--imgsz", "--img-size", type=int, default=640, help="Image size for inference (default: 640)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model (default: True)")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify", help="Disable ONNX simplification")
    
    args = parser.parse_args()
    
    # Create converter
    converter = YOLOToONNXConverter(
        input_path=args.input,
        output_path=args.output,
        imgsz=args.imgsz,
        simplify=args.simplify
    )
    
    # Convert model
    success = converter.convert()
    
    if success:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 ONNX model saved to: {args.output}")
    else:
        print(f"\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
