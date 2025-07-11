#!/usr/bin/env python3
"""
YOLOv7 Checkpoint Inspector

This script analyzes the structure and contents of YOLOv7 checkpoint files
to understand their format for conversion purposes.
"""

import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any, List

def analyze_tensor_info(tensor):
    """Analyze tensor properties."""
    if isinstance(tensor, torch.Tensor):
        return {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
            'size_mb': tensor.numel() * tensor.element_size() / (1024 * 1024)
        }
    return None

def explore_dict_structure(obj, max_depth=3, current_depth=0, path=""):
    """Recursively explore dictionary structure."""
    structure = {}
    
    if current_depth >= max_depth:
        return {"...": "max_depth_reached"}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, torch.Tensor):
                structure[key] = {
                    'type': 'Tensor',
                    'info': analyze_tensor_info(value)
                }
            elif isinstance(value, dict):
                structure[key] = {
                    'type': 'Dict',
                    'keys': list(value.keys())[:10],  # First 10 keys
                    'total_keys': len(value),
                    'structure': explore_dict_structure(value, max_depth, current_depth + 1, current_path)
                }
            elif isinstance(value, (list, tuple)):
                structure[key] = {
                    'type': type(value).__name__,
                    'length': len(value),
                    'sample': str(value[:3]) if len(value) <= 10 else f"{str(value[:3])}... ({len(value)} items)"
                }
            else:
                structure[key] = {
                    'type': type(value).__name__,
                    'value': str(value)[:100]  # First 100 chars
                }
    
    return structure

def inspect_checkpoint(checkpoint_path: str):
    """Inspect YOLOv7 checkpoint structure."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"🔍 Inspecting checkpoint: {checkpoint_path.name}")
    print(f"📁 File size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
    print("=" * 60)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None
    
    # Basic info
    print(f"\n📊 Checkpoint Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"🔑 Top-level keys: {list(checkpoint.keys())}")
        print(f"📝 Total keys: {len(checkpoint)}")
        
        # Analyze each top-level key
        print("\n🔍 Detailed Structure Analysis:")
        print("-" * 40)
        
        structure = explore_dict_structure(checkpoint, max_depth=3)
        
        for key, info in structure.items():
            print(f"\n🔹 {key}:")
            if info['type'] == 'Tensor':
                tensor_info = info['info']
                print(f"   📐 Shape: {tensor_info['shape']}")
                print(f"   🏷️  Type: {tensor_info['dtype']}")
                print(f"   💾 Size: {tensor_info['size_mb']:.2f} MB")
            elif info['type'] == 'Dict':
                print(f"   📂 Keys ({info['total_keys']}): {info['keys']}")
                if 'structure' in info and info['structure']:
                    for subkey, subinfo in info['structure'].items():
                        if isinstance(subinfo, dict) and 'type' in subinfo:
                            print(f"      └─ {subkey}: {subinfo['type']}")
            else:
                print(f"   📄 Type: {info['type']}")
                if 'value' in info:
                    print(f"   📝 Value: {info['value']}")
        
        # Look for model-specific information
        print("\n🤖 Model Information:")
        print("-" * 40)
        
        # Check for common YOLOv7 keys
        model_keys = ['model', 'ema', 'state_dict', 'net']
        found_model_keys = []
        
        for key in model_keys:
            if key in checkpoint:
                found_model_keys.append(key)
                print(f"✅ Found model key: '{key}'")
                
                model_obj = checkpoint[key]
                if hasattr(model_obj, 'state_dict'):
                    print(f"   📋 Has state_dict method")
                elif isinstance(model_obj, dict):
                    print(f"   📂 Dict with {len(model_obj)} keys")
                    if len(model_obj) <= 10:
                        print(f"   🔑 Keys: {list(model_obj.keys())}")
                else:
                    print(f"   📄 Type: {type(model_obj)}")
        
        if not found_model_keys:
            print("⚠️  No standard model keys found")
            print("🔍 Searching for potential model data...")
            
            # Look for keys that might contain model weights
            potential_keys = []
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 1:
                    potential_keys.append(key)
                elif isinstance(value, dict) and any('weight' in k or 'bias' in k for k in str(value.keys()).lower()):
                    potential_keys.append(key)
            
            if potential_keys:
                print(f"🎯 Potential model keys: {potential_keys}")
        
        # Check for metadata
        print("\n📋 Training Metadata:")
        print("-" * 40)
        
        metadata_keys = ['epoch', 'best_fitness', 'model', 'optimizer', 'lr_scheduler', 'date', 'names']
        for key in metadata_keys:
            if key in checkpoint:
                value = checkpoint[key]
                if isinstance(value, (int, float, str)):
                    print(f"✅ {key}: {value}")
                elif isinstance(value, dict) and key == 'names':
                    print(f"✅ {key}: {value}")
                else:
                    print(f"✅ {key}: {type(value)} ({len(value) if hasattr(value, '__len__') else 'N/A'})")
        
        # Check for class names
        print("\n🏷️  Class Information:")
        print("-" * 40)
        
        class_keys = ['names', 'class_names', 'classes']
        found_classes = False
        
        for key in class_keys:
            if key in checkpoint:
                classes = checkpoint[key]
                print(f"✅ Found classes in '{key}': {classes}")
                found_classes = True
                break
        
        if not found_classes:
            print("⚠️  No class information found in top-level keys")
            
            # Search deeper
            for key, value in checkpoint.items():
                if isinstance(value, dict) and 'names' in value:
                    print(f"🔍 Found classes in '{key}.names': {value['names']}")
                    found_classes = True
                    break
        
        # Generate conversion suggestions
        print("\n💡 Conversion Suggestions:")
        print("-" * 40)
        
        if 'model' in checkpoint:
            print("✅ Standard YOLOv7 format - should work with ultralytics YOLO")
        elif 'ema' in checkpoint:
            print("🔄 EMA model detected - try using 'ema' key instead of 'model'")
        elif 'state_dict' in checkpoint:
            print("🔄 State dict format - may need manual model reconstruction")
        else:
            print("⚠️  Non-standard format - manual conversion required")
            print("💭 Consider using original YOLOv7 export script")
        
        return checkpoint
    
    else:
        print(f"⚠️  Unexpected checkpoint format: {type(checkpoint)}")
        return checkpoint

def main():
    parser = argparse.ArgumentParser(description='Inspect YOLOv7 checkpoint structure')
    parser.add_argument('checkpoint', help='Path to .ckpt file')
    parser.add_argument('--save-structure', '-s', help='Save structure analysis to JSON file')
    
    args = parser.parse_args()
    
    try:
        checkpoint = inspect_checkpoint(args.checkpoint)
        
        if args.save_structure and isinstance(checkpoint, dict):
            # Save structure to JSON
            structure = explore_dict_structure(checkpoint, max_depth=5)
            
            # Convert tensors to serializable format
            def make_serializable(obj):
                if isinstance(obj, dict):
                    result = {}
                    for k, v in obj.items():
                        result[k] = make_serializable(v)
                    return result
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, 'tolist'):  # torch tensors
                    return {'tensor_shape': list(obj.shape) if hasattr(obj, 'shape') else 'unknown'}
                else:
                    return str(obj)
            
            serializable_structure = make_serializable(structure)
            
            with open(args.save_structure, 'w') as f:
                json.dump(serializable_structure, f, indent=2)
            
            print(f"\n💾 Structure saved to: {args.save_structure}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()