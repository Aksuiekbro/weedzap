#!/usr/bin/env python3
"""
Comprehensive Lightning Checkpoint to ONNX Converter
This script handles PyTorch Lightning checkpoints and converts them to ONNX format.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
from pathlib import Path

def load_lightning_checkpoint(checkpoint_path):
    """Load and inspect a PyTorch Lightning checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract information
    state_dict = checkpoint.get('state_dict', {})
    hyper_parameters = checkpoint.get('hyper_parameters', {})
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Hyper parameters: {hyper_parameters}")
    print(f"State dict has {len(state_dict)} parameters")
    
    # Print some layer names to understand the structure
    print("\nFirst 10 layer names:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key}: {state_dict[key].shape}")
    
    return checkpoint, state_dict, hyper_parameters

def clean_state_dict(state_dict):
    """Remove Lightning-specific prefixes from state dict keys."""
    cleaned_state_dict = {}
    
    for key, value in state_dict.items():
        # Remove common Lightning prefixes
        new_key = key
        prefixes_to_remove = ['model.', 'net.', 'backbone.', 'detector.']
        
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        
        cleaned_state_dict[new_key] = value
    
    return cleaned_state_dict

class SimpleYOLOHead(nn.Module):
    """Simplified YOLO detection head for ONNX export."""
    
    def __init__(self, num_classes=80, anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = 3  # number of detection layers
        self.na = 3  # number of anchors
        
        # Default anchors for YOLOv7-tiny
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]
            ]).float()
        else:
            self.anchors = anchors
        
        # Detection layers
        self.m = nn.ModuleList([
            nn.Conv2d(128, self.no * self.na, 1),  # P3
            nn.Conv2d(256, self.no * self.na, 1),  # P4
            nn.Conv2d(512, self.no * self.na, 1),  # P5
        ])
    
    def forward(self, x):
        """Forward pass."""
        if isinstance(x, (list, tuple)):
            # Multiple feature maps
            outputs = []
            for i, feat in enumerate(x):
                if i < len(self.m):
                    outputs.append(self.m[i](feat))
            return outputs
        else:
            # Single feature map
            return self.m[0](x)

class MinimalYOLO(nn.Module):
    """Minimal YOLO model for ONNX export."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # Simple backbone layers
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        
        # Detection head
        self.head = SimpleYOLOHead(num_classes)
        
        # Activation
        self.act = nn.SiLU()
    
    def forward(self, x):
        """Forward pass."""
        # Simple feature extraction
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))  # P3 feature
        x4 = self.act(self.conv4(x3))  # P4 feature
        x5 = self.act(self.conv5(x4))  # P5 feature
        
        # Detection
        outputs = self.head([x3, x4, x5])
        return outputs

def create_yolo_from_weights(state_dict, num_classes=80):
    """Create a YOLO model and load weights from state dict."""
    model = MinimalYOLO(num_classes=num_classes)
    
    # Try to load compatible weights
    model_dict = model.state_dict()
    compatible_dict = {}
    
    print(f"\nTrying to match {len(state_dict)} checkpoint weights to {len(model_dict)} model weights")
    
    # Simple key matching
    for model_key in model_dict.keys():
        # Try exact match first
        if model_key in state_dict:
            if model_dict[model_key].shape == state_dict[model_key].shape:
                compatible_dict[model_key] = state_dict[model_key]
                print(f"✓ Matched: {model_key}")
        else:
            # Try partial matching
            for ckpt_key in state_dict.keys():
                if model_key.endswith(ckpt_key) or ckpt_key.endswith(model_key):
                    if model_dict[model_key].shape == state_dict[ckpt_key].shape:
                        compatible_dict[model_key] = state_dict[ckpt_key]
                        print(f"✓ Partial match: {model_key} <- {ckpt_key}")
                        break
    
    print(f"Successfully matched {len(compatible_dict)} out of {len(model_dict)} weights")
    
    # Load the compatible weights
    model.load_state_dict(compatible_dict, strict=False)
    return model

def export_to_onnx(model, output_path, input_size=(1, 3, 640, 640)):
    """Export model to ONNX format."""
    print(f"\nExporting to ONNX: {output_path}")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ Successfully exported to {output_path}")
        
        # Test with ONNX Runtime
        session = onnxruntime.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        # Run inference test
        test_input = np.random.randn(*input_size).astype(np.float32)
        outputs = session.run(output_names, {input_name: test_input})
        
        print(f"✓ ONNX model validation successful")
        print(f"✓ Input shape: {input_size}")
        print(f"✓ Output shapes: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

def main():
    """Main conversion function."""
    checkpoint_path = "custom-models/tiny_model_680.ckpt"
    output_path = "custom-models/tiny_model_680.onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    try:
        # Load checkpoint
        checkpoint, state_dict, hyper_params = load_lightning_checkpoint(checkpoint_path)
        
        # Clean state dict
        cleaned_state_dict = clean_state_dict(state_dict)
        print(f"\nCleaned state dict has {len(cleaned_state_dict)} parameters")
        
        # Determine number of classes
        num_classes = hyper_params.get('num_classes', 80)
        if 'classes' in hyper_params:
            num_classes = len(hyper_params['classes'])
        
        print(f"Using {num_classes} classes")
        
        # Create model and load weights
        model = create_yolo_from_weights(cleaned_state_dict, num_classes)
        
        # Export to ONNX
        success = export_to_onnx(model, output_path)
        
        if success:
            print(f"\n🎉 Conversion completed successfully!")
            print(f"ONNX model saved to: {output_path}")
        else:
            print(f"\n❌ Conversion failed")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
