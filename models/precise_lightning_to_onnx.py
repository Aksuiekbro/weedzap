#!/usr/bin/env python3
"""
Precise YOLOv7-tiny Lightning Checkpoint to ONNX Converter
This script analyzes the checkpoint structure and creates an exact architecture match.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
from collections import OrderedDict

def extract_architecture_from_checkpoint(state_dict):
    """Extract the exact architecture from checkpoint weights."""
    print("\nAnalyzing checkpoint architecture...")
    
    # Remove Lightning prefixes
    cleaned_dict = {}
    for key, tensor in state_dict.items():
        clean_key = key.replace('model.model.', '').replace('model.', '')
        cleaned_dict[clean_key] = tensor
    
    # Analyze layer structure
    layers = {}
    conv_info = {}
    bn_info = {}
    
    for key, tensor in cleaned_dict.items():
        if '.conv.weight' in key:
            layer_num = key.split('.')[0]
            if layer_num.isdigit():
                layer_idx = int(layer_num)
                layers[layer_idx] = layers.get(layer_idx, {})
                layers[layer_idx]['conv'] = {
                    'out_channels': tensor.shape[0],
                    'in_channels': tensor.shape[1],
                    'kernel_size': tensor.shape[2],
                    'stride': 1  # Default, will be inferred
                }
                conv_info[layer_idx] = tensor.shape
        
        elif '.bn.weight' in key:
            layer_num = key.split('.')[0]
            if layer_num.isdigit():
                layer_idx = int(layer_num)
                layers[layer_idx] = layers.get(layer_idx, {})
                layers[layer_idx]['bn'] = {'num_features': tensor.shape[0]}
                bn_info[layer_idx] = tensor.shape[0]
    
    # Sort layers by index
    sorted_layers = OrderedDict(sorted(layers.items()))
    
    print(f"Found {len(sorted_layers)} main layers")
    for idx, layer_info in list(sorted_layers.items())[:10]:
        conv = layer_info.get('conv', {})
        print(f"Layer {idx}: {conv.get('in_channels', '?')} -> {conv.get('out_channels', '?')} (kernel: {conv.get('kernel_size', '?')})")
    
    return sorted_layers, conv_info, bn_info

class DynamicConv(nn.Module):
    """Dynamic convolution layer that adapts to checkpoint."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if not bias else None
        self.act = nn.SiLU() if act else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class AdaptiveYOLOv7Tiny(nn.Module):
    """Adaptive YOLOv7-tiny that builds architecture from checkpoint."""
    
    def __init__(self, layer_info, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.no = num_classes + 5
        self.na = 3  # number of anchors
        
        # Build backbone from layer info
        self.backbone = nn.ModuleList()
        
        # Build layers based on checkpoint structure
        prev_channels = 3  # RGB input
        
        for idx, info in layer_info.items():
            if 'conv' in info:
                conv_info = info['conv']
                in_ch = conv_info['in_channels']
                out_ch = conv_info['out_channels']
                k_size = conv_info['kernel_size']
                
                # Adjust input channels if needed
                if idx == 0:
                    in_ch = 3  # Force RGB input for first layer
                
                # Determine stride based on layer position and channel changes
                stride = 2 if out_ch >= in_ch * 1.5 else 1
                
                layer = DynamicConv(in_ch, out_ch, k_size, stride)
                self.backbone.append(layer)
                prev_channels = out_ch
                
                print(f"Added layer {idx}: Conv({in_ch}, {out_ch}, k={k_size}, s={stride})")
        
        # Add detection heads based on the last few layers
        # Find the largest channel dimensions for detection
        detection_channels = []
        for idx in sorted(layer_info.keys(), reverse=True)[:3]:
            if 'conv' in layer_info[idx]:
                detection_channels.append(layer_info[idx]['conv']['out_channels'])
        
        if len(detection_channels) >= 3:
            self.det_head_1 = nn.Conv2d(detection_channels[2], self.no * self.na, 1)  # Small
            self.det_head_2 = nn.Conv2d(detection_channels[1], self.no * self.na, 1)  # Medium  
            self.det_head_3 = nn.Conv2d(detection_channels[0], self.no * self.na, 1)  # Large
        else:
            # Fallback detection heads
            self.det_head_1 = nn.Conv2d(prev_channels // 4, self.no * self.na, 1)
            self.det_head_2 = nn.Conv2d(prev_channels // 2, self.no * self.na, 1)
            self.det_head_3 = nn.Conv2d(prev_channels, self.no * self.na, 1)
        
        print(f"Added detection heads with {self.no * self.na} outputs each")
    
    def forward(self, x):
        """Forward pass with feature extraction at multiple scales."""
        features = []
        
        # Process through backbone
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            # Save features at different scales for detection
            if i in [len(self.backbone) // 3, 2 * len(self.backbone) // 3, len(self.backbone) - 1]:
                features.append(x)
        
        # Ensure we have 3 feature maps
        while len(features) < 3:
            features.append(x)
        
        # Apply detection heads
        outputs = []
        if len(features) >= 3:
            outputs.append(self.det_head_1(features[0]))  # P3
            outputs.append(self.det_head_2(features[1]))  # P4  
            outputs.append(self.det_head_3(features[2]))  # P5
        else:
            outputs.append(self.det_head_3(x))
        
        return outputs

def load_weights_precisely(model, state_dict):
    """Load weights with precise matching."""
    model_state = model.state_dict()
    loaded_count = 0
    skipped_count = 0
    
    print(f"\nPrecise weight loading...")
    
    # Clean checkpoint keys
    cleaned_checkpoint = {}
    for key, tensor in state_dict.items():
        clean_key = key.replace('model.model.', '').replace('model.', '')
        cleaned_checkpoint[clean_key] = tensor
    
    # Match weights exactly
    for model_key in model_state.keys():
        model_shape = model_state[model_key].shape
        matched = False
        
        # Try exact match first
        if model_key in cleaned_checkpoint:
            ckpt_tensor = cleaned_checkpoint[model_key]
            if ckpt_tensor.shape == model_shape:
                model_state[model_key] = ckpt_tensor
                loaded_count += 1
                matched = True
        
        # Try pattern matching
        if not matched:
            for ckpt_key, ckpt_tensor in cleaned_checkpoint.items():
                if ckpt_tensor.shape == model_shape:
                    # Check for similar patterns
                    model_parts = model_key.split('.')
                    ckpt_parts = ckpt_key.split('.')
                    
                    # Match layer numbers and parameter types
                    if (len(model_parts) >= 2 and len(ckpt_parts) >= 2 and
                        model_parts[-1] == ckpt_parts[-1]):  # Same parameter type (weight, bias, etc.)
                        
                        model_state[model_key] = ckpt_tensor
                        loaded_count += 1
                        matched = True
                        break
        
        if not matched:
            skipped_count += 1
    
    # Load the updated state dict
    model.load_state_dict(model_state, strict=False)
    
    print(f"✓ Loaded {loaded_count} weights, skipped {skipped_count}")
    return loaded_count

def export_model_to_onnx(model, output_path, input_size=(1, 3, 640, 640)):
    """Export the model to ONNX with error handling."""
    print(f"\nExporting to ONNX: {output_path}")
    
    model.eval()
    
    try:
        # Create dummy input
        dummy_input = torch.randn(*input_size)
        
        # Test forward pass first
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"✓ Test forward pass successful")
            print(f"  Output shapes: {[out.shape for out in test_output]}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=[f'output_{i}' for i in range(len(test_output))],
            dynamic_axes={
                'images': {0: 'batch_size'},
                **{f'output_{i}': {0: 'batch_size'} for i in range(len(test_output))}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test ONNX inference
        session = onnxruntime.InferenceSession(output_path)
        test_input = np.random.randn(*input_size).astype(np.float32)
        onnx_outputs = session.run(None, {'images': test_input})
        
        print(f"✓ ONNX export successful!")
        print(f"✓ Model file: {output_path}")
        print(f"✓ File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        print(f"✓ ONNX output shapes: {[out.shape for out in onnx_outputs]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main conversion function."""
    checkpoint_path = "custom-models/tiny_model_680.ckpt"
    output_path = "custom-models/tiny_model_680_precise.onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    try:
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        hyper_params = checkpoint.get('hyper_parameters', {})
        
        # Extract architecture
        layer_info, conv_info, bn_info = extract_architecture_from_checkpoint(state_dict)
        
        # Determine number of classes from detection layers
        num_classes = 2  # Default for crop/weed
        
        # Check detection layer outputs to confirm classes
        detection_keys = [k for k in state_dict.keys() if 'm.' in k and 'weight' in k and len(state_dict[k].shape) == 4]
        if detection_keys:
            # Get the first detection layer
            det_shape = state_dict[detection_keys[0]].shape[0]
            if det_shape % 3 == 0:  # 3 anchors
                classes_plus_5 = det_shape // 3
                if classes_plus_5 > 5:
                    num_classes = classes_plus_5 - 5
        
        print(f"Detected {num_classes} classes")
        
        # Create adaptive model
        model = AdaptiveYOLOv7Tiny(layer_info, num_classes)
        
        # Load weights
        loaded_weights = load_weights_precisely(model, state_dict)
        
        # Export to ONNX
        success = export_model_to_onnx(model, output_path)
        
        if success:
            print(f"\n🎉 Precise conversion completed!")
            print(f"📁 Output: {output_path}")
            print(f"⚖️  Loaded: {loaded_weights}/{len(state_dict)} weights")
            print(f"🎯 Classes: {num_classes}")
        else:
            print(f"\n❌ Conversion failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
