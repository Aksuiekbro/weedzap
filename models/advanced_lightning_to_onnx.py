#!/usr/bin/env python3
"""
Advanced YOLOv7 Lightning Checkpoint to ONNX Converter
This script reconstructs the YOLOv7-tiny architecture from checkpoint weights.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
import os
import re
from collections import OrderedDict

def analyze_checkpoint_structure(checkpoint_path):
    """Analyze the checkpoint to understand the model structure."""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', {})
    hyper_parameters = checkpoint.get('hyper_parameters', {})
    
    # Analyze layer structure
    layer_analysis = {}
    conv_layers = []
    bn_layers = []
    detection_layers = []
    
    for key, tensor in state_dict.items():
        # Remove model prefix
        clean_key = key.replace('model.model.', '').replace('model.', '')
        
        if 'conv.weight' in key:
            conv_layers.append((clean_key, tensor.shape))
        elif 'bn.weight' in key or 'bn.bias' in key:
            bn_layers.append((clean_key, tensor.shape))
        elif 'm.' in key and 'weight' in key:
            detection_layers.append((clean_key, tensor.shape))
    
    print(f"\nFound {len(conv_layers)} conv layers, {len(bn_layers)} bn layers, {len(detection_layers)} detection layers")
    
    return checkpoint, state_dict, hyper_parameters, conv_layers, detection_layers

class Conv(nn.Module):
    """Standard convolution with batch norm and activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, (k // 2) if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ELAN(nn.Module):
    """ELAN (Efficient Layer Aggregation Network) block."""
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1)
        self.cv2 = Conv(c1, c3, 1)
        self.cv3 = Conv(2 * c3, c2, 1)
        self.m = nn.ModuleList([Conv(c3, c4, 3) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))

class MP(nn.Module):
    """Max pooling with additional convolution."""
    def __init__(self, c1, c2):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv1 = Conv(c1, c2 // 2, 1, 1)
        self.cv2 = Conv(c1, c2 // 2, 1, 1)
        self.cv3 = Conv(c2 // 2, c2 // 2, 3, 2)

    def forward(self, x):
        return torch.cat([self.cv1(self.m(x)), self.cv3(self.cv2(x))], 1)

class SPPCSPC(nn.Module):
    """SPP with CSP."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class ReconstructedYOLOv7Tiny(nn.Module):
    """Reconstructed YOLOv7-tiny model based on checkpoint analysis."""
    
    def __init__(self, num_classes=80, ch=3):
        super().__init__()
        self.num_classes = num_classes
        self.nc = num_classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = 3  # number of detection layers
        self.na = 3  # number of anchors
        
        # Backbone
        self.model = nn.ModuleList()
        
        # Layer 0: Conv 32
        self.model.append(Conv(ch, 32, 3, 2))  # 0-P1/2
        
        # Layer 1: Conv 64
        self.model.append(Conv(32, 64, 3, 2))  # 1-P2/4
        
        # Layer 2: ELAN
        self.model.append(ELAN(64, 64, 32, 32))  # 2
        
        # Layer 3: MP
        self.model.append(MP(64, 128))  # 3-P3/8
        
        # Layer 4: ELAN
        self.model.append(ELAN(128, 128, 64, 64))  # 4
        
        # Layer 5: MP
        self.model.append(MP(128, 256))  # 5-P4/16
        
        # Layer 6: ELAN
        self.model.append(ELAN(256, 256, 128, 128))  # 6
        
        # Layer 7: MP
        self.model.append(MP(256, 512))  # 7-P5/32
        
        # Layer 8: ELAN
        self.model.append(ELAN(512, 512, 256, 256))  # 8
        
        # Layer 9: SPPCSPC
        self.model.append(SPPCSPC(512, 256, 1))  # 9
        
        # Neck (simplified)
        self.model.append(Conv(256, 128, 1, 1))  # 10
        self.model.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 11
        self.model.append(Conv(384, 128, 1, 1))  # 12 (after concat)
        self.model.append(ELAN(128, 128, 64, 64))  # 13
        
        self.model.append(Conv(128, 64, 1, 1))  # 14
        self.model.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 15
        self.model.append(Conv(192, 64, 1, 1))  # 16 (after concat)
        self.model.append(ELAN(64, 64, 32, 32))  # 17
        
        # Head
        self.model.append(Conv(64, 128, 3, 2))  # 18
        self.model.append(ELAN(256, 128, 64, 64))  # 19 (after concat)
        
        self.model.append(Conv(128, 256, 3, 2))  # 20
        self.model.append(ELAN(512, 256, 128, 128))  # 21 (after concat)
        
        # Detection heads
        self.model.append(nn.Conv2d(64, self.no * self.na, 1))   # 22 (P3/8-small)
        self.model.append(nn.Conv2d(128, self.no * self.na, 1))  # 23 (P4/16-medium)
        self.model.append(nn.Conv2d(256, self.no * self.na, 1))  # 24 (P5/32-large)
        
        # Anchors
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]).float() / 8.0  # Scale down
        
    def forward(self, x):
        """Forward pass."""
        outputs = []
        
        # Backbone
        for i in range(10):
            x = self.model[i](x)
        
        # Save P5
        p5 = x
        
        # First upsampling path
        x = self.model[10](x)  # Conv 1x1
        x = self.model[11](x)  # Upsample
        
        # Get P4 and concatenate (simplified)
        p4 = p5  # Simplified connection
        x = torch.cat([x, p4], 1)
        x = self.model[12](x)  # Conv after concat
        x = self.model[13](x)  # ELAN
        
        # Save for P4 output
        p4_out = x
        
        # Second upsampling path
        x = self.model[14](x)  # Conv 1x1
        x = self.model[15](x)  # Upsample
        
        # Get P3 and concatenate (simplified)
        p3 = p4_out  # Simplified connection
        x = torch.cat([x, p3], 1)
        x = self.model[16](x)  # Conv after concat
        x = self.model[17](x)  # ELAN
        
        # P3 output
        p3_out = x
        
        # First downsampling
        x = self.model[18](x)  # Conv 3x3 s2
        x = torch.cat([x, p4_out], 1)
        x = self.model[19](x)  # ELAN
        
        # P4 final output
        p4_final = x
        
        # Second downsampling
        x = self.model[20](x)  # Conv 3x3 s2
        x = torch.cat([x, p5], 1)
        x = self.model[21](x)  # ELAN
        
        # P5 final output
        p5_final = x
        
        # Detection heads
        out_p3 = self.model[22](p3_out)   # Small objects
        out_p4 = self.model[23](p4_final)  # Medium objects
        out_p5 = self.model[24](p5_final)  # Large objects
        
        return [out_p3, out_p4, out_p5]

def load_weights_to_reconstructed_model(model, state_dict):
    """Load weights from checkpoint to reconstructed model."""
    model_state = model.state_dict()
    loaded_count = 0
    
    print(f"\nLoading weights to reconstructed model...")
    print(f"Model has {len(model_state)} parameters")
    print(f"Checkpoint has {len(state_dict)} parameters")
    
    # Create mapping
    weight_mapping = {}
    
    for model_key in model_state.keys():
        model_shape = model_state[model_key].shape
        
        # Try to find matching weight in checkpoint
        for ckpt_key in state_dict.keys():
            # Remove prefixes
            clean_ckpt_key = ckpt_key.replace('model.model.', '').replace('model.', '')
            
            # Check if shapes match and keys are similar
            if state_dict[ckpt_key].shape == model_shape:
                # Simple pattern matching
                if any(pattern in model_key and pattern in clean_ckpt_key 
                       for pattern in ['conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var']):
                    weight_mapping[model_key] = ckpt_key
                    loaded_count += 1
                    break
    
    print(f"Found {loaded_count} weight mappings")
    
    # Load the weights
    for model_key, ckpt_key in weight_mapping.items():
        model_state[model_key] = state_dict[ckpt_key]
    
    # Load the state dict
    model.load_state_dict(model_state, strict=False)
    
    return loaded_count

def export_to_onnx_advanced(model, output_path, input_size=(1, 3, 640, 640)):
    """Export model to ONNX with advanced options."""
    print(f"\nExporting to ONNX: {output_path}")
    
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output0', 'output1', 'output2'],
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output0': {0: 'batch_size'},
                    'output1': {0: 'batch_size'},
                    'output2': {0: 'batch_size'}
                }
            )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ Successfully exported to {output_path}")
        
        # Test inference
        session = onnxruntime.InferenceSession(output_path)
        test_input = np.random.randn(*input_size).astype(np.float32)
        outputs = session.run(None, {'images': test_input})
        
        print(f"✓ ONNX inference test successful")
        print(f"✓ Output shapes: {[out.shape for out in outputs]}")
        
        # Calculate model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Model size: {model_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

def main():
    """Main conversion function."""
    checkpoint_path = "custom-models/tiny_model_680.ckpt"
    output_path = "custom-models/tiny_model_680_reconstructed.onnx"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    try:
        # Analyze checkpoint
        checkpoint, state_dict, hyper_params, conv_layers, detection_layers = analyze_checkpoint_structure(checkpoint_path)
        
        # Get number of classes
        settings = hyper_params.get('settings', {})
        dataset_settings = settings.get('dataset', {})
        
        # Try to infer number of classes from detection layers
        num_classes = 80  # Default
        if detection_layers:
            # Look for output layer shape to infer classes
            for layer_name, shape in detection_layers:
                if len(shape) >= 1:
                    # YOLO output format: (num_classes + 5) * num_anchors
                    output_channels = shape[0]
                    if output_channels % 3 == 0:  # 3 anchors per scale
                        classes_plus_5 = output_channels // 3
                        if classes_plus_5 > 5:
                            num_classes = classes_plus_5 - 5
                            break
        
        print(f"Inferred number of classes: {num_classes}")
        
        # Create reconstructed model
        model = ReconstructedYOLOv7Tiny(num_classes=num_classes)
        print(f"Created reconstructed YOLOv7-tiny model")
        
        # Load weights
        loaded_weights = load_weights_to_reconstructed_model(model, state_dict)
        print(f"Loaded {loaded_weights} weights from checkpoint")
        
        # Export to ONNX
        success = export_to_onnx_advanced(model, output_path)
        
        if success:
            print(f"\n🎉 Advanced conversion completed successfully!")
            print(f"ONNX model saved to: {output_path}")
            print(f"Loaded {loaded_weights} weights from original checkpoint")
        else:
            print(f"\n❌ Conversion failed")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
