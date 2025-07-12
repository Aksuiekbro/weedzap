#!/usr/bin/env python3
"""
ONNX Model Usage Example
This script demonstrates how to use the converted ONNX model for inference.
"""

import onnxruntime
import numpy as np
import cv2
import time
from pathlib import Path

def load_onnx_model(model_path):
    """Load the ONNX model and return session."""
    try:
        # Create inference session
        session = onnxruntime.InferenceSession(model_path)
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Input: {input_name} {input_shape}")
        print(f"📈 Outputs: {len(output_names)} layers")
        
        return session, input_name, output_names
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for YOLO inference."""
    if isinstance(image_path, str) and Path(image_path).exists():
        # Load image from file
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"📷 Loaded image: {image_path} ({image.shape})")
    else:
        # Create dummy image for testing
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"🎲 Created random test image: {image.shape}")
    
    # Resize image
    original_shape = image.shape[:2]
    image_resized = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to CHW format and add batch dimension
    image_chw = np.transpose(image_normalized, (2, 0, 1))  # HWC -> CHW
    image_batch = np.expand_dims(image_chw, axis=0)  # Add batch dimension
    
    return image_batch, original_shape

def postprocess_outputs(outputs, confidence_threshold=0.5):
    """Basic postprocessing of YOLO outputs."""
    detections = []
    
    for i, output in enumerate(outputs):
        batch_size, channels, height, width = output.shape
        
        # Reshape to [batch, num_anchors, height, width, predictions]
        num_anchors = 3
        predictions_per_anchor = channels // num_anchors
        
        output_reshaped = output.reshape(batch_size, num_anchors, predictions_per_anchor, height, width)
        output_reshaped = np.transpose(output_reshaped, (0, 1, 3, 4, 2))  # [B, A, H, W, P]
        
        # Extract confidence scores (assuming objectness is at index 4)
        if predictions_per_anchor > 4:
            objectness = output_reshaped[..., 4]  # [B, A, H, W]
            
            # Count detections above threshold
            confident_detections = np.sum(objectness > confidence_threshold)
            detections.append({
                'scale': f'Scale {i+1}',
                'shape': output.shape,
                'total_predictions': objectness.size,
                'confident_predictions': confident_detections,
                'max_confidence': np.max(objectness),
                'mean_confidence': np.mean(objectness)
            })
    
    return detections

def run_inference_example():
    """Run a complete inference example."""
    print("🚀 ONNX Model Inference Example")
    print("=" * 40)
    
    # Model path
    model_path = "custom-models/tiny_model_680_final.onnx"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("💡 Run final_lightning_to_onnx.py first to create the model")
        return
    
    # Load model
    session, input_name, output_names = load_onnx_model(model_path)
    if session is None:
        return
    
    # Preprocess image (using dummy image for demo)
    print(f"\n📝 Preprocessing image...")
    image_batch, original_shape = preprocess_image(None)  # Creates dummy image
    print(f"✅ Preprocessed image shape: {image_batch.shape}")
    
    # Run inference
    print(f"\n🔄 Running inference...")
    start_time = time.time()
    
    outputs = session.run(output_names, {input_name: image_batch})
    
    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.3f} seconds")
    
    # Postprocess outputs
    print(f"\n📊 Analyzing outputs...")
    detections = postprocess_outputs(outputs, confidence_threshold=0.1)
    
    total_confident = sum(det['confident_predictions'] for det in detections)
    
    for detection in detections:
        print(f"   {detection['scale']}: {detection['shape']}")
        print(f"      Confident predictions: {detection['confident_predictions']:,}")
        print(f"      Max confidence: {detection['max_confidence']:.4f}")
        print(f"      Mean confidence: {detection['mean_confidence']:.4f}")
    
    print(f"\n🎯 Summary:")
    print(f"   Total confident detections: {total_confident:,}")
    print(f"   Inference time: {inference_time:.3f}s")
    print(f"   FPS estimate: {1/inference_time:.1f}")
    
    # Usage tips
    print(f"\n💡 Usage Tips:")
    print(f"   • Input format: RGB images, 640x640 pixels")
    print(f"   • Output format: 3 scales for multi-scale detection")
    print(f"   • Classes: 0=crop, 1=weed (inferred)")
    print(f"   • Confidence threshold: Adjust based on your needs")
    print(f"   • Post-processing: Implement NMS for final detections")

def benchmark_model():
    """Benchmark the model performance."""
    print(f"\n⚡ Performance Benchmark")
    print("-" * 25)
    
    model_path = "custom-models/tiny_model_680_final.onnx"
    session, input_name, output_names = load_onnx_model(model_path)
    
    if session is None:
        return
    
    # Prepare test data
    test_image = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        session.run(output_names, {input_name: test_image})
    
    # Benchmark
    num_runs = 50
    start_time = time.time()
    
    for _ in range(num_runs):
        outputs = session.run(output_names, {input_name: test_image})
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    
    print(f"📊 Benchmark Results ({num_runs} runs):")
    print(f"   Average inference time: {avg_time:.4f}s")
    print(f"   Estimated FPS: {1/avg_time:.1f}")
    print(f"   Total time: {total_time:.2f}s")

def main():
    """Main function."""
    print("🎯 YOLOv7-tiny ONNX Model Usage")
    print("=" * 50)
    
    # Run inference example
    run_inference_example()
    
    # Benchmark performance
    benchmark_model()
    
    print(f"\n✨ Ready to use your converted ONNX model!")
    print(f"📚 Next steps:")
    print(f"   1. Implement proper post-processing (NMS, etc.)")
    print(f"   2. Test with real crop/weed images")
    print(f"   3. Fine-tune confidence thresholds")
    print(f"   4. Integrate into your application")

if __name__ == "__main__":
    main()
