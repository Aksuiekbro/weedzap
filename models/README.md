# LaserWeed YOLO Models

This directory contains YOLO models for weed detection in the LaserWeed system. Models are loaded client-side using TensorFlow.js for real-time inference.

## Directory Structure

```
models/
├── yolov5s-weed/          # Pre-trained weed detection model
│   ├── model.json         # TensorFlow.js model architecture
│   ├── model.bin          # Model weights
│   └── classes.json       # Class labels
├── custom-models/         # User uploaded models
│   └── your-model/        # Custom model folder
└── convert_model.py       # Model conversion script
```

## Supported Model Formats

- **TensorFlow.js**: Ready-to-use format (`.json` + `.bin`)
- **ONNX**: Can be converted to TensorFlow.js
- **PyTorch**: Requires conversion via ONNX

## Model Requirements

### Performance Considerations
- **Model Size**: Keep under 50MB for good browser performance
- **Input Size**: Recommended 640x640 or 416x416 for real-time inference
- **Quantization**: Use INT8 or FP16 for faster inference

### File Structure
Each model should have:
- `model.json` - TensorFlow.js model architecture
- `model.bin` - Model weights (or multiple .bin files)
- `classes.json` - Class labels and configuration

## Converting Models to TensorFlow.js

### From PyTorch (YOLOv5/YOLOv8)

```bash
# 1. Install dependencies
pip install tensorflowjs torch torchvision ultralytics

# 2. Export to ONNX
python -c "
import torch
from ultralytics import YOLO

# Load your trained model
model = YOLO('your-model.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640)
"

# 3. Convert ONNX to TensorFlow.js
tensorflowjs_converter --input_format=onnx --output_format=tfjs_graph_model \
    your-model.onnx models/custom-models/your-model/
```

### From ONNX Directly

```bash
# Install converter
pip install tensorflowjs

# Convert ONNX to TensorFlow.js
tensorflowjs_converter --input_format=onnx --output_format=tfjs_graph_model \
    your-model.onnx models/custom-models/your-model/
```

### Optimization Options

```bash
# Quantize model for smaller size and faster inference
tensorflowjs_converter --input_format=onnx --output_format=tfjs_graph_model \
    --quantize_float16 \
    your-model.onnx models/custom-models/your-model/
```

## Model Configuration

### classes.json Format

```json
{
  "classes": [
    "weed",
    "crop",
    "soil"
  ],
  "modelType": "yolov5",
  "inputSize": [640, 640],
  "anchors": [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326]
  ],
  "strides": [8, 16, 32],
  "threshold": 0.5,
  "iouThreshold": 0.45
}
```

## Pre-trained Models

### YOLOv5s-Weed
- **Description**: Lightweight weed detection model
- **Classes**: Common weeds and crops
- **Size**: ~28MB
- **Performance**: ~30 FPS on modern browsers

### Custom Models
Place your converted models in the `custom-models/` directory with the following structure:
```
custom-models/
└── your-model-name/
    ├── model.json
    ├── model.bin
    └── classes.json
```

## Usage in Web Interface

1. **Loading Models**: Use the model selector dropdown in the web interface
2. **Switching Models**: Select different models without page reload
3. **Upload Models**: Drag and drop model files to upload new models
4. **Performance**: Monitor inference time and accuracy in the interface

## Training Your Own Models

### Dataset Preparation
1. Collect weed images with bounding box annotations
2. Use tools like LabelImg or Roboflow for annotation
3. Split data into train/validation/test sets

### Training with YOLOv5
```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install requirements
pip install -r requirements.txt

# Train model
python train.py --data your-dataset.yaml --weights yolov5s.pt --epochs 100
```

### Training with YOLOv8
```bash
# Install ultralytics
pip install ultralytics

# Train model
yolo detect train data=your-dataset.yaml model=yolov8s.pt epochs=100
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check model file paths
   - Verify TensorFlow.js format
   - Ensure classes.json exists

2. **Slow Inference**
   - Reduce model size
   - Use quantization
   - Check browser WebGL support

3. **Poor Detection**
   - Adjust confidence threshold
   - Retrain with more data
   - Verify class labels

### Browser Compatibility
- **Chrome**: Full support with WebGL
- **Firefox**: Good support
- **Safari**: Limited WebGL support
- **Mobile**: May have performance limitations

## Performance Optimization

### Model Optimization
- Use quantized models (INT8/FP16)
- Reduce input resolution if needed
- Prune unnecessary layers

### Browser Optimization
- Enable WebGL acceleration
- Use Web Workers for inference
- Implement model caching

## Example Integration

```javascript
// Load model
const model = await tf.loadGraphModel('/models/yolov5s-weed/model.json');

// Preprocess image
const input = tf.image.resizeBilinear(image, [640, 640]);
const normalized = input.div(255.0);

// Run inference
const predictions = model.predict(normalized.expandDims(0));

// Post-process results
const detections = await processYOLOOutput(predictions);
```

## Resources

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [Model Conversion Guide](https://www.tensorflow.org/js/guide/conversion)

## Support

For issues with model integration, check:
1. Browser console for errors
2. Model file integrity
3. TensorFlow.js version compatibility
4. WebGL support in browser