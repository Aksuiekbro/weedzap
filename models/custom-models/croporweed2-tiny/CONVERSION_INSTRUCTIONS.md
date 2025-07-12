# Manual Conversion Instructions

## Checkpoint Information
- **File**: CropOrWeed2_640px_yolov7-tiny_epoch=37_lr=_batch=48_val_loss=11.115_map=0.592.ckpt
- **Type**: YOLOv7 Checkpoint
- **Size**: 46.1 MB

## Available Keys in Checkpoint
['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']

## Recommended Conversion Methods

### Method 1: Original YOLOv7 Export
```bash
# Clone YOLOv7 repository
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7

# Export your checkpoint
python export.py --weights custom-models/CropOrWeed2_640px_yolov7-tiny_epoch=37_lr=_batch=48_val_loss=11.115_map=0.592.ckpt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640
```

### Method 2: Ultralytics Export
```bash
# Install ultralytics
pip install ultralytics

# Try direct export
yolo export model=custom-models/CropOrWeed2_640px_yolov7-tiny_epoch=37_lr=_batch=48_val_loss=11.115_map=0.592.ckpt format=onnx imgsz=640
```

### Method 3: Custom Script
Create a custom export script based on your training setup.

## Next Steps
1. Use one of the methods above to create an ONNX file
2. Convert ONNX to TensorFlow.js using: `tensorflowjs_converter --input_format=onnx --output_format=tfjs_graph_model your_model.onnx ./output_dir`
3. Copy the generated files to replace this placeholder

## Classes Configuration
The classes.json file has been created with the detected or default class configuration.
Update it if your model uses different classes.
