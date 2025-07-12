#!/usr/bin/env python3
"""
Simple Flask API for Crop/Weed Detection using ONNX
Fast, lightweight backend for web deployment.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import cv2
import base64
import io
from PIL import Image
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Global model session
model_session = None
model_path = "custom-models/tiny_model_680_final.onnx"

def load_model():
    """Load ONNX model on startup."""
    global model_session
    try:
        print(f"Loading ONNX model: {model_path}")
        model_session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def preprocess_image(image_data):
    """Preprocess image for model inference."""
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to 640x640
    image = image.resize((640, 640))
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def postprocess_outputs(outputs, confidence_threshold=0.3):
    """Process model outputs into meaningful results."""
    results = {
        'total_detections': 0,
        'scales': [],
        'crop_count': 0,
        'weed_count': 0,
        'confidence_avg': 0.0
    }
    
    all_confidences = []
    
    for i, output in enumerate(outputs):
        batch_size, channels, height, width = output.shape
        
        # Reshape output for processing
        num_anchors = 3
        predictions_per_anchor = channels // num_anchors
        
        output_reshaped = output.reshape(batch_size, num_anchors, predictions_per_anchor, height, width)
        output_reshaped = np.transpose(output_reshaped, (0, 1, 3, 4, 2))
        
        # Extract predictions
        confident_detections = 0
        scale_crop_count = 0
        scale_weed_count = 0
        
        for b in range(batch_size):
            for a in range(num_anchors):
                for h in range(height):
                    for w in range(width):
                        prediction = output_reshaped[b, a, h, w]
                        
                        if len(prediction) >= 7:  # x, y, w, h, obj, class1, class2
                            objectness = prediction[4]
                            
                            if objectness > confidence_threshold:
                                confident_detections += 1
                                all_confidences.append(objectness)
                                
                                # Get class probabilities
                                class_probs = prediction[5:7]  # 2 classes
                                predicted_class = np.argmax(class_probs)
                                
                                if predicted_class == 0:
                                    scale_crop_count += 1
                                else:
                                    scale_weed_count += 1
        
        results['scales'].append({
            'scale': f'Scale {i+1}',
            'size': f'{height}x{width}',
            'detections': confident_detections,
            'crops': scale_crop_count,
            'weeds': scale_weed_count
        })
        
        results['total_detections'] += confident_detections
        results['crop_count'] += scale_crop_count
        results['weed_count'] += scale_weed_count
    
    if all_confidences:
        results['confidence_avg'] = float(np.mean(all_confidences))
    
    return results

@app.route('/')
def home():
    """Simple web interface for testing."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crop/Weed Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .results { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>🌱 Crop/Weed Detection API</h1>
        <p>Upload an image to detect crops and weeds</p>
        
        <div class="upload-area">
            <input type="file" id="imageInput" accept="image/*">
            <br><br>
            <button class="button" onclick="analyzeImage()">Analyze Image</button>
        </div>
        
        <div id="results"></div>

        <script>
            async function analyzeImage() {
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image first');
                    return;
                }
                
                // Convert to base64
                const reader = new FileReader();
                reader.onload = async function(e) {
                    const base64Image = e.target.result;
                    
                    document.getElementById('results').innerHTML = '<div>🔄 Analyzing image...</div>';
                    
                    try {
                        const response = await fetch('/analyze', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                image: base64Image,
                                confidence_threshold: 0.3
                            })
                        });
                        
                        const result = await response.json();
                        displayResults(result);
                        
                    } catch (error) {
                        document.getElementById('results').innerHTML = '<div style="color: red;">❌ Error: ' + error.message + '</div>';
                    }
                };
                reader.readAsDataURL(file);
            }
            
            function displayResults(result) {
                if (result.error) {
                    document.getElementById('results').innerHTML = '<div style="color: red;">❌ ' + result.error + '</div>';
                    return;
                }
                
                let html = '<div class="results">';
                html += '<h3>🎯 Detection Results</h3>';
                html += `<p><strong>Processing Time:</strong> ${result.processing_time}ms</p>`;
                html += `<p><strong>Total Detections:</strong> ${result.data.total_detections}</p>`;
                html += `<p><strong>Crops:</strong> ${result.data.crop_count} | <strong>Weeds:</strong> ${result.data.weed_count}</p>`;
                html += `<p><strong>Average Confidence:</strong> ${result.data.confidence_avg.toFixed(3)}</p>`;
                
                html += '<h4>Scale Breakdown:</h4>';
                result.data.scales.forEach(scale => {
                    html += `<p><strong>${scale.scale}</strong> (${scale.size}): ${scale.detections} detections (${scale.crops} crops, ${scale.weeds} weeds)</p>`;
                });
                
                html += '</div>';
                document.getElementById('results').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return html_template

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for image analysis."""
    start_time = time.time()
    
    try:
        if not model_session:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        image_data = data.get('image')
        confidence_threshold = data.get('confidence_threshold', 0.3)
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        input_array = preprocess_image(image_data)
        
        # Run inference
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: input_array})
        
        # Postprocess results
        results = postprocess_outputs(outputs, confidence_threshold)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'processing_time': processing_time,
            'data': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_session is not None,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("🚀 Starting Flask server...")
        app.run(host='0.0.0.0', port=8888, debug=True)
    else:
        print("❌ Cannot start server without model")
