# api/analyze.py - Vercel serverless function
"""
Serverless API for crop/weed detection using ONNX
Deploy to Vercel for instant global scaling.
"""

import json
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import time

# Global model session (cached across requests)
model_session = None

def load_model():
    """Load ONNX model (cached)."""
    global model_session
    if model_session is None:
        try:
            # Model should be in your project root or public folder
            model_session = ort.InferenceSession('./tiny_model_680_final.onnx')
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model_session

def handler(request):
    """Vercel serverless function handler."""
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': headers,
            'body': json.dumps({'error': 'Method not allowed'})
        }
    
    try:
        # Load model
        session = load_model()
        if not session:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': 'Model not available'})
            }
        
        # Parse request
        body = json.loads(request.body)
        image_data = body.get('image')
        
        if not image_data:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'error': 'No image provided'})
            }
        
        start_time = time.time()
        
        # Process image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((640, 640))
        
        # Convert to model input format
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image_array})
        
        # Quick analysis
        total_detections = 0
        for output in outputs:
            # Simple confidence counting
            confident = np.sum(output > 0.3)
            total_detections += confident
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'success': True,
                'detections': int(total_detections),
                'processing_time': processing_time,
                'message': 'Analysis complete'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }
