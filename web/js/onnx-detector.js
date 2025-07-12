/**
 * ONNX Detector for LaserWeed System
 * 
 * This class handles loading and running ONNX models using ONNX.js
 * for real-time weed detection in the browser.
 */
class ONNXDetector {
    constructor() {
        this.session = null;
        this.modelConfig = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.inputSize = [640, 640];
        this.classes = [];
        this.anchors = [];
        this.strides = [8, 16, 32];
        this.threshold = 0.5;
        this.iouThreshold = 0.45;
        this.inputName = 'images';
        this.outputNames = ['output'];
    }

    /**
     * Load ONNX model
     * @param {string} modelPath - Path to .onnx file
     * @param {string} configPath - Path to config.json file
     */
    async loadModel(modelPath, configPath) {
        if (this.isLoading) {
            console.log('Model is already loading...');
            return false;
        }

        this.isLoading = true;
        console.log(`Loading ONNX model from: ${modelPath}`);

        try {
            // Load model configuration
            const configResponse = await fetch(configPath);
            if (!configResponse.ok) {
                throw new Error(`Failed to load config: ${configResponse.status}`);
            }
            this.modelConfig = await configResponse.json();
            
            // Apply configuration
            this.inputSize = this.modelConfig.input_size || [640, 640];
            this.classes = this.modelConfig.classes || [];
            this.anchors = this.modelConfig.anchors || [];
            this.strides = this.modelConfig.strides || [8, 16, 32];
            this.threshold = this.modelConfig.threshold || 0.5;
            this.iouThreshold = this.modelConfig.iou_threshold || 0.45;
            this.inputName = this.modelConfig.input_name || 'images';
            this.outputNames = this.modelConfig.output_names || ['output'];

            console.log('Model config loaded:', this.modelConfig);

            // Initialize ONNX session
            this.session = new onnx.InferenceSession();
            
            // Load the ONNX model
            await this.session.loadModel(modelPath);
            
            this.isLoaded = true;
            this.isLoading = false;
            
            console.log('✅ ONNX model loaded successfully!');
            console.log(`Model info: ${this.modelConfig.name}`);
            console.log(`Classes: ${this.classes.join(', ')}`);
            console.log(`Input size: ${this.inputSize[0]}x${this.inputSize[1]}`);
            
            return true;
            
        } catch (error) {
            this.isLoading = false;
            this.isLoaded = false;
            console.error('❌ Failed to load ONNX model:', error);
            throw error;
        }
    }

    /**
     * Run detection on an image
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageElement 
     * @returns {Array} Array of detection objects
     */
    async detect(imageElement) {
        if (!this.isLoaded || !this.session) {
            console.warn('ONNX model not loaded');
            return [];
        }

        try {
            // Preprocess image
            const inputTensor = this.preprocessImage(imageElement);
            
            // Create input object
            const inputs = {};
            inputs[this.inputName] = inputTensor;
            
            // Run inference
            const outputs = await this.session.run(inputs);
            
            // Get output tensor
            const outputTensor = outputs[this.outputNames[0]];
            
            // Post-process results
            const detections = await this.postProcess(
                outputTensor, 
                imageElement.width || imageElement.videoWidth, 
                imageElement.height || imageElement.videoHeight
            );
            
            return detections;
            
        } catch (error) {
            console.error('Detection failed:', error);
            return [];
        }
    }

    /**
     * Preprocess image for ONNX input
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageElement 
     * @returns {onnx.Tensor} Preprocessed tensor
     */
    preprocessImage(imageElement) {
        // Create canvas for image processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = this.inputSize[0];
        canvas.height = this.inputSize[1];
        
        // Draw and resize image
        ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
        
        // Get image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Convert to RGB float array [1, 3, H, W]
        const inputArray = new Float32Array(1 * 3 * canvas.height * canvas.width);
        
        for (let i = 0; i < canvas.height; i++) {
            for (let j = 0; j < canvas.width; j++) {
                const pixelIndex = (i * canvas.width + j) * 4;
                const tensorIndex = i * canvas.width + j;
                
                // Normalize to [0, 1] and arrange as CHW
                inputArray[tensorIndex] = data[pixelIndex] / 255.0; // R
                inputArray[canvas.height * canvas.width + tensorIndex] = data[pixelIndex + 1] / 255.0; // G
                inputArray[2 * canvas.height * canvas.width + tensorIndex] = data[pixelIndex + 2] / 255.0; // B
            }
        }
        
        // Create ONNX tensor
        return new onnx.Tensor(inputArray, 'float32', [1, 3, canvas.height, canvas.width]);
    }

    /**
     * Post-process ONNX predictions
     * @param {onnx.Tensor} predictions - Raw model predictions
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array} Array of detection objects
     */
    async postProcess(predictions, originalWidth, originalHeight) {
        try {
            const predData = predictions.data;
            const predShape = predictions.dims;
            
            console.log('ONNX Output shape:', predShape);
            console.log('ONNX Output data length:', predData.length);
            
            // Parse detections based on output format
            let detections = [];
            
            if (predShape.length === 3 && predShape[2] > 5) {
                // Format: [batch, num_detections, 5+classes]
                detections = this.parseYOLOv7Output(predData, predShape, originalWidth, originalHeight);
            } else if (predShape.length === 4) {
                // Format: [batch, channels, height, width] - need to decode anchors
                detections = this.parseGridOutput(predData, predShape, originalWidth, originalHeight);
            }
            
            // Apply confidence threshold
            detections = detections.filter(det => det.confidence >= this.threshold);
            
            // Apply NMS
            detections = this.applyNMS(detections);
            
            return detections;
            
        } catch (error) {
            console.error('Post-processing failed:', error);
            return [];
        }
    }

    /**
     * Parse YOLOv7 output format
     * @param {Float32Array} predData - Prediction data
     * @param {Array} predShape - Prediction shape
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array} Array of detection objects
     */
    parseYOLOv7Output(predData, predShape, originalWidth, originalHeight) {
        const detections = [];
        const [batch, numDetections, outputSize] = predShape;
        const numClasses = outputSize - 5; // x, y, w, h, conf + classes
        
        const xScale = originalWidth / this.inputSize[0];
        const yScale = originalHeight / this.inputSize[1];
        
        for (let i = 0; i < numDetections; i++) {
            const offset = i * outputSize;
            
            // Extract box coordinates (normalized)
            const x = predData[offset] * xScale;
            const y = predData[offset + 1] * yScale;
            const w = predData[offset + 2] * xScale;
            const h = predData[offset + 3] * yScale;
            const confidence = predData[offset + 4];
            
            if (confidence >= this.threshold) {
                // Find best class
                let bestClass = 0;
                let bestScore = predData[offset + 5];
                
                for (let c = 1; c < numClasses; c++) {
                    const score = predData[offset + 5 + c];
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = c;
                    }
                }
                
                const finalScore = confidence * bestScore;
                
                if (finalScore >= this.threshold) {
                    detections.push({
                        x: Math.max(0, x - w / 2),
                        y: Math.max(0, y - h / 2),
                        width: Math.min(w, originalWidth),
                        height: Math.min(h, originalHeight),
                        confidence: finalScore,
                        class: bestClass,
                        className: this.classes[bestClass] || `class_${bestClass}`
                    });
                }
            }
        }
        
        return detections;
    }

    /**
     * Parse grid-based output format
     * @param {Float32Array} predData - Prediction data
     * @param {Array} predShape - Prediction shape [batch, channels, height, width]
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array} Array of detection objects
     */
    parseGridOutput(predData, predShape, originalWidth, originalHeight) {
        const detections = [];
        const [batch, channels, gridH, gridW] = predShape;
        const numAnchors = 3;
        const numClasses = this.classes.length;
        const stride = this.inputSize[0] / gridW;
        
        const xScale = originalWidth / this.inputSize[0];
        const yScale = originalHeight / this.inputSize[1];
        
        for (let a = 0; a < numAnchors; a++) {
            for (let y = 0; y < gridH; y++) {
                for (let x = 0; x < gridW; x++) {
                    const baseIndex = a * (5 + numClasses) * gridH * gridW + y * gridW + x;
                    
                    // Extract predictions
                    const tx = predData[baseIndex];
                    const ty = predData[baseIndex + gridH * gridW];
                    const tw = predData[baseIndex + 2 * gridH * gridW];
                    const th = predData[baseIndex + 3 * gridH * gridW];
                    const conf = this.sigmoid(predData[baseIndex + 4 * gridH * gridW]);
                    
                    if (conf >= this.threshold) {
                        // Decode box coordinates
                        const cx = (this.sigmoid(tx) + x) * stride * xScale;
                        const cy = (this.sigmoid(ty) + y) * stride * yScale;
                        const w = Math.exp(tw) * this.anchors[0][a][0] * xScale;
                        const h = Math.exp(th) * this.anchors[0][a][1] * yScale;
                        
                        // Find best class
                        let bestClass = 0;
                        let bestScore = this.sigmoid(predData[baseIndex + (5 + 0) * gridH * gridW]);
                        
                        for (let c = 1; c < numClasses; c++) {
                            const score = this.sigmoid(predData[baseIndex + (5 + c) * gridH * gridW]);
                            if (score > bestScore) {
                                bestScore = score;
                                bestClass = c;
                            }
                        }
                        
                        const finalScore = conf * bestScore;
                        
                        if (finalScore >= this.threshold) {
                            detections.push({
                                x: Math.max(0, cx - w / 2),
                                y: Math.max(0, cy - h / 2),
                                width: Math.min(w, originalWidth),
                                height: Math.min(h, originalHeight),
                                confidence: finalScore,
                                class: bestClass,
                                className: this.classes[bestClass] || `class_${bestClass}`
                            });
                        }
                    }
                }
            }
        }
        
        return detections;
    }

    /**
     * Apply Non-Maximum Suppression
     * @param {Array} detections - Array of detection objects
     * @returns {Array} Filtered detections
     */
    applyNMS(detections) {
        // Sort by confidence
        detections.sort((a, b) => b.confidence - a.confidence);
        
        const result = [];
        
        while (detections.length > 0) {
            const current = detections.shift();
            result.push(current);
            
            detections = detections.filter(det => {
                const iou = this.calculateIOU(current, det);
                return iou < this.iouThreshold;
            });
        }
        
        return result;
    }

    /**
     * Calculate Intersection over Union (IoU)
     * @param {Object} box1 - First bounding box
     * @param {Object} box2 - Second bounding box
     * @returns {number} IoU value
     */
    calculateIOU(box1, box2) {
        const x1 = Math.max(box1.x, box2.x);
        const y1 = Math.max(box1.y, box2.y);
        const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
        
        if (x2 <= x1 || y2 <= y1) return 0;
        
        const intersection = (x2 - x1) * (y2 - y1);
        const area1 = box1.width * box1.height;
        const area2 = box2.width * box2.height;
        const union = area1 + area2 - intersection;
        
        return intersection / union;
    }

    /**
     * Sigmoid activation function
     * @param {number} x - Input value
     * @returns {number} Sigmoid output
     */
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Update model configuration
     * @param {Object} config - New configuration
     */
    updateConfig(config) {
        if (config.threshold !== undefined) this.threshold = config.threshold;
        if (config.iouThreshold !== undefined) this.iouThreshold = config.iouThreshold;
        if (config.inputSize !== undefined) this.inputSize = config.inputSize;
    }

    /**
     * Get model information
     * @returns {Object} Model information
     */
    getModelInfo() {
        return {
            isLoaded: this.isLoaded,
            modelConfig: this.modelConfig,
            inputSize: this.inputSize,
            classes: this.classes,
            threshold: this.threshold,
            iouThreshold: this.iouThreshold
        };
    }

    /**
     * Dispose of the model and free memory
     */
    dispose() {
        if (this.session) {
            this.session = null;
        }
        this.isLoaded = false;
        this.modelConfig = null;
        console.log('ONNX model disposed');
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXDetector;
}
