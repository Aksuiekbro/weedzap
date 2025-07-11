/**
 * YOLO Detector for LaserWeed System
 * 
 * This class handles loading and running YOLO models using TensorFlow.js
 * for real-time weed detection in the browser.
 */
class YOLODetector {
    constructor() {
        this.model = null;
        this.modelConfig = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.inputSize = [640, 640];
        this.classes = [];
        this.anchors = [];
        this.strides = [8, 16, 32];
        this.threshold = 0.5;
        this.iouThreshold = 0.45;
        this.modelType = 'yolov5'; // Default model type
        this.modelVariant = 'base';
    }

    /**
     * Load YOLO model from TensorFlow.js format
     * @param {string} modelPath - Path to model.json file
     * @param {string} configPath - Path to classes.json file
     */
    async loadModel(modelPath, configPath) {
        try {
            this.isLoading = true;
            console.log(`Loading YOLO model from: ${modelPath}`);

            // Load model configuration
            const configResponse = await fetch(configPath);
            if (!configResponse.ok) {
                throw new Error(`Failed to load config: ${configResponse.status}`);
            }
            this.modelConfig = await configResponse.json();
            
            // Set configuration
            this.classes = this.modelConfig.classes || [];
            this.inputSize = this.modelConfig.inputSize || [640, 640];
            this.anchors = this.modelConfig.anchors || [];
            this.strides = this.modelConfig.strides || [8, 16, 32];
            this.threshold = this.modelConfig.threshold || 0.5;
            this.iouThreshold = this.modelConfig.iouThreshold || 0.45;
            this.modelType = this.modelConfig.modelType || 'yolov5';
            this.modelVariant = this.extractModelVariant(this.modelConfig.modelType || 'yolov5');
            
            // Adjust settings for YOLOv7 models
            if (this.modelType.includes('yolov7')) {
                this.adjustYOLOv7Settings();
            }

            // Load TensorFlow.js model
            this.model = await tf.loadGraphModel(modelPath);
            
            console.log(`Model loaded successfully with ${this.classes.length} classes`);
            console.log(`Input size: ${this.inputSize[0]}x${this.inputSize[1]}`);
            
            this.isLoaded = true;
            this.isLoading = false;
            
            return true;
            
        } catch (error) {
            console.error('Error loading YOLO model:', error);
            this.isLoaded = false;
            this.isLoading = false;
            throw error;
        }
    }

    /**
     * Run detection on an image
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageElement 
     * @returns {Array} Array of detection objects
     */
    async detect(imageElement) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        try {
            // Preprocess image
            const input = this.preprocessImage(imageElement);
            
            // Run inference
            const predictions = this.model.predict(input);
            
            // Post-process results
            const detections = await this.postProcess(predictions, imageElement.width, imageElement.height);
            
            // Clean up tensors
            input.dispose();
            if (Array.isArray(predictions)) {
                predictions.forEach(pred => pred.dispose());
            } else {
                predictions.dispose();
            }
            
            return detections;
            
        } catch (error) {
            console.error('Error during detection:', error);
            throw error;
        }
    }

    /**
     * Preprocess image for YOLO input
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageElement 
     * @returns {tf.Tensor} Preprocessed tensor
     */
    preprocessImage(imageElement) {
        // Convert image to tensor
        const tensor = tf.browser.fromPixels(imageElement);
        
        // Resize to model input size
        const resized = tf.image.resizeBilinear(tensor, this.inputSize);
        
        // Normalize to [0, 1]
        const normalized = resized.div(255.0);
        
        // Add batch dimension
        const batched = normalized.expandDims(0);
        
        // Clean up intermediate tensors
        tensor.dispose();
        resized.dispose();
        normalized.dispose();
        
        return batched;
    }

    /**
     * Post-process YOLO predictions
     * @param {tf.Tensor|Array} predictions - Raw model predictions
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array} Array of detection objects
     */
    async postProcess(predictions, originalWidth, originalHeight) {
        try {
            // Handle different model output formats
            let outputTensor;
            if (Array.isArray(predictions)) {
                // Multiple outputs (YOLOv5 format)
                outputTensor = predictions[0];
            } else {
                // Single output
                outputTensor = predictions;
            }

            // Get predictions data
            const predData = await outputTensor.data();
            const predShape = outputTensor.shape;
            
            console.log('Prediction shape:', predShape);
            console.log('Prediction data length:', predData.length);
            
            // Parse predictions based on model type and output format
            let detections;
            if (this.modelType.includes('yolov7')) {
                detections = this.parseYOLOv7Output(predData, predShape, originalWidth, originalHeight);
            } else {
                detections = this.parseYOLOOutput(predData, predShape, originalWidth, originalHeight);
            }
            
            // Apply Non-Maximum Suppression
            const filteredDetections = this.applyNMS(detections);
            
            return filteredDetections;
            
        } catch (error) {
            console.error('Error in post-processing:', error);
            return [];
        }
    }

    /**
     * Parse YOLO output format
     * @param {Float32Array} predData - Prediction data
     * @param {Array} predShape - Prediction shape
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array} Array of detection objects
     */
    parseYOLOOutput(predData, predShape, originalWidth, originalHeight) {
        const detections = [];
        
        // Determine output format
        if (predShape.length === 3) {
            // Format: [batch, predictions, values]
            const numPredictions = predShape[1];
            const numValues = predShape[2];
            
            for (let i = 0; i < numPredictions; i++) {
                const startIdx = i * numValues;
                
                // Extract values: [x, y, w, h, confidence, class_probs...]
                const centerX = predData[startIdx] / this.inputSize[0] * originalWidth;
                const centerY = predData[startIdx + 1] / this.inputSize[1] * originalHeight;
                const width = predData[startIdx + 2] / this.inputSize[0] * originalWidth;
                const height = predData[startIdx + 3] / this.inputSize[1] * originalHeight;
                const confidence = predData[startIdx + 4];
                
                // Skip low confidence detections
                if (confidence < this.threshold) continue;
                
                // Find best class
                let bestClass = 0;
                let bestScore = 0;
                
                for (let j = 0; j < this.classes.length; j++) {
                    const classScore = predData[startIdx + 5 + j] * confidence;
                    if (classScore > bestScore) {
                        bestScore = classScore;
                        bestClass = j;
                    }
                }
                
                // Skip if final score is too low
                if (bestScore < this.threshold) continue;
                
                // Convert to corner coordinates
                const x = centerX - width / 2;
                const y = centerY - height / 2;
                
                detections.push({
                    x: Math.max(0, x),
                    y: Math.max(0, y),
                    width: Math.min(width, originalWidth - x),
                    height: Math.min(height, originalHeight - y),
                    confidence: bestScore,
                    class: bestClass,
                    className: this.classes[bestClass] || `class_${bestClass}`
                });
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
        if (detections.length === 0) return [];
        
        // Sort by confidence (descending)
        detections.sort((a, b) => b.confidence - a.confidence);
        
        const filtered = [];
        
        while (detections.length > 0) {
            const best = detections.shift();
            filtered.push(best);
            
            // Remove overlapping detections
            detections = detections.filter(detection => {
                const iou = this.calculateIOU(best, detection);
                return iou < this.iouThreshold;
            });
        }
        
        return filtered;
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
     * Extract model variant from model type string
     * @param {string} modelType - Model type string (e.g., 'yolov7-tiny', 'yolov5s')
     * @returns {string} Model variant
     */
    extractModelVariant(modelType) {
        const type = modelType.toLowerCase();
        
        if (type.includes('tiny')) return 'tiny';
        if (type.includes('x')) return 'x';
        if (type.includes('e6')) return 'e6';
        if (type.includes('s')) return 's';
        if (type.includes('m')) return 'm';
        if (type.includes('l')) return 'l';
        
        return 'base';
    }

    /**
     * Adjust settings specifically for YOLOv7 models
     */
    adjustYOLOv7Settings() {
        console.log(`Applying YOLOv7 optimizations for variant: ${this.modelVariant}`);
        
        // YOLOv7-specific optimizations
        switch (this.modelVariant) {
            case 'tiny':
                // Tiny models are faster but need lower thresholds for agricultural detection
                this.threshold = Math.min(this.threshold, 0.35);
                this.iouThreshold = Math.max(this.iouThreshold, 0.65);
                break;
                
            case 'x':
                // X models are more accurate, can use higher thresholds
                this.threshold = Math.max(this.threshold, 0.4);
                this.iouThreshold = 0.45;
                break;
                
            case 'e6':
                // E6 models are highest accuracy, optimize for precision
                this.threshold = Math.max(this.threshold, 0.45);
                this.iouThreshold = 0.4;
                break;
                
            default:
                // Standard YOLOv7 optimizations
                this.threshold = Math.max(this.threshold, 0.35);
                this.iouThreshold = 0.5;
        }
        
        console.log(`YOLOv7 settings: threshold=${this.threshold}, iouThreshold=${this.iouThreshold}`);
    }

    /**
     * Enhanced output parsing for YOLOv7 models
     * @param {Float32Array} predData - Prediction data
     * @param {Array} predShape - Prediction shape
     * @param {number} originalWidth - Original image width
     * @param {number} originalHeight - Original image height
     * @returns {Array} Array of detection objects
     */
    parseYOLOv7Output(predData, predShape, originalWidth, originalHeight) {
        const detections = [];
        
        // YOLOv7 typically outputs in format [batch, predictions, values]
        if (predShape.length === 3) {
            const numPredictions = predShape[1];
            const numValues = predShape[2];
            
            // YOLOv7 format: [x, y, w, h, objectness, class1, class2, ...]
            for (let i = 0; i < numPredictions; i++) {
                const startIdx = i * numValues;
                
                // Extract bounding box coordinates (center format)
                const centerX = predData[startIdx] / this.inputSize[0] * originalWidth;
                const centerY = predData[startIdx + 1] / this.inputSize[1] * originalHeight;
                const width = predData[startIdx + 2] / this.inputSize[0] * originalWidth;
                const height = predData[startIdx + 3] / this.inputSize[1] * originalHeight;
                const objectness = predData[startIdx + 4];
                
                // Skip low objectness detections
                if (objectness < this.threshold) continue;
                
                // Find best class with class probability
                let bestClass = 0;
                let bestClassProb = 0;
                
                for (let j = 0; j < this.classes.length; j++) {
                    const classProb = predData[startIdx + 5 + j];
                    if (classProb > bestClassProb) {
                        bestClassProb = classProb;
                        bestClass = j;
                    }
                }
                
                // Calculate final confidence (objectness * class_probability)
                const finalConfidence = objectness * bestClassProb;
                
                // Skip if final confidence is too low
                if (finalConfidence < this.threshold) continue;
                
                // Convert center coordinates to corner coordinates
                const x = Math.max(0, centerX - width / 2);
                const y = Math.max(0, centerY - height / 2);
                
                detections.push({
                    x: x,
                    y: y,
                    width: Math.min(width, originalWidth - x),
                    height: Math.min(height, originalHeight - y),
                    confidence: finalConfidence,
                    objectness: objectness,
                    classProb: bestClassProb,
                    class: bestClass,
                    className: this.classes[bestClass] || `class_${bestClass}`
                });
            }
        }
        
        return detections;
    }

    /**
     * Update model configuration
     * @param {Object} config - New configuration
     */
    updateConfig(config) {
        if (config.threshold !== undefined) {
            this.threshold = config.threshold;
        }
        if (config.iouThreshold !== undefined) {
            this.iouThreshold = config.iouThreshold;
        }
        console.log(`Updated config: threshold=${this.threshold}, iouThreshold=${this.iouThreshold}`);
    }

    /**
     * Get model information
     * @returns {Object} Model information
     */
    getModelInfo() {
        return {
            isLoaded: this.isLoaded,
            isLoading: this.isLoading,
            classes: this.classes,
            inputSize: this.inputSize,
            threshold: this.threshold,
            iouThreshold: this.iouThreshold,
            modelType: this.modelType,
            modelVariant: this.modelVariant,
            optimizedFor: this.modelType.includes('yolov7') ? 'agricultural_detection' : 'general_detection'
        };
    }

    /**
     * Dispose of the model and free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isLoaded = false;
        this.isLoading = false;
        console.log('YOLO model disposed');
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = YOLODetector;
}