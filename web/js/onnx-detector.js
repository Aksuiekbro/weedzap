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
     * Load ONNX model with robust error handling and recovery
     * @param {string} modelPath - Path to .onnx file
     * @param {string} configPath - Path to config.json file
     * @returns {Promise<boolean>} Success status
     */
    async loadModel(modelPath, configPath) {
        if (this.isLoading) {
            console.log('⚠️ Model is already loading...');
            return false;
        }

        this.isLoading = true;
        this.loadingStartTime = performance.now();
        console.log(`🔄 Loading ONNX model from: ${modelPath}`);

        try {
            // Phase 1: Load and validate configuration
            const configResult = await this._loadModelConfig(configPath);
            if (!configResult.success) {
                throw new Error(`Configuration error: ${configResult.error}`);
            }

            // Phase 2: Validate model file accessibility
            const modelValidation = await this._validateModelFile(modelPath);
            if (!modelValidation.success) {
                throw new Error(`Model validation error: ${modelValidation.error}`);
            }

            // Phase 3: Initialize ONNX session with timeout and retry
            const sessionResult = await this._initializeONNXSession(modelPath);
            if (!sessionResult.success) {
                throw new Error(`Session initialization error: ${sessionResult.error}`);
            }

            // Phase 4: Final validation and setup
            const loadTime = performance.now() - this.loadingStartTime;
            this.isLoaded = true;
            this.isLoading = false;
            this.lastLoadTime = loadTime;
            
            console.log(`✅ ONNX model loaded successfully in ${loadTime.toFixed(1)}ms!`);
            console.log(`📊 Model: ${this.modelConfig.name}`);
            console.log(`📊 Classes: ${this.classes.join(', ')}`);
            console.log(`📊 Input size: ${this.inputSize[0]}x${this.inputSize[1]}`);
            console.log(`📊 Memory usage: ~${this.modelConfig.performance?.size_mb || 'unknown'}MB`);
            
            return true;
            
        } catch (error) {
            return this._handleLoadingError(error, modelPath, configPath);
        }
    }

    /**
     * Load and validate model configuration
     * @private
     */
    async _loadModelConfig(configPath) {
        try {
            console.log(`📋 Loading configuration: ${configPath}`);
            
            // Add timeout to config loading
            const configController = new AbortController();
            const configTimeout = setTimeout(() => configController.abort(), 10000); // 10s timeout
            
            const configResponse = await fetch(configPath, { 
                signal: configController.signal,
                cache: 'no-cache'
            });
            clearTimeout(configTimeout);
            
            if (!configResponse.ok) {
                return { 
                    success: false, 
                    error: `HTTP ${configResponse.status}: ${configResponse.statusText}` 
                };
            }

            const configText = await configResponse.text();
            let config;
            
            try {
                config = JSON.parse(configText);
            } catch (parseError) {
                return { 
                    success: false, 
                    error: `Invalid JSON format: ${parseError.message}` 
                };
            }

            // Validate required configuration fields
            const validation = this._validateConfig(config);
            if (!validation.valid) {
                return { 
                    success: false, 
                    error: `Config validation failed: ${validation.errors.join(', ')}` 
                };
            }

            // Apply configuration with defaults
            this.modelConfig = config;
            this.inputSize = config.input_size || [640, 640];
            this.classes = config.classes || [];
            this.anchors = config.anchors || [];
            this.strides = config.strides || [8, 16, 32];
            this.threshold = config.threshold || 0.5;
            this.iouThreshold = config.iou_threshold || 0.45;
            this.inputName = config.input_name || 'images';
            this.outputNames = config.output_names || ['output'];

            console.log('✅ Configuration loaded and validated');
            return { success: true };
            
        } catch (error) {
            if (error.name === 'AbortError') {
                return { success: false, error: 'Configuration loading timeout (10s)' };
            }
            return { success: false, error: error.message };
        }
    }

    /**
     * Validate model configuration
     * @private
     */
    _validateConfig(config) {
        const errors = [];
        
        if (!config.name) errors.push('Missing model name');
        if (!config.classes || !Array.isArray(config.classes) || config.classes.length === 0) {
            errors.push('Missing or invalid classes array');
        }
        if (!config.input_size || !Array.isArray(config.input_size) || config.input_size.length !== 2) {
            errors.push('Missing or invalid input_size');
        }
        if (config.threshold && (config.threshold < 0 || config.threshold > 1)) {
            errors.push('Threshold must be between 0 and 1');
        }
        
        return { valid: errors.length === 0, errors };
    }

    /**
     * Validate model file accessibility
     * @private
     */
    async _validateModelFile(modelPath) {
        try {
            console.log(`🔍 Validating model file: ${modelPath}`);
            
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5000); // 5s timeout
            
            const response = await fetch(modelPath, { 
                method: 'HEAD',
                signal: controller.signal,
                cache: 'no-cache'
            });
            clearTimeout(timeout);
            
            if (!response.ok) {
                return { 
                    success: false, 
                    error: `Model file not accessible: HTTP ${response.status}` 
                };
            }

            const contentLength = response.headers.get('Content-Length');
            const fileSize = contentLength ? parseInt(contentLength) : 0;
            
            if (fileSize === 0) {
                return { 
                    success: false, 
                    error: 'Model file appears to be empty' 
                };
            }

            if (fileSize > 500 * 1024 * 1024) { // 500MB limit
                console.warn(`⚠️ Large model file detected: ${(fileSize / 1024 / 1024).toFixed(1)}MB`);
            }

            console.log(`✅ Model file validated: ${(fileSize / 1024 / 1024).toFixed(1)}MB`);
            return { success: true, fileSize };
            
        } catch (error) {
            if (error.name === 'AbortError') {
                return { success: false, error: 'Model file validation timeout' };
            }
            return { success: false, error: `File validation failed: ${error.message}` };
        }
    }

    /**
     * Initialize ONNX session with retry logic
     * @private
     */
    async _initializeONNXSession(modelPath, maxRetries = 2) {
        let lastError = null;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                console.log(`🚀 Initializing ONNX session (attempt ${attempt}/${maxRetries})`);
                
                // Dispose previous session if exists
                if (this.session) {
                    this.session = null;
                }
                
                // PHASE 3 DEEP FIX: Use ONNX Runtime Web (same as working test)
                console.log('🔧 Using ONNX Runtime Web for maximum compatibility');
                this.session = await ort.InferenceSession.create(modelPath, {
                    executionProviders: ['wasm'], // Use WASM for compatibility
                    logSeverityLevel: 0 // Enable verbose logging
                });
                
                const loadStartTime = performance.now();
                const loadTime = performance.now() - loadStartTime;
                
                console.log(`✅ ONNX session initialized in ${loadTime.toFixed(1)}ms`);
                
                // PHASE 3 ADDITION: Model introspection for output compatibility
                this._introspectModel();
                
                // PHASE 3 DEEP FIX: Test with dummy input to verify model functionality
                await this._testModelWithDummyInput();
                
                return { success: true, loadTime };
                
            } catch (error) {
                lastError = error;
                console.error(`❌ Session init attempt ${attempt} failed:`, error.message);
                
                if (attempt < maxRetries) {
                    console.log(`🔄 Retrying in ${attempt}s...`);
                    await new Promise(resolve => setTimeout(resolve, attempt * 1000));
                }
            }
        }
        
        return { 
            success: false, 
            error: `Failed after ${maxRetries} attempts: ${lastError?.message || 'Unknown error'}` 
        };
    }

    /**
     * Load model with timeout
     * @private
     */
    async _loadModelWithTimeout(modelPath, timeoutMs) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Model loading timeout after ${timeoutMs}ms`));
            }, timeoutMs);
            
            this.session.loadModel(modelPath)
                .then(() => {
                    clearTimeout(timeout);
                    resolve();
                })
                .catch((error) => {
                    clearTimeout(timeout);
                    reject(error);
                });
        });
    }

    /**
     * Introspect loaded model for compatibility analysis
     * @private
     */
    _introspectModel() {
        try {
            console.log('🔍 PHASE 3: Model introspection starting...');
            
            // PHASE 3 DEEP FIX: Check ONNX Runtime Web version and capabilities
            console.log('🔍 ONNX Runtime Web version:', ort.version || 'unknown');
            console.log('🔍 Available execution providers:', ort.env.wasm ? 'WASM available' : 'WASM not available');
            
            if (!this.session) {
                console.warn('⚠️ No session available for introspection');
                return;
            }
            
            // Check if session has input/output metadata
            if (this.session.inputNames) {
                console.log('📥 Model input names:', this.session.inputNames);
                
                // Verify our input name matches
                if (this.session.inputNames.length > 0) {
                    const actualInputName = this.session.inputNames[0];
                    if (actualInputName !== this.inputName) {
                        console.warn(`⚠️ Input name mismatch! Config: "${this.inputName}", Model: "${actualInputName}"`);
                        console.log(`🔄 Updating input name to match model: "${actualInputName}"`);
                        this.inputName = actualInputName;
                    }
                }
            }
            
            if (this.session.outputNames) {
                console.log('📤 Model output names:', this.session.outputNames);
                
                // Verify our output names match
                if (this.session.outputNames.length > 0) {
                    const actualOutputNames = this.session.outputNames;
                    const configOutputName = this.outputNames[0];
                    
                    if (!actualOutputNames.includes(configOutputName)) {
                        console.warn(`⚠️ Output name mismatch! Config: "${configOutputName}", Model: [${actualOutputNames.join(', ')}]`);
                        console.log(`🔄 Updating output names to match model: [${actualOutputNames.join(', ')}]`);
                        this.outputNames = actualOutputNames;
                    }
                }
            }
            
            // Log final configuration
            console.log('✅ Model introspection complete:');
            console.log(`📥 Input name: "${this.inputName}"`);
            console.log(`📤 Output names: [${this.outputNames.join(', ')}]`);
            
        } catch (error) {
            console.warn('⚠️ Model introspection failed:', error.message);
        }
    }

    /**
     * Test model with dummy input to verify basic functionality
     * @private
     */
    async _testModelWithDummyInput() {
        try {
            console.log('🧪 PHASE 3 DEEP FIX: Testing model with dummy input...');
            
            // Create dummy input tensor with correct dimensions [1, 3, 640, 640]
            const dummyData = new Float32Array(1 * 3 * 640 * 640);
            
            // Fill with normalized values (0-1 range, typical for YOLO models)
            for (let i = 0; i < dummyData.length; i++) {
                dummyData[i] = Math.random(); // Random values between 0-1
            }
            
            const dummyTensor = new ort.Tensor('float32', dummyData, [1, 3, 640, 640]);
            console.log('🧪 Created dummy tensor:', {
                type: dummyTensor.type,
                dims: dummyTensor.dims,
                dataLength: dummyTensor.data.length
            });
            
            // Create input object
            const testInputs = {};
            testInputs[this.inputName] = dummyTensor;
            
            console.log('🧪 Running dummy inference...');
            const testOutputs = await this.session.run(testInputs);
            
            console.log('🧪 Dummy inference results:');
            console.log('🧪 Output type:', typeof testOutputs);
            console.log('🧪 Output constructor:', testOutputs?.constructor?.name);
            console.log('🧪 Is Map?', testOutputs instanceof Map);
            console.log('🧪 Raw outputs:', testOutputs);
            
            // ONNX Runtime Web returns outputs as Object (not Map)
            if (testOutputs && typeof testOutputs === 'object') {
                const keys = Object.keys(testOutputs);
                console.log('🧪 Object keys:', keys);
                
                if (keys.length > 0) {
                    console.log('✅ DUMMY TEST SUCCESS: Model produces outputs as Object!');
                    const firstKey = keys[0];
                    const firstOutput = testOutputs[firstKey];
                    console.log('🧪 First output key:', firstKey);
                    console.log('🧪 First output tensor:', {
                        type: firstOutput?.type,
                        dims: firstOutput?.dims,
                        dataLength: firstOutput?.data?.length
                    });
                    
                    // Update our expected output names
                    if (!keys.includes(this.outputNames[0])) {
                        console.log(`🔄 Updating output names from [${this.outputNames.join(', ')}] to [${keys.join(', ')}]`);
                        this.outputNames = keys;
                    }
                } else {
                    console.error('❌ DUMMY TEST FAILED: Model returns empty Object');
                }
            } else {
                console.error('❌ DUMMY TEST FAILED: Unknown output format:', testOutputs);
            }
            
        } catch (error) {
            console.error('❌ Dummy input test failed:', error);
            console.error('❌ This suggests fundamental ONNX.js or model compatibility issues');
            throw error;
        }
    }

    /**
     * Handle loading errors with recovery strategies
     * @private
     */
    _handleLoadingError(error, modelPath, configPath) {
        this.isLoading = false;
        this.isLoaded = false;
        this.lastError = error;
        this.lastErrorTime = Date.now();
        
        // Categorize error for better user feedback
        let errorCategory = 'unknown';
        let userMessage = error.message;
        let canRetry = false;
        
        if (error.message.includes('timeout')) {
            errorCategory = 'timeout';
            userMessage = 'Model loading timed out. Please check your network connection.';
            canRetry = true;
        } else if (error.message.includes('HTTP 404') || error.message.includes('not accessible')) {
            errorCategory = 'not_found';
            userMessage = 'Model file not found. Please check the model path.';
            canRetry = false;
        } else if (error.message.includes('Invalid JSON') || error.message.includes('Config validation')) {
            errorCategory = 'config_error';
            userMessage = 'Model configuration is invalid.';
            canRetry = false;
        } else if (error.message.includes('Session initialization')) {
            errorCategory = 'session_error';
            userMessage = 'Failed to initialize ONNX runtime. Browser may not support this model.';
            canRetry = true;
        }
        
        console.error(`❌ ONNX loading failed (${errorCategory}):`, error);
        console.error(`📋 Model: ${modelPath}`);
        console.error(`📋 Config: ${configPath}`);
        console.error(`📋 Can retry: ${canRetry}`);
        
        // Store error details for debugging
        this.lastLoadingError = {
            category: errorCategory,
            message: userMessage,
            originalError: error.message,
            modelPath,
            configPath,
            canRetry,
            timestamp: Date.now()
        };
        
        throw new Error(userMessage);
    }

    /**
     * Run detection on an image
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageElement 
     * @returns {Array} Array of detection objects
     */
    async detect(imageElement) {
        if (!this.isLoaded || !this.session) {
            console.warn('⚠️ ONNX model not loaded');
            return [];
        }

        try {
            console.log('🔍 Starting ONNX detection...');
            
            // Get image dimensions
            const originalWidth = imageElement.width || imageElement.videoWidth || imageElement.naturalWidth;
            const originalHeight = imageElement.height || imageElement.videoHeight || imageElement.naturalHeight;
            
            console.log(`📐 Original image dimensions: ${originalWidth}x${originalHeight}`);
            
            // Preprocess image
            console.log('🔧 Preprocessing image...');
            const inputTensor = this.preprocessImage(imageElement);
            console.log('✅ Image preprocessed, tensor shape:', inputTensor.dims);
            
            // PHASE 3 ADDITION: Enhanced input validation
            console.log('🔍 PHASE 3: Input validation...');
            if (!inputTensor || !inputTensor.data) {
                throw new Error('Invalid input tensor: missing data');
            }
            if (!inputTensor.dims || inputTensor.dims.length !== 4) {
                throw new Error(`Invalid input tensor shape: expected 4D tensor, got ${inputTensor.dims}`);
            }
            
            const [batch, channels, height, width] = inputTensor.dims;
            if (batch !== 1) {
                console.warn(`⚠️ Unexpected batch size: ${batch}, expected 1`);
            }
            if (channels !== 3) {
                console.warn(`⚠️ Unexpected channels: ${channels}, expected 3`);
            }
            if (height !== this.inputSize[1] || width !== this.inputSize[0]) {
                console.warn(`⚠️ Size mismatch: tensor ${width}x${height}, expected ${this.inputSize[0]}x${this.inputSize[1]}`);
            }
            
            // Validate data range
            const dataArray = Array.from(inputTensor.data);
            const minVal = Math.min(...dataArray.slice(0, 100)); // Sample first 100 values
            const maxVal = Math.max(...dataArray.slice(0, 100));
            console.log(`🔍 Input data range (sample): [${minVal.toFixed(3)}, ${maxVal.toFixed(3)}]`);
            
            if (minVal < -10 || maxVal > 10) {
                console.warn(`⚠️ Unusual input data range: [${minVal}, ${maxVal}] - may need normalization`);
            }
            
            // Create input object
            const inputs = {};
            inputs[this.inputName] = inputTensor;
            console.log(`🎯 Created input object with key: "${this.inputName}"`);
            console.log('🎯 Input tensor type:', inputTensor.type);
            console.log('🎯 Input tensor dims:', inputTensor.dims);
            console.log('🎯 Input tensor data length:', inputTensor.data.length);
            console.log('🎯 Expected data length:', batch * channels * height * width);
            
            // Run inference
            console.log('🚀 Running ONNX inference...');
            const startTime = performance.now();
            const outputs = await this.session.run(inputs);
            const inferenceTime = performance.now() - startTime;
            console.log(`⚡ Inference completed in ${inferenceTime.toFixed(2)}ms`);
            
            // PHASE 3 FIX: Enhanced output debugging and parsing
            console.log('🔍 DEBUG: Raw ONNX outputs:', outputs);
            console.log('🔍 DEBUG: Output type:', typeof outputs);
            console.log('🔍 DEBUG: Output constructor:', outputs?.constructor?.name);
            console.log('🔍 DEBUG: Is Map?', outputs instanceof Map);
            console.log('🔍 DEBUG: Is Object?', outputs && typeof outputs === 'object' && !(outputs instanceof Map));
            console.log('🔍 DEBUG: Expected output name:', this.outputNames[0]);
            
            // ONNX Runtime Web returns Object format
            let outputTensor = null;
            let availableKeys = [];
            
            if (outputs && typeof outputs === 'object') {
                console.log('✅ Detected Object output format (ONNX Runtime Web)');
                availableKeys = Object.keys(outputs);
                console.log('🔍 Available output keys:', availableKeys);
                
                outputTensor = outputs[this.outputNames[0]];
                
                if (!outputTensor && availableKeys.length > 0) {
                    // Try first available key
                    const firstKey = availableKeys[0];
                    console.log(`🔄 Trying first available key: ${firstKey}`);
                    outputTensor = outputs[firstKey];
                }
            } else {
                console.error('❌ Unknown output format:', outputs);
                return [];
            }
            
            if (!outputTensor) {
                console.error(`❌ Output tensor not found for key: ${this.outputNames[0]}`);
                console.error('❌ Available output keys:', availableKeys);
                console.error('❌ Output size/length:', outputs instanceof Map ? outputs.size : availableKeys.length);
                
                if (availableKeys.length === 0) {
                    console.error('❌ CRITICAL: No outputs returned from model inference!');
                    console.error('❌ This suggests input preprocessing or model compatibility issues');
                    
                    // Log input details for debugging
                    console.error('🔍 Input debugging:');
                    console.error('🔍 Input name used:', this.inputName);
                    console.error('🔍 Input tensor type:', inputs[this.inputName]?.type);
                    console.error('🔍 Input tensor dims:', inputs[this.inputName]?.dims);
                    console.error('🔍 Input tensor data length:', inputs[this.inputName]?.data?.length);
                    
                    return [];
                }
                
                // Try to use any available output as fallback
                const fallbackKey = availableKeys[0];
                console.log(`🔄 Using fallback output key: ${fallbackKey}`);
                outputTensor = outputs[fallbackKey];
            }
            
            console.log('📦 Output tensor found:', outputTensor);
            console.log('📦 Output tensor type:', outputTensor?.type);
            console.log('📦 Output tensor dims:', outputTensor?.dims);
            
            // Post-process results
            console.log('🔄 Post-processing results...');
            const detections = await this.postProcess(
                outputTensor, 
                originalWidth, 
                originalHeight
            );
            
            console.log(`✅ ONNX detection completed, found ${detections.length} detections`);
            return detections;
            
        } catch (error) {
            console.error('❌ ONNX detection failed:', error);
            console.error('❌ Error stack:', error.stack);
            console.error('❌ Session state:', {
                isLoaded: this.isLoaded,
                sessionExists: !!this.session,
                inputName: this.inputName,
                outputNames: this.outputNames
            });
            return [];
        }
    }

    /**
     * Preprocess image for ONNX input
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageElement 
     * @returns {onnx.Tensor} Preprocessed tensor
     */
    preprocessImage(imageElement) {
        try {
            console.log('🔧 Preprocessing image for ONNX input');
            console.log('🔧 Target input size:', this.inputSize);
            
            // Create canvas for image processing
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = this.inputSize[0];
            canvas.height = this.inputSize[1];
            
            console.log(`🔧 Canvas created: ${canvas.width}x${canvas.height}`);
            
            // Draw and resize image
            ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
            console.log('✅ Image drawn to canvas');
            
            // Get image data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            console.log(`📊 Image data extracted: ${data.length} bytes (RGBA)`);
            
            // Convert to RGB float array [1, 3, H, W]
            const totalPixels = canvas.height * canvas.width;
            const inputArray = new Float32Array(1 * 3 * totalPixels);
            
            console.log(`📊 Creating input array: ${inputArray.length} elements`);
            
            for (let i = 0; i < canvas.height; i++) {
                for (let j = 0; j < canvas.width; j++) {
                    const pixelIndex = (i * canvas.width + j) * 4; // RGBA index
                    const tensorIndex = i * canvas.width + j; // HW index
                    
                    // Normalize to [0, 1] and arrange as CHW (Channel-Height-Width)
                    inputArray[tensorIndex] = data[pixelIndex] / 255.0; // R channel
                    inputArray[totalPixels + tensorIndex] = data[pixelIndex + 1] / 255.0; // G channel  
                    inputArray[2 * totalPixels + tensorIndex] = data[pixelIndex + 2] / 255.0; // B channel
                }
            }
            
            // Verify input array statistics (avoid spread operator for large arrays)
            let minVal = inputArray[0];
            let maxVal = inputArray[0];
            let sum = 0;
            
            for (let i = 0; i < inputArray.length; i++) {
                const val = inputArray[i];
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
                sum += val;
            }
            const meanVal = sum / inputArray.length;
            
            console.log(`📊 Input tensor statistics: min=${minVal.toFixed(3)}, max=${maxVal.toFixed(3)}, mean=${meanVal.toFixed(3)}`);
            
            // Create ONNX Runtime Web tensor
            const tensorShape = [1, 3, canvas.height, canvas.width];
            const tensor = new ort.Tensor('float32', inputArray, tensorShape);
            
            console.log('✅ ONNX tensor created successfully');
            console.log('📊 Tensor shape:', tensorShape);
            console.log('📊 Tensor type:', tensor.type);
            console.log('📊 Tensor data length:', tensor.data.length);
            
            return tensor;
            
        } catch (error) {
            console.error('❌ Preprocessing failed:', error);
            throw error;
        }
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
            // Comprehensive debugging for tensor analysis
            console.log('🔍 DEBUG: postProcess called');
            console.log('🔍 DEBUG: predictions type:', typeof predictions);
            console.log('🔍 DEBUG: predictions constructor:', predictions?.constructor?.name);
            console.log('🔍 DEBUG: predictions object:', predictions);
            
            if (predictions && typeof predictions === 'object') {
                console.log('🔍 DEBUG: predictions keys:', Object.keys(predictions));
                console.log('🔍 DEBUG: predictions prototype:', Object.getPrototypeOf(predictions));
            }
            
            // Handle different possible tensor formats with robust error checking
            let predData = null;
            let predShape = null;
            
            // Method 1: Standard ONNX.js tensor format (most common)
            if (predictions && typeof predictions.data !== 'undefined' && predictions.dims) {
                predData = predictions.data;
                predShape = predictions.dims;
                console.log('✅ Using standard ONNX tensor format (data + dims)');
                console.log('📊 Data type:', typeof predData, 'Length:', predData?.length);
                console.log('📊 Shape:', predShape);
            }
            
            // Method 2: Alternative ONNX.js format with cpuData
            else if (predictions && typeof predictions.cpuData !== 'undefined' && predictions.dims) {
                predData = predictions.cpuData;
                predShape = predictions.dims;
                console.log('✅ Using ONNX cpuData format');
                console.log('📊 cpuData type:', typeof predData, 'Length:', predData?.length);
                console.log('📊 Shape:', predShape);
            }
            
            // Method 3: ONNX.js tensor with different property names
            else if (predictions && typeof predictions.getData === 'function') {
                predData = predictions.getData();
                predShape = predictions.dims || predictions.shape;
                console.log('✅ Using ONNX tensor getData() method');
                console.log('📊 getData result:', typeof predData, 'Length:', predData?.length);
                console.log('📊 Shape:', predShape);
            }
            
            // Method 4: Check for .type property (ONNX.js tensor indicator)
            else if (predictions && predictions.type && (predictions.data || predictions.cpuData)) {
                predData = predictions.data || predictions.cpuData;
                predShape = predictions.dims || predictions.shape || [1, predData?.length || 0];
                console.log('✅ Using ONNX tensor with type property');
                console.log('📊 Tensor type:', predictions.type);
                console.log('📊 Data length:', predData?.length);
                console.log('📊 Shape:', predShape);
            }
            
            // Method 5: Direct array format (fallback)
            else if (Array.isArray(predictions)) {
                predData = predictions;
                predShape = [1, predictions.length];
                console.log('✅ Using direct array format (fallback)');
                console.log('📊 Array length:', predData.length);
            }
            
            // Method 6: Generic object search (last resort)
            else if (predictions && typeof predictions === 'object') {
                console.log('🔍 Attempting generic object property search...');
                
                // Try common property names
                const possibleDataProps = ['data', 'cpuData', 'values', 'output', 'tensor', 'result'];
                const possibleShapeProps = ['dims', 'shape', 'dimensions'];
                
                for (let dataProp of possibleDataProps) {
                    if (predictions[dataProp] !== undefined) {
                        predData = predictions[dataProp];
                        console.log(`📊 Found data in property: ${dataProp}`);
                        break;
                    }
                }
                
                for (let shapeProp of possibleShapeProps) {
                    if (predictions[shapeProp] !== undefined) {
                        predShape = predictions[shapeProp];
                        console.log(`📊 Found shape in property: ${shapeProp}`);
                        break;
                    }
                }
                
                // If we found data but no shape, estimate it
                if (predData && !predShape) {
                    predShape = [1, predData.length || 0];
                    console.log('📊 Estimated shape:', predShape);
                }
                
                if (predData) {
                    console.log('✅ Using generic object property search');
                } else {
                    console.log('❌ No data found in generic search');
                }
            }
            
            // Final validation
            if (!predData || predData.length === 0) {
                console.error('❌ No valid prediction data found');
                console.error('Available properties:', Object.keys(predictions || {}));
                return [];
            }
            
            if (!predShape || predShape.length === 0) {
                console.warn('⚠️ No shape information, using fallback');
                predShape = [1, predData.length];
            }
            
            console.log('📊 Final ONNX Output shape:', predShape);
            console.log('📊 Final ONNX Output data length:', predData.length);
            console.log('📊 Data sample (first 10 values):', Array.from(predData).slice(0, 10));
            
            // Validate data format
            if (!Array.isArray(predData) && !predData.length) {
                console.error('❌ Prediction data is not array-like:', typeof predData);
                return [];
            }
            
            // Parse detections based on output format
            let detections = [];
            
            if (predShape.length === 3 && predShape[2] > 5) {
                // Format: [batch, num_detections, 5+classes] - YOLOv7/YOLOv5 output
                console.log('🎯 Parsing YOLOv7 output format');
                detections = this.parseYOLOv7Output(predData, predShape, originalWidth, originalHeight);
            } else if (predShape.length === 4) {
                // Format: [batch, channels, height, width] - Grid-based output
                console.log('🎯 Parsing grid-based output format');
                detections = this.parseGridOutput(predData, predShape, originalWidth, originalHeight);
            } else if (predShape.length === 2 && predShape[1] > 5) {
                // Format: [num_detections, 5+classes] - Flattened output
                console.log('🎯 Parsing flattened output format');
                detections = this.parseYOLOv7Output(predData, [1, ...predShape], originalWidth, originalHeight);
            } else {
                console.warn('⚠️ Unknown output format, attempting YOLOv7 parsing as fallback');
                console.log('Shape analysis:', {
                    length: predShape.length,
                    dimensions: predShape,
                    totalElements: predShape.reduce((a, b) => a * b, 1)
                });
                
                // Try to guess the format
                const totalElements = predData.length;
                const numClasses = this.classes.length || 2;
                const outputSize = 5 + numClasses; // x,y,w,h,conf + classes
                const possibleDetections = Math.floor(totalElements / outputSize);
                
                if (possibleDetections > 0) {
                    console.log(`🔍 Guessing format: [1, ${possibleDetections}, ${outputSize}]`);
                    detections = this.parseYOLOv7Output(predData, [1, possibleDetections, outputSize], originalWidth, originalHeight);
                }
            }
            
            console.log(`🎯 Raw detections found: ${detections.length}`);
            
            // Apply confidence threshold
            const filteredDetections = detections.filter(det => det.confidence >= this.threshold);
            console.log(`🎯 After confidence threshold (${this.threshold}): ${filteredDetections.length}`);
            
            // Apply NMS
            const finalDetections = this.applyNMS(filteredDetections);
            console.log(`🎯 After NMS: ${finalDetections.length}`);
            
            return finalDetections;
            
        } catch (error) {
            console.error('❌ Post-processing failed:', error);
            console.error('❌ Error stack:', error.stack);
            console.error('❌ Predictions object that caused error:', predictions);
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
     * Dispose of the model and free memory with comprehensive cleanup
     */
    dispose() {
        console.log('🧹 Starting ONNX model disposal...');
        
        try {
            // Performance monitoring
            const disposeStart = performance.now();
            let memoryBefore = null;
            
            // Get memory info if available
            if (performance.memory) {
                memoryBefore = {
                    used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                    total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024)
                };
            }
            
            // 1. Dispose ONNX session
            if (this.session) {
                try {
                    // Call dispose method if available
                    if (typeof this.session.dispose === 'function') {
                        this.session.dispose();
                        console.log('✅ ONNX session disposed properly');
                    } else {
                        console.log('⚠️ ONNX session dispose method not available');
                    }
                } catch (sessionError) {
                    console.warn('⚠️ Error disposing ONNX session:', sessionError);
                }
                this.session = null;
            }
            
            // 2. Clear model configuration and data
            this.modelConfig = null;
            this.classes = [];
            this.anchors = [];
            this.inputSize = [640, 640];
            this.strides = [8, 16, 32];
            
            // 3. Clear cached tensors and temporary data
            if (this.lastInputTensor) {
                this.lastInputTensor = null;
            }
            if (this.lastOutputTensor) {
                this.lastOutputTensor = null;
            }
            
            // 4. Clear error states
            this.lastError = null;
            this.lastErrorTime = null;
            this.lastLoadingError = null;
            
            // 5. Reset state flags
            this.isLoaded = false;
            this.isLoading = false;
            
            // 6. Clear performance metrics
            this.lastLoadTime = null;
            this.loadingStartTime = null;
            this.inferenceCount = 0;
            this.totalInferenceTime = 0;
            
            const disposeTime = performance.now() - disposeStart;
            
            // Memory check after disposal
            if (performance.memory && memoryBefore) {
                const memoryAfter = {
                    used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                    total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024)
                };
                const memoryFreed = memoryBefore.used - memoryAfter.used;
                
                console.log(`📊 Memory before disposal: ${memoryBefore.used}MB`);
                console.log(`📊 Memory after disposal: ${memoryAfter.used}MB`);
                if (memoryFreed > 0) {
                    console.log(`✅ Memory freed: ${memoryFreed}MB`);
                }
            }
            
            console.log(`✅ ONNX model disposed in ${disposeTime.toFixed(1)}ms`);
            
            // Suggest garbage collection if available
            if (window.gc && typeof window.gc === 'function') {
                setTimeout(() => {
                    try {
                        window.gc();
                        console.log('🗑️ Garbage collection triggered');
                    } catch (gcError) {
                        console.log('⚠️ Manual garbage collection failed:', gcError);
                    }
                }, 100);
            }
            
        } catch (error) {
            console.error('❌ Error during disposal:', error);
            // Ensure state is reset even if disposal fails
            this.session = null;
            this.isLoaded = false;
            this.modelConfig = null;
        }
    }

    /**
     * Get memory usage information
     * @returns {Object} Memory usage details
     */
    getMemoryUsage() {
        const usage = {
            jsHeap: null,
            modelEstimate: null,
            tensorCache: 0,
            sessionActive: !!this.session
        };
        
        if (performance.memory) {
            usage.jsHeap = {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
            };
        }
        
        if (this.modelConfig && this.modelConfig.performance && this.modelConfig.performance.size_mb) {
            usage.modelEstimate = this.modelConfig.performance.size_mb;
        }
        
        // Estimate tensor cache size
        if (this.lastInputTensor && this.lastInputTensor.data) {
            usage.tensorCache += this.lastInputTensor.data.length * 4 / 1024 / 1024; // 4 bytes per float32
        }
        if (this.lastOutputTensor && this.lastOutputTensor.data) {
            usage.tensorCache += this.lastOutputTensor.data.length * 4 / 1024 / 1024;
        }
        
        return usage;
    }

    /**
     * Check if memory usage is within acceptable limits
     * @returns {Object} Memory status
     */
    checkMemoryHealth() {
        const usage = this.getMemoryUsage();
        const status = {
            healthy: true,
            warnings: [],
            critical: false
        };
        
        if (usage.jsHeap) {
            const memoryUsagePercent = (usage.jsHeap.used / usage.jsHeap.limit) * 100;
            
            if (memoryUsagePercent > 90) {
                status.healthy = false;
                status.critical = true;
                status.warnings.push(`Critical memory usage: ${memoryUsagePercent.toFixed(1)}%`);
            } else if (memoryUsagePercent > 70) {
                status.healthy = false;
                status.warnings.push(`High memory usage: ${memoryUsagePercent.toFixed(1)}%`);
            }
        }
        
        if (usage.tensorCache > 50) {
            status.healthy = false;
            status.warnings.push(`Large tensor cache: ${usage.tensorCache.toFixed(1)}MB`);
        }
        
        return status;
    }

    /**
     * Comprehensive model validation before loading
     * @param {string} modelPath - Path to model file
     * @param {string} configPath - Path to config file
     * @returns {Promise<Object>} Validation result
     */
    async validateModel(modelPath, configPath) {
        console.log(`🔍 Starting comprehensive model validation...`);
        const validationStart = performance.now();
        
        const result = {
            valid: false,
            errors: [],
            warnings: [],
            details: {},
            score: 0 // Compatibility score 0-100
        };

        try {
            // 1. Validate file paths and accessibility
            const pathValidation = await this._validateModelPaths(modelPath, configPath);
            result.details.paths = pathValidation;
            
            if (!pathValidation.valid) {
                result.errors.push(...pathValidation.errors);
                return result;
            }

            // 2. Load and validate configuration
            const configValidation = await this._validateModelConfiguration(configPath);
            result.details.config = configValidation;
            
            if (!configValidation.valid) {
                result.errors.push(...configValidation.errors);
                result.warnings.push(...configValidation.warnings);
            }

            // 3. Validate model file format and structure
            const modelValidation = await this._validateModelStructure(modelPath);
            result.details.model = modelValidation;
            
            if (!modelValidation.valid) {
                result.errors.push(...modelValidation.errors);
                result.warnings.push(...modelValidation.warnings);
            }

            // 4. Check browser compatibility
            const browserValidation = this._validateBrowserCompatibility();
            result.details.browser = browserValidation;
            
            if (!browserValidation.valid) {
                result.errors.push(...browserValidation.errors);
                result.warnings.push(...browserValidation.warnings);
            }

            // 5. Calculate compatibility score
            result.score = this._calculateCompatibilityScore(result.details);
            result.valid = result.errors.length === 0 && result.score >= 70;

            const validationTime = performance.now() - validationStart;
            console.log(`✅ Model validation completed in ${validationTime.toFixed(1)}ms`);
            console.log(`📊 Compatibility score: ${result.score}/100`);

            return result;

        } catch (error) {
            result.errors.push(`Validation failed: ${error.message}`);
            console.error('❌ Model validation error:', error);
            return result;
        }
    }

    /**
     * Validate model file paths and accessibility
     * @private
     */
    async _validateModelPaths(modelPath, configPath) {
        const result = { valid: true, errors: [], warnings: [] };

        try {
            // Check model file
            const modelCheck = await fetch(modelPath, { method: 'HEAD' });
            if (!modelCheck.ok) {
                result.valid = false;
                result.errors.push(`Model file not accessible: ${modelPath} (${modelCheck.status})`);
            } else {
                const modelSize = parseInt(modelCheck.headers.get('Content-Length') || '0');
                if (modelSize === 0) {
                    result.valid = false;
                    result.errors.push('Model file appears to be empty');
                } else if (modelSize > 100 * 1024 * 1024) { // 100MB
                    result.warnings.push(`Large model file: ${(modelSize / 1024 / 1024).toFixed(1)}MB`);
                }
            }

            // Check config file
            const configCheck = await fetch(configPath, { method: 'HEAD' });
            if (!configCheck.ok) {
                result.valid = false;
                result.errors.push(`Config file not accessible: ${configPath} (${configCheck.status})`);
            }

        } catch (error) {
            result.valid = false;
            result.errors.push(`Path validation failed: ${error.message}`);
        }

        return result;
    }

    /**
     * Validate model configuration structure and content
     * @private
     */
    async _validateModelConfiguration(configPath) {
        const result = { valid: true, errors: [], warnings: [], details: {} };

        try {
            const response = await fetch(configPath);
            const config = await response.json();
            result.details.config = config;

            // Required fields validation
            const requiredFields = ['name', 'classes', 'input_size', 'output_names'];
            for (const field of requiredFields) {
                if (!config[field]) {
                    result.valid = false;
                    result.errors.push(`Missing required field: ${field}`);
                }
            }

            // Classes validation
            if (config.classes) {
                if (!Array.isArray(config.classes) || config.classes.length === 0) {
                    result.valid = false;
                    result.errors.push('Classes must be a non-empty array');
                } else if (config.classes.length > 100) {
                    result.warnings.push('Large number of classes may impact performance');
                }
            }

            // Input size validation
            if (config.input_size) {
                if (!Array.isArray(config.input_size) || config.input_size.length !== 2) {
                    result.valid = false;
                    result.errors.push('Input size must be [width, height] array');
                } else {
                    const [width, height] = config.input_size;
                    if (width < 32 || height < 32) {
                        result.valid = false;
                        result.errors.push('Input size too small (minimum 32x32)');
                    } else if (width > 2048 || height > 2048) {
                        result.warnings.push('Large input size may impact performance');
                    }
                }
            }

            // Threshold validation
            if (config.threshold !== undefined) {
                if (config.threshold < 0 || config.threshold > 1) {
                    result.valid = false;
                    result.errors.push('Threshold must be between 0 and 1');
                }
            }

            // Model type validation
            if (config.type && config.type !== 'onnx') {
                result.warnings.push(`Model type '${config.type}' - expected 'onnx'`);
            }

        } catch (error) {
            result.valid = false;
            result.errors.push(`Config validation failed: ${error.message}`);
        }

        return result;
    }

    /**
     * Validate model structure and format
     * @private
     */
    async _validateModelStructure(modelPath) {
        const result = { valid: true, errors: [], warnings: [], details: {} };

        try {
            // Basic ONNX file validation
            const response = await fetch(modelPath, { method: 'HEAD' });
            const contentType = response.headers.get('Content-Type');
            
            // Check file extension
            if (!modelPath.toLowerCase().endsWith('.onnx')) {
                result.warnings.push('File does not have .onnx extension');
            }

            // Check content type if available
            if (contentType && !contentType.includes('application/octet-stream') && 
                !contentType.includes('application/x-protobuf')) {
                result.warnings.push(`Unexpected content type: ${contentType}`);
            }

            // File size analysis
            const fileSize = parseInt(response.headers.get('Content-Length') || '0');
            result.details.fileSize = fileSize;
            
            if (fileSize < 1024) { // Less than 1KB
                result.valid = false;
                result.errors.push('Model file too small to be valid ONNX model');
            } else if (fileSize < 100 * 1024) { // Less than 100KB
                result.warnings.push('Model file seems unusually small');
            }

            // Try to perform a quick ONNX compatibility check
            const compatibilityCheck = await this._quickONNXCompatibilityCheck();
            result.details.onnxSupport = compatibilityCheck;
            
            if (!compatibilityCheck.supported) {
                result.valid = false;
                result.errors.push('ONNX.js not available or not compatible');
            }

        } catch (error) {
            result.valid = false;
            result.errors.push(`Model structure validation failed: ${error.message}`);
        }

        return result;
    }

    /**
     * Check browser compatibility with ONNX
     * @private
     */
    _validateBrowserCompatibility() {
        const result = { valid: true, errors: [], warnings: [], details: {} };

        try {
            // Check ONNX Runtime Web availability
            if (typeof ort === 'undefined') {
                result.valid = false;
                result.errors.push('ONNX Runtime Web library not loaded');
                return result;
            }

            // Check WebGL support
            const canvas = document.createElement('canvas');
            const webglContext = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            result.details.webgl = !!webglContext;
            
            if (!webglContext) {
                result.warnings.push('WebGL not available - inference will use CPU (slower)');
            }

            // Check WebAssembly support
            result.details.wasm = typeof WebAssembly !== 'undefined';
            if (!result.details.wasm) {
                result.warnings.push('WebAssembly not supported - limited performance');
            }

            // Check available memory
            if (navigator.deviceMemory) {
                result.details.deviceMemory = navigator.deviceMemory;
                if (navigator.deviceMemory < 2) {
                    result.warnings.push('Low device memory - may impact performance');
                }
            }

            // Check performance API
            result.details.performanceAPI = typeof performance !== 'undefined' && 
                                          typeof performance.now === 'function';

            // Browser-specific checks
            const userAgent = navigator.userAgent;
            result.details.browser = this._detectBrowser(userAgent);
            
            if (result.details.browser.name === 'Safari' && 
                result.details.browser.version < 14) {
                result.warnings.push('Safari version may have limited ONNX support');
            }

        } catch (error) {
            result.warnings.push(`Browser compatibility check failed: ${error.message}`);
        }

        return result;
    }

    /**
     * Quick ONNX compatibility check
     * @private
     */
    async _quickONNXCompatibilityCheck() {
        const result = { supported: false, version: null, features: {} };

        try {
            if (typeof ort !== 'undefined') {
                result.supported = true;
                result.version = ort.version || 'unknown';
                
                // Check if InferenceSession is available
                result.features.inferenceSession = typeof ort.InferenceSession === 'object' && typeof ort.InferenceSession.create === 'function';
                
                // Check if Tensor is available
                result.features.tensor = typeof ort.Tensor === 'function';
                
                // Check WASM support
                result.features.wasmSupport = ort.env?.wasm !== undefined;
                
                result.features.sessionCreation = true; // ONNX Runtime Web doesn't need dummy session test
            }
        } catch (error) {
            console.warn('ONNX Runtime Web compatibility check failed:', error);
        }

        return result;
    }

    /**
     * Detect browser type and version
     * @private
     */
    _detectBrowser(userAgent) {
        let browser = { name: 'unknown', version: 0 };

        if (userAgent.includes('Chrome')) {
            browser.name = 'Chrome';
            const match = userAgent.match(/Chrome\/(\d+)/);
            browser.version = match ? parseInt(match[1]) : 0;
        } else if (userAgent.includes('Firefox')) {
            browser.name = 'Firefox';
            const match = userAgent.match(/Firefox\/(\d+)/);
            browser.version = match ? parseInt(match[1]) : 0;
        } else if (userAgent.includes('Safari')) {
            browser.name = 'Safari';
            const match = userAgent.match(/Version\/(\d+)/);
            browser.version = match ? parseInt(match[1]) : 0;
        } else if (userAgent.includes('Edge')) {
            browser.name = 'Edge';
            const match = userAgent.match(/Edg\/(\d+)/);
            browser.version = match ? parseInt(match[1]) : 0;
        }

        return browser;
    }

    /**
     * Calculate overall compatibility score
     * @private
     */
    _calculateCompatibilityScore(details) {
        let score = 100;

        // Deduct points for errors and warnings
        const errors = Object.values(details).reduce((sum, detail) => 
            sum + (detail.errors ? detail.errors.length : 0), 0);
        const warnings = Object.values(details).reduce((sum, detail) => 
            sum + (detail.warnings ? detail.warnings.length : 0), 0);

        score -= errors * 25; // 25 points per error
        score -= warnings * 5; // 5 points per warning

        // Bonus points for good features
        if (details.browser?.webgl) score += 10;
        if (details.browser?.wasm) score += 5;
        if (details.browser?.deviceMemory >= 4) score += 5;

        return Math.max(0, Math.min(100, score));
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ONNXDetector;
}
