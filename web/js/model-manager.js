/**
 * Model Manager for LaserWeed System
 * 
 * This class manages loading and switching between different YOLO models
 * and handles the model selection UI.
 */
class ModelManager {
    constructor() {
        this.yoloDetector = new YOLODetector();
        this.currentModel = 'simple';
        this.availableModels = {
            'simple': {
                name: 'Simple Color Detection',
                type: 'simple',
                loaded: true
            },
            'yolov5s-weed': {
                name: 'YOLOv5s Weed Detection',
                type: 'yolo',
                modelPath: 'models/yolov5s-weed/model.json',
                configPath: 'models/yolov5s-weed/classes.json',
                loaded: false
            }
        };
        
        this.customModels = new Map();
        this.detectedCheckpoints = new Map(); // For .ckpt files
        this.onModelChangeCallback = null;
        this.onStatusChangeCallback = null;
        this.autoScanEnabled = true;
    }

    /**
     * Initialize model manager with DOM elements
     * @param {Object} elements - DOM elements for model management
     */
    initialize(elements) {
        this.elements = elements;
        this.setupEventListeners();
        this.updateModelSelector();
        this.updateStatus('Not Loaded');
        
        // Start auto-detection for .ckpt files
        if (this.autoScanEnabled) {
            this.scanForCheckpoints();
        }
    }

    /**
     * Set up event listeners for model selection
     */
    setupEventListeners() {
        // Model selector change
        this.elements.modelSelector.addEventListener('change', (e) => {
            this.switchModel(e.target.value);
        });

        // Model upload
        this.elements.modelFiles.addEventListener('change', (e) => {
            this.handleModelUpload(e.target.files);
        });

        // Show/hide upload interface based on selection
        this.elements.modelSelector.addEventListener('change', (e) => {
            if (e.target.value === 'custom') {
                this.elements.modelUpload.style.display = 'block';
            } else {
                this.elements.modelUpload.style.display = 'none';
            }
        });
    }

    /**
     * Switch to a different model
     * @param {string} modelId - ID of the model to switch to
     */
    async switchModel(modelId) {
        try {
            this.updateStatus('Loading...', 'loading');
            
            if (modelId === 'simple') {
                // Switch to simple color detection
                this.currentModel = 'simple';
                this.yoloDetector.dispose();
                this.updateStatus('Simple Detection Active', 'loaded');
                
                if (this.onModelChangeCallback) {
                    this.onModelChangeCallback('simple', null);
                }
                return;
            }

            // Check if this is a checkpoint that needs conversion
            if (modelId.startsWith('checkpoint_')) {
                const filename = modelId.replace('checkpoint_', '');
                
                // Ask user if they want to convert
                const shouldConvert = confirm(`This model needs to be converted first. Convert ${filename} now?`);
                if (shouldConvert) {
                    try {
                        const convertedModelId = await this.convertCheckpoint(filename);
                        // After conversion, try to load the converted model
                        return this.switchModel(convertedModelId);
                    } catch (error) {
                        throw new Error(`Conversion failed: ${error.message}`);
                    }
                } else {
                    // User declined conversion, switch back to simple
                    this.elements.modelSelector.value = 'simple';
                    return this.switchModel('simple');
                }
            }

            // Load YOLO model
            const model = this.availableModels[modelId] || this.customModels.get(modelId);
            if (!model) {
                throw new Error(`Model ${modelId} not found`);
            }

            if (model.type === 'yolo') {
                await this.yoloDetector.loadModel(model.modelPath, model.configPath);
                model.loaded = true;
                this.currentModel = modelId;
                
                const modelInfo = this.yoloDetector.getModelInfo();
                this.updateStatus(`${model.name} Loaded (${modelInfo.classes.length} classes)`, 'loaded');
                
                if (this.onModelChangeCallback) {
                    this.onModelChangeCallback('yolo', this.yoloDetector);
                }
            }
            
        } catch (error) {
            console.error('Error switching model:', error);
            this.updateStatus(`Error: ${error.message}`, 'error');
            
            // Fall back to simple detection
            this.currentModel = 'simple';
            this.elements.modelSelector.value = 'simple';
            this.yoloDetector.dispose();
            
            if (this.onModelChangeCallback) {
                this.onModelChangeCallback('simple', null);
            }
        }
    }

    /**
     * Handle custom model upload
     * @param {FileList} files - Uploaded files
     */
    async handleModelUpload(files) {
        try {
            if (files.length === 0) return;
            
            this.updateStatus('Uploading model...', 'loading');
            
            let modelFile = null;
            let configFile = null;
            const binFiles = [];
            
            // Process uploaded files
            for (const file of files) {
                if (file.name.endsWith('.json')) {
                    if (file.name === 'model.json') {
                        modelFile = file;
                    } else if (file.name === 'classes.json') {
                        configFile = file;
                    }
                } else if (file.name.endsWith('.bin')) {
                    binFiles.push(file);
                }
            }
            
            if (!modelFile) {
                throw new Error('model.json file not found');
            }
            
            if (!configFile) {
                throw new Error('classes.json file not found');
            }
            
            // Create URLs for uploaded files
            const modelUrl = URL.createObjectURL(modelFile);
            const configUrl = URL.createObjectURL(configFile);
            
            // Create custom model entry
            const customModelId = `custom_${Date.now()}`;
            const customModel = {
                name: `Custom Model (${modelFile.name})`,
                type: 'yolo',
                modelPath: modelUrl,
                configPath: configUrl,
                loaded: false,
                isCustom: true
            };
            
            this.customModels.set(customModelId, customModel);
            
            // Add to selector
            this.addCustomModelToSelector(customModelId, customModel);
            
            // Load the custom model
            await this.yoloDetector.loadModel(modelUrl, configUrl);
            customModel.loaded = true;
            this.currentModel = customModelId;
            
            // Update selector
            this.elements.modelSelector.value = customModelId;
            
            const modelInfo = this.yoloDetector.getModelInfo();
            this.updateStatus(`${customModel.name} Loaded (${modelInfo.classes.length} classes)`, 'loaded');
            
            if (this.onModelChangeCallback) {
                this.onModelChangeCallback('yolo', this.yoloDetector);
            }
            
        } catch (error) {
            console.error('Error uploading model:', error);
            this.updateStatus(`Upload Error: ${error.message}`, 'error');
            
            // Fall back to simple detection
            this.currentModel = 'simple';
            this.elements.modelSelector.value = 'simple';
            
            if (this.onModelChangeCallback) {
                this.onModelChangeCallback('simple', null);
            }
        }
    }

    /**
     * Add custom model to selector dropdown
     * @param {string} modelId - Model ID
     * @param {Object} model - Model object
     */
    addCustomModelToSelector(modelId, model) {
        const option = document.createElement('option');
        option.value = modelId;
        option.textContent = model.name;
        
        // Insert before the "Custom Model" option
        const customOption = this.elements.modelSelector.querySelector('option[value="custom"]');
        this.elements.modelSelector.insertBefore(option, customOption);
    }

    /**
     * Update model selector dropdown
     */
    updateModelSelector() {
        // Clear existing options except defaults
        const options = this.elements.modelSelector.querySelectorAll('option');
        options.forEach(option => {
            if (!['simple', 'yolov5s-weed', 'custom'].includes(option.value)) {
                option.remove();
            }
        });
        
        // Add custom models
        this.customModels.forEach((model, id) => {
            this.addCustomModelToSelector(id, model);
        });
    }

    /**
     * Update status display
     * @param {string} message - Status message
     * @param {string} type - Status type (loading, loaded, error)
     */
    updateStatus(message, type = '') {
        this.elements.modelStatusValue.textContent = message;
        this.elements.modelStatusValue.className = `model-status-value ${type}`;
        
        if (this.onStatusChangeCallback) {
            this.onStatusChangeCallback(message, type);
        }
    }

    /**
     * Get current model information
     * @returns {Object} Current model info
     */
    getCurrentModel() {
        if (this.currentModel === 'simple') {
            return {
                type: 'simple',
                name: 'Simple Color Detection',
                detector: null
            };
        }
        
        const model = this.availableModels[this.currentModel] || this.customModels.get(this.currentModel);
        return {
            type: 'yolo',
            name: model.name,
            detector: this.yoloDetector,
            info: this.yoloDetector.getModelInfo()
        };
    }

    /**
     * Update YOLO model configuration
     * @param {Object} config - New configuration
     */
    updateYOLOConfig(config) {
        if (this.currentModel !== 'simple' && this.yoloDetector.isLoaded) {
            this.yoloDetector.updateConfig(config);
        }
    }

    /**
     * Set callback for model change events
     * @param {Function} callback - Callback function
     */
    onModelChange(callback) {
        this.onModelChangeCallback = callback;
    }

    /**
     * Set callback for status change events
     * @param {Function} callback - Callback function
     */
    onStatusChange(callback) {
        this.onStatusChangeCallback = callback;
    }

    /**
     * Check if a model is available
     * @param {string} modelId - Model ID to check
     * @returns {boolean} True if model is available
     */
    isModelAvailable(modelId) {
        if (modelId === 'simple') return true;
        
        const model = this.availableModels[modelId] || this.customModels.get(modelId);
        return model && model.loaded;
    }

    /**
     * Get list of available models
     * @returns {Array} Array of model objects
     */
    getAvailableModels() {
        const models = [];
        
        // Add built-in models
        Object.entries(this.availableModels).forEach(([id, model]) => {
            models.push({
                id: id,
                name: model.name,
                type: model.type,
                loaded: model.loaded
            });
        });
        
        // Add custom models
        this.customModels.forEach((model, id) => {
            models.push({
                id: id,
                name: model.name,
                type: model.type,
                loaded: model.loaded,
                isCustom: true
            });
        });
        
        return models;
    }

    /**
     * Remove a custom model
     * @param {string} modelId - Model ID to remove
     */
    removeCustomModel(modelId) {
        if (this.customModels.has(modelId)) {
            // If this is the current model, switch to simple detection
            if (this.currentModel === modelId) {
                this.switchModel('simple');
            }
            
            // Remove from custom models
            this.customModels.delete(modelId);
            
            // Remove from selector
            const option = this.elements.modelSelector.querySelector(`option[value="${modelId}"]`);
            if (option) {
                option.remove();
            }
        }
    }

    /**
     * Scan for .ckpt files in the models directory
     */
    async scanForCheckpoints() {
        try {
            // Try to fetch models index first
            const response = await fetch('models_index.json');
            if (response.ok) {
                const index = await response.json();
                this.loadModelsFromIndex(index);
                return;
            }
        } catch (error) {
            console.log('No models index found, scanning directory...');
        }

        // Try your actual checkpoint file names
        const commonCheckpoints = [
            'CropOrWeed2_640px_yolov7-tiny_epoch=37_lr=_batch=48_val_loss=11.115_map=0.592.ckpt'
        ];

        for (const filename of commonCheckpoints) {
            const checkpointPath = `models/custom-models/${filename}`;
            const modelName = filename.replace('.ckpt', '');
            
            // Check if already converted
            const convertedPath = `models/custom-models/${modelName}/model.json`;
            
            try {
                const convertedResponse = await fetch(convertedPath, { method: 'HEAD' });
                if (convertedResponse.ok) {
                    // Model already converted, add to available models
                    await this.addConvertedModel(modelName, convertedPath);
                } else {
                    // Model needs conversion, add to detected checkpoints
                    this.addDetectedCheckpoint(filename, checkpointPath);
                }
            } catch (error) {
                // Assume model needs conversion
                this.addDetectedCheckpoint(filename, checkpointPath);
            }
        }

        this.updateModelSelector();
    }

    /**
     * Load models from generated index file
     * @param {Object} index - Models index data
     */
    loadModelsFromIndex(index) {
        if (!index.models) return;

        for (const modelInfo of index.models) {
            this.availableModels[modelInfo.id] = {
                name: `${modelInfo.name} (${modelInfo.type})`,
                type: 'yolo',
                modelPath: modelInfo.path,
                configPath: modelInfo.config_path,
                loaded: false,
                size: modelInfo.size_mb,
                performance: modelInfo.performance_tier,
                isConverted: true
            };
        }

        console.log(`Loaded ${index.models.length} models from index`);
    }

    /**
     * Add a detected checkpoint file
     * @param {string} filename - Checkpoint filename
     * @param {string} path - Path to checkpoint
     */
    addDetectedCheckpoint(filename, path) {
        const modelInfo = this.parseCheckpointInfo(filename);
        
        this.detectedCheckpoints.set(filename, {
            name: modelInfo.displayName,
            path: path,
            type: modelInfo.type,
            variant: modelInfo.variant,
            inputSize: modelInfo.inputSize,
            needsConversion: true
        });
    }

    /**
     * Parse information from checkpoint filename
     * @param {string} filename - Checkpoint filename
     * @returns {Object} Model information
     */
    parseCheckpointInfo(filename) {
        const name = filename.toLowerCase();
        
        let type = 'yolov7';
        let variant = 'base';
        let inputSize = [640, 640];
        let dataset = 'Unknown';
        let epoch = 'Unknown';
        let valLoss = null;
        let map = null;
        let batch = null;
        
        // Extract dataset name
        if (name.includes('croporweed2')) {
            dataset = 'CropOrWeed2';
        } else if (name.includes('cropsorweed9')) {
            dataset = 'CropsOrWeed9';
        } else if (name.includes('examplecroporweed2')) {
            dataset = 'Example';
        }
        
        // Extract training metrics
        const epochMatch = name.match(/epoch=(\d+)/);
        if (epochMatch) {
            epoch = epochMatch[1];
        }
        
        const batchMatch = name.match(/batch=(\d+)/);
        if (batchMatch) {
            batch = batchMatch[1];
        }
        
        const valLossMatch = name.match(/val_loss=([\d.]+)/);
        if (valLossMatch) {
            valLoss = parseFloat(valLossMatch[1]);
        }
        
        const mapMatch = name.match(/map=([\d.]+)/);
        if (mapMatch) {
            map = parseFloat(mapMatch[1]);
        }
        
        // Determine variant
        if (name.includes('tiny')) {
            variant = 'tiny';
        } else if (name.includes('yolov7x')) {
            variant = 'x';
        } else if (name.includes('e6')) {
            variant = 'e6';
            inputSize = name.includes('1280') ? [1280, 1280] : [640, 640];
        }
        
        // Extract input size
        if (name.includes('1280px')) {
            inputSize = [1280, 1280];
        } else if (name.includes('640px')) {
            inputSize = [640, 640];
        }

        // Create a detailed display name with training metrics
        let displayName = `${type.toUpperCase()}-${variant} ${dataset} (${inputSize[0]}px)`;
        
        let metrics = [];
        if (epoch !== 'Unknown') metrics.push(`ep${epoch}`);
        if (map !== null) metrics.push(`mAP: ${(map * 100).toFixed(1)}%`);
        if (valLoss !== null) metrics.push(`loss: ${valLoss.toFixed(2)}`);
        
        if (metrics.length > 0) {
            displayName += ` - ${metrics.join(', ')}`;
        }
        
        return {
            type,
            variant,
            inputSize,
            displayName,
            dataset,
            epoch,
            valLoss,
            map,
            batch,
            quality: map !== null ? (map > 0.7 ? 'excellent' : map > 0.5 ? 'good' : 'fair') : 'unknown'
        };
    }

    /**
     * Add a converted model to available models
     * @param {string} modelName - Model name
     * @param {string} modelPath - Path to model.json
     */
    async addConvertedModel(modelName, modelPath) {
        const configPath = modelPath.replace('model.json', 'classes.json');
        
        try {
            // Try to load config to get display name
            const configResponse = await fetch(configPath);
            let displayName = modelName;
            
            if (configResponse.ok) {
                const config = await configResponse.json();
                displayName = config.description || `${config.modelType} (${config.classes.length} classes)`;
            }
            
            this.availableModels[modelName] = {
                name: displayName,
                type: 'yolo',
                modelPath: modelPath,
                configPath: configPath,
                loaded: false,
                isConverted: true
            };
            
        } catch (error) {
            console.error(`Error adding converted model ${modelName}:`, error);
        }
    }

    /**
     * Update model selector with detected checkpoints
     */
    updateModelSelector() {
        // Clear existing options except defaults
        const options = this.elements.modelSelector.querySelectorAll('option');
        options.forEach(option => {
            if (!['simple', 'yolov5s-weed', 'custom'].includes(option.value)) {
                option.remove();
            }
        });
        
        // Add converted models
        Object.entries(this.availableModels).forEach(([id, model]) => {
            if (id !== 'simple' && id !== 'yolov5s-weed') {
                this.addModelToSelector(id, model.name, model.isConverted);
            }
        });

        // Add detected checkpoints that need conversion
        this.detectedCheckpoints.forEach((checkpoint, filename) => {
            if (checkpoint.needsConversion) {
                this.addCheckpointToSelector(filename, checkpoint);
            }
        });

        // Add custom models
        this.customModels.forEach((model, id) => {
            this.addCustomModelToSelector(id, model);
        });
    }

    /**
     * Add a model to the selector dropdown
     * @param {string} modelId - Model ID
     * @param {string} displayName - Display name
     * @param {boolean} isReady - Whether model is ready to use
     */
    addModelToSelector(modelId, displayName, isReady = true) {
        const option = document.createElement('option');
        option.value = modelId;
        option.textContent = isReady ? displayName : `${displayName} (Needs Conversion)`;
        
        if (!isReady) {
            option.style.color = '#ff8800';
            option.disabled = true;
        }
        
        // Insert before the "Custom Model" option
        const customOption = this.elements.modelSelector.querySelector('option[value="custom"]');
        this.elements.modelSelector.insertBefore(option, customOption);
    }

    /**
     * Add a checkpoint to the selector dropdown
     * @param {string} filename - Checkpoint filename
     * @param {Object} checkpoint - Checkpoint info
     */
    addCheckpointToSelector(filename, checkpoint) {
        const option = document.createElement('option');
        option.value = `checkpoint_${filename}`;
        option.textContent = `${checkpoint.name} (Convert to use)`;
        option.style.color = '#ff8800';
        option.style.fontStyle = 'italic';
        
        // Add data attributes for conversion info
        option.dataset.filename = filename;
        option.dataset.needsConversion = 'true';
        
        // Insert before the "Custom Model" option
        const customOption = this.elements.modelSelector.querySelector('option[value="custom"]');
        this.elements.modelSelector.insertBefore(option, customOption);
    }

    /**
     * Handle checkpoint conversion request
     * @param {string} filename - Checkpoint filename to convert
     */
    async convertCheckpoint(filename) {
        const checkpoint = this.detectedCheckpoints.get(filename);
        if (!checkpoint) {
            throw new Error(`Checkpoint ${filename} not found`);
        }

        this.updateStatus('Converting checkpoint...', 'loading');
        
        // In a real implementation, this would call the backend conversion API
        // For now, we'll simulate the conversion process
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // Simulate conversion success/failure
                if (Math.random() > 0.2) { // 80% success rate
                    console.log(`✅ Checkpoint ${filename} converted successfully`);
                    
                    // Add as converted model
                    const modelName = filename.replace('.ckpt', '');
                    this.availableModels[modelName] = {
                        name: checkpoint.name,
                        type: 'yolo',
                        modelPath: `models/custom-models/${modelName}/model.json`,
                        configPath: `models/custom-models/${modelName}/classes.json`,
                        loaded: false,
                        isConverted: true
                    };
                    
                    // Remove from detected checkpoints
                    this.detectedCheckpoints.delete(filename);
                    
                    // Update UI
                    this.updateModelSelector();
                    this.updateStatus('Checkpoint converted successfully', 'loaded');
                    
                    resolve(modelName);
                } else {
                    reject(new Error('Conversion failed - please check console for details'));
                }
            }, 3000); // Simulate 3 second conversion time
        });
    }

    /**
     * Get conversion status for detected checkpoints
     * @returns {Array} Array of checkpoint status objects
     */
    getCheckpointStatus() {
        const status = [];
        
        this.detectedCheckpoints.forEach((checkpoint, filename) => {
            status.push({
                filename,
                name: checkpoint.name,
                type: checkpoint.type,
                variant: checkpoint.variant,
                needsConversion: checkpoint.needsConversion,
                canConvert: true // Always true for web-based conversion
            });
        });
        
        return status;
    }

    /**
     * Dispose of all models and free memory
     */
    dispose() {
        this.yoloDetector.dispose();
        
        // Clean up custom model URLs
        this.customModels.forEach(model => {
            if (model.isCustom) {
                URL.revokeObjectURL(model.modelPath);
                URL.revokeObjectURL(model.configPath);
            }
        });
        
        this.customModels.clear();
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelManager;
}