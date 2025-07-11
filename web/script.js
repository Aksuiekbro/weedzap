class LaserWeedController {
    constructor() {
        this.currentPower = 0;
        this.isConnected = false;
        this.weedCount = 0;
        this.videoStreamUrl = '';
        this.esp8266Url = 'http://192.168.4.1';
        this.connectionCheckInterval = null;
        this.lastUpdateTime = Date.now();
        
        // Camera and CV properties
        this.cameraSource = 'raspberry';
        this.laptopStream = null;
        this.isProcessing = false;
        this.detectionSensitivity = 50;
        this.showBoundingBoxes = true;
        this.processingInterval = null;
        
        // Model management
        this.modelManager = new ModelManager();
        this.currentDetectionMode = 'simple';
        this.yoloDetector = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeModels();
        this.startConnectionCheck();
        this.initializeVideoStream();
        this.startClock();
    }

    initializeElements() {
        // Control elements
        this.powerSlider = document.getElementById('powerSlider');
        this.powerValue = document.getElementById('powerValue');
        this.sliderFill = document.getElementById('sliderFill');
        this.laserOnBtn = document.getElementById('laserOnBtn');
        this.laserOffBtn = document.getElementById('laserOffBtn');
        
        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.laserStatus = document.getElementById('laserStatus');
        this.weedCountElement = document.getElementById('weedCount');
        this.timestamp = document.getElementById('timestamp');
        
        // Video elements
        this.videoFeed = document.getElementById('videoFeed');
        this.videoStream = document.getElementById('videoStream');
        this.laptopVideo = document.getElementById('laptopVideo');
        this.processCanvas = document.getElementById('processCanvas');
        this.fullscreenBtn = document.getElementById('fullscreenBtn');
        
        // Camera controls
        this.cameraSelector = document.getElementById('cameraSource');
        this.detectionSettings = document.getElementById('detectionSettings');
        this.sensitivitySlider = document.getElementById('sensitivitySlider');
        this.sensitivityValue = document.getElementById('sensitivityValue');
        this.showBoundingBoxesCheckbox = document.getElementById('showBoundingBoxes');
        
        // Model controls
        this.modelSelector = document.getElementById('modelSelector');
        this.modelUpload = document.getElementById('modelUpload');
        this.modelFiles = document.getElementById('modelFiles');
        this.modelStatusValue = document.getElementById('modelStatusValue');
    }

    setupEventListeners() {
        // Power slider
        this.powerSlider.addEventListener('input', (e) => {
            this.updatePowerDisplay(e.target.value);
        });

        this.powerSlider.addEventListener('change', (e) => {
            this.setPowerLevel(e.target.value);
        });

        // Control buttons
        this.laserOnBtn.addEventListener('click', () => {
            this.setPowerLevel(100);
        });

        this.laserOffBtn.addEventListener('click', () => {
            this.setPowerLevel(0);
        });

        // Video controls
        this.fullscreenBtn.addEventListener('click', () => {
            this.toggleFullscreen();
        });

        // Camera controls
        this.cameraSelector.addEventListener('change', (e) => {
            this.switchCameraSource(e.target.value);
        });

        // Detection settings
        this.sensitivitySlider.addEventListener('input', (e) => {
            this.detectionSensitivity = parseInt(e.target.value);
            this.sensitivityValue.textContent = this.detectionSensitivity;
            
            // Update YOLO threshold based on sensitivity
            const threshold = this.detectionSensitivity / 100;
            this.modelManager.updateYOLOConfig({ threshold: threshold });
        });

        this.showBoundingBoxesCheckbox.addEventListener('change', (e) => {
            this.showBoundingBoxes = e.target.checked;
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.setPowerLevel(0);
            } else if (e.key === ' ') {
                e.preventDefault();
                this.setPowerLevel(this.currentPower === 0 ? 100 : 0);
            }
        });
    }

    // Initialize model management
    initializeModels() {
        const modelElements = {
            modelSelector: this.modelSelector,
            modelUpload: this.modelUpload,
            modelFiles: this.modelFiles,
            modelStatusValue: this.modelStatusValue
        };
        
        this.modelManager.initialize(modelElements);
        
        // Set up model change callback
        this.modelManager.onModelChange((detectionMode, detector) => {
            this.currentDetectionMode = detectionMode;
            this.yoloDetector = detector;
            
            console.log(`Switched to ${detectionMode} detection mode`);
            
            // Reset weed count when switching models
            this.weedCount = 0;
            this.weedCountElement.textContent = this.weedCount;
        });
        
        // Set up status change callback
        this.modelManager.onStatusChange((message, type) => {
            if (type === 'error') {
                this.showNotification(`Model Error: ${message}`, 'error');
            } else if (type === 'loaded') {
                this.showNotification(message, 'success');
            }
        });
    }

    updatePowerDisplay(power) {
        this.currentPower = parseInt(power);
        this.powerValue.textContent = this.currentPower;
        this.powerSlider.value = this.currentPower;
        
        // Update slider fill
        const fillPercentage = (this.currentPower / 100) * 100;
        this.sliderFill.style.width = `${fillPercentage}%`;
        
        // Update laser status
        this.updateLaserStatus();
    }

    updateLaserStatus() {
        const statusDot = this.laserStatus.querySelector('.status-dot');
        const statusText = this.laserStatus.querySelector('span');
        
        if (this.currentPower > 0) {
            statusDot.classList.remove('laser-off');
            statusText.textContent = `Laser ON (${this.currentPower}%)`;
        } else {
            statusDot.classList.add('laser-off');
            statusText.textContent = 'Laser OFF';
        }
    }

    async setPowerLevel(power) {
        try {
            const response = await fetch(`${this.esp8266Url}/set?val=${power}`, {
                method: 'GET',
                timeout: 5000
            });

            if (response.ok) {
                this.updatePowerDisplay(power);
                this.showNotification(`Power set to ${power}%`, 'success');
            } else {
                throw new Error('Failed to set power level');
            }
        } catch (error) {
            console.error('Error setting power level:', error);
            this.showNotification('Failed to set power level', 'error');
        }
    }

    async checkConnection() {
        try {
            const response = await fetch(`${this.esp8266Url}/status`, {
                method: 'GET',
                timeout: 3000
            });

            if (response.ok) {
                if (!this.isConnected) {
                    this.isConnected = true;
                    this.updateConnectionStatus();
                }
            } else {
                throw new Error('Connection failed');
            }
        } catch (error) {
            if (this.isConnected) {
                this.isConnected = false;
                this.updateConnectionStatus();
            }
        }
    }

    updateConnectionStatus() {
        const statusDot = this.connectionStatus.querySelector('.status-dot');
        const statusText = this.connectionStatus.querySelector('span');
        
        if (this.isConnected) {
            statusDot.style.background = 'var(--accent-green)';
            statusDot.style.boxShadow = '0 0 10px var(--accent-green)';
            statusText.textContent = 'Connected';
        } else {
            statusDot.style.background = 'var(--accent-red)';
            statusDot.style.boxShadow = '0 0 10px var(--accent-red)';
            statusText.textContent = 'Disconnected';
        }
    }

    startConnectionCheck() {
        // Check connection immediately
        this.checkConnection();
        
        // Set up periodic connection check
        this.connectionCheckInterval = setInterval(() => {
            this.checkConnection();
        }, 5000); // Check every 5 seconds
    }

    initializeVideoStream() {
        // Try to detect Raspberry Pi IP and set up video stream
        this.detectVideoStream();
        
        // Set up periodic video stream check
        setInterval(() => {
            this.checkVideoStream();
        }, 10000); // Check every 10 seconds
    }

    async detectVideoStream() {
        // Common IP addresses for Raspberry Pi on the ESP8266 network
        const possibleIPs = [
            '192.168.4.2',
            '192.168.4.3',
            '192.168.4.4',
            '192.168.4.5'
        ];

        for (const ip of possibleIPs) {
            try {
                const streamUrl = `http://${ip}:8080/stream`;
                const response = await fetch(streamUrl, { 
                    method: 'HEAD',
                    timeout: 2000 
                });
                
                if (response.ok) {
                    this.videoStreamUrl = streamUrl;
                    this.startVideoStream();
                    return;
                }
            } catch (error) {
                // Continue to next IP
            }
        }
        
        // If no stream found, show placeholder
        this.showVideoPlaceholder();
    }

    startVideoStream() {
        if (this.videoStreamUrl) {
            this.videoStream.src = this.videoStreamUrl;
            this.videoStream.style.display = 'block';
            this.videoFeed.style.display = 'none';
            
            this.videoStream.onload = () => {
                this.showNotification('Video stream connected', 'success');
            };
            
            this.videoStream.onerror = () => {
                this.showVideoPlaceholder();
            };
        }
    }

    showVideoPlaceholder() {
        this.videoStream.style.display = 'none';
        this.videoFeed.style.display = 'flex';
        this.videoFeed.innerHTML = `
            <div class="video-placeholder">
                <div class="loading-spinner"></div>
                <p>Searching for video stream...</p>
                <small>Make sure Raspberry Pi is connected and streaming</small>
            </div>
        `;
    }

    checkVideoStream() {
        if (this.videoStreamUrl) {
            // Check if video stream is still available
            const testImg = new Image();
            testImg.onload = () => {
                // Stream is working
            };
            testImg.onerror = () => {
                this.showVideoPlaceholder();
                this.detectVideoStream();
            };
            testImg.src = this.videoStreamUrl + '?t=' + Date.now();
        }
    }

    toggleFullscreen() {
        const videoContainer = document.querySelector('.video-wrapper');
        
        if (!document.fullscreenElement) {
            videoContainer.requestFullscreen().catch(err => {
                console.error('Error attempting to enable fullscreen:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }

    startClock() {
        this.updateClock();
        setInterval(() => {
            this.updateClock();
        }, 1000);
    }

    updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        this.timestamp.textContent = timeString;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            backdrop-filter: blur(10px);
        `;
        
        // Set background based on type
        switch (type) {
            case 'success':
                notification.style.background = 'rgba(0, 255, 136, 0.2)';
                notification.style.border = '1px solid var(--accent-green)';
                break;
            case 'error':
                notification.style.background = 'rgba(255, 68, 68, 0.2)';
                notification.style.border = '1px solid var(--accent-red)';
                break;
            default:
                notification.style.background = 'rgba(255, 255, 255, 0.1)';
                notification.style.border = '1px solid var(--border-glass)';
        }
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Remove after delay
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Camera switching functionality
    async switchCameraSource(source) {
        this.cameraSource = source;
        
        // Stop any existing processing
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        // Hide all video elements
        this.videoStream.style.display = 'none';
        this.laptopVideo.style.display = 'none';
        this.processCanvas.style.display = 'none';
        
        if (source === 'raspberry') {
            this.detectionSettings.style.display = 'none';
            this.initializeVideoStream();
        } else if (source === 'laptop') {
            this.detectionSettings.style.display = 'block';
            await this.initializeLaptopCamera();
        }
    }

    // Initialize laptop camera
    async initializeLaptopCamera() {
        try {
            // Stop existing stream if any
            if (this.laptopStream) {
                this.laptopStream.getTracks().forEach(track => track.stop());
            }
            
            // Request camera access
            this.laptopStream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Prefer back camera on mobile
                },
                audio: false
            });
            
            // Set up video element
            this.laptopVideo.srcObject = this.laptopStream;
            this.laptopVideo.style.display = 'block';
            this.videoFeed.style.display = 'none';
            
            // Set up canvas for processing
            this.processCanvas.width = 1280;
            this.processCanvas.height = 720;
            this.processCanvas.style.display = 'block';
            this.processCanvas.style.position = 'absolute';
            this.processCanvas.style.top = '0';
            this.processCanvas.style.left = '0';
            this.processCanvas.style.pointerEvents = 'none';
            
            // Start computer vision processing
            this.laptopVideo.onloadedmetadata = () => {
                this.startComputerVision();
            };
            
            this.showNotification('Laptop camera connected', 'success');
            
        } catch (error) {
            console.error('Error accessing laptop camera:', error);
            this.showNotification('Failed to access laptop camera', 'error');
            this.showVideoPlaceholder();
        }
    }

    // Computer vision processing
    startComputerVision() {
        this.isProcessing = true;
        this.weedCount = 0; // Reset counter
        
        // Process frames every 100ms
        this.processingInterval = setInterval(() => {
            this.processFrame();
        }, 100);
    }

    async processFrame() {
        if (!this.isProcessing || !this.laptopVideo.videoWidth) return;
        
        const canvas = this.processCanvas;
        const ctx = canvas.getContext('2d');
        const video = this.laptopVideo;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        let detectedWeeds = [];
        
        try {
            if (this.currentDetectionMode === 'yolo' && this.yoloDetector) {
                // Use YOLO detection
                detectedWeeds = await this.yoloDetector.detect(video);
            } else {
                // Use simple color detection
                detectedWeeds = this.detectGreenAreasSimple(video);
            }
            
            // Draw bounding boxes if enabled
            if (this.showBoundingBoxes && detectedWeeds.length > 0) {
                this.drawBoundingBoxes(ctx, detectedWeeds, canvas.width, canvas.height, video.videoWidth, video.videoHeight);
            }
            
            // Update weed count
            this.weedCount = detectedWeeds.length;
            this.weedCountElement.textContent = this.weedCount;
            
            // Add visual feedback if weeds detected
            if (detectedWeeds.length > 0) {
                this.weedCountElement.style.animation = 'none';
                setTimeout(() => {
                    this.weedCountElement.style.animation = 'pulse 1s ease-in-out';
                }, 10);
            }
            
        } catch (error) {
            console.error('Error in processFrame:', error);
            // Fall back to simple detection on error
            if (this.currentDetectionMode === 'yolo') {
                detectedWeeds = this.detectGreenAreasSimple(video);
                this.drawBoundingBoxes(ctx, detectedWeeds, canvas.width, canvas.height, video.videoWidth, video.videoHeight);
            }
        }
    }

    // Simple green area detection for video element
    detectGreenAreasSimple(video) {
        // Create a temporary canvas for image processing
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        
        // Draw video frame
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        
        // Get image data for processing
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        return this.detectGreenAreas(data, tempCanvas.width, tempCanvas.height);
    }

    // Simple green area detection from image data
    detectGreenAreas(imageData, width, height) {
        const detectedAreas = [];
        const minSize = 20; // Minimum area size
        const sensitivity = this.detectionSensitivity / 100;
        
        // Grid-based detection for performance
        const gridSize = 20;
        
        for (let y = 0; y < height - gridSize; y += gridSize) {
            for (let x = 0; x < width - gridSize; x += gridSize) {
                let greenPixels = 0;
                let totalPixels = 0;
                
                // Check pixels in this grid cell
                for (let dy = 0; dy < gridSize; dy++) {
                    for (let dx = 0; dx < gridSize; dx++) {
                        const index = ((y + dy) * width + (x + dx)) * 4;
                        const r = imageData[index];
                        const g = imageData[index + 1];
                        const b = imageData[index + 2];
                        
                        // Simple green detection
                        if (g > r * 1.2 && g > b * 1.2 && g > 50) {
                            greenPixels++;
                        }
                        totalPixels++;
                    }
                }
                
                // If enough green pixels detected
                const greenRatio = greenPixels / totalPixels;
                if (greenRatio > sensitivity * 0.3) {
                    detectedAreas.push({
                        x: x,
                        y: y,
                        width: gridSize,
                        height: gridSize,
                        confidence: greenRatio
                    });
                }
            }
        }
        
        return detectedAreas;
    }

    // Draw bounding boxes on canvas
    drawBoundingBoxes(ctx, detectedAreas, canvasWidth, canvasHeight, videoWidth, videoHeight) {
        const scaleX = canvasWidth / videoWidth;
        const scaleY = canvasHeight / videoHeight;
        
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;
        ctx.font = '14px Inter';
        ctx.fillStyle = '#00ff88';
        
        detectedAreas.forEach((area, index) => {
            const x = area.x * scaleX;
            const y = area.y * scaleY;
            const width = area.width * scaleX;
            const height = area.height * scaleY;
            
            // Draw bounding box
            ctx.strokeRect(x, y, width, height);
            
            // Draw label (different for YOLO vs simple detection)
            let label;
            if (area.className && area.confidence) {
                // YOLO detection format
                label = `${area.className} (${Math.round(area.confidence * 100)}%)`;
            } else {
                // Simple detection format
                label = `Weed ${index + 1}`;
            }
            
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x, y - 20, textWidth + 8, 20);
            ctx.fillStyle = '#000';
            ctx.fillText(label, x + 4, y - 6);
            ctx.fillStyle = '#00ff88';
        });
    }

    // Simulate weed detection for demo purposes (for Raspberry Pi mode)
    simulateWeedDetection() {
        setInterval(() => {
            if (this.cameraSource === 'raspberry' && Math.random() > 0.7) { // 30% chance to detect a weed
                this.weedCount++;
                this.weedCountElement.textContent = this.weedCount;
                
                // Add some visual feedback
                this.weedCountElement.style.animation = 'none';
                setTimeout(() => {
                    this.weedCountElement.style.animation = 'pulse 1s ease-in-out';
                }, 10);
            }
        }, 2000); // Check every 2 seconds
    }

    // Cleanup when page is unloaded
    destroy() {
        if (this.connectionCheckInterval) {
            clearInterval(this.connectionCheckInterval);
        }
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
        }
        if (this.laptopStream) {
            this.laptopStream.getTracks().forEach(track => track.stop());
        }
        if (this.modelManager) {
            this.modelManager.dispose();
        }
        this.isProcessing = false;
    }
}

// Initialize the controller when page loads
document.addEventListener('DOMContentLoaded', () => {
    const controller = new LaserWeedController();
    
    // Start demo weed detection simulation
    controller.simulateWeedDetection();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        controller.destroy();
    });
});

// Add some additional CSS for notifications via JavaScript
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .notification {
        font-family: inherit;
    }
`;
document.head.appendChild(style);