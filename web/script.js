class LaserWeedController {
    constructor() {
        this.currentPower = 0;
        this.isConnected = false;
        this.weedCount = 0;
        this.videoStreamUrl = '';
        this.esp8266Url = 'http://192.168.4.1';
        this.connectionCheckInterval = null;
        this.lastUpdateTime = Date.now();
        
        this.initializeElements();
        this.setupEventListeners();
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
        this.fullscreenBtn = document.getElementById('fullscreenBtn');
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

    // Simulate weed detection for demo purposes
    simulateWeedDetection() {
        setInterval(() => {
            if (Math.random() > 0.7) { // 30% chance to detect a weed
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