# 🎯 LaserWeed Control System

A complete IoT system for precision weed detection and laser targeting using computer vision and web-based control interface.

![LaserWeed System](https://img.shields.io/badge/Status-Active-green) ![YOLOv7](https://img.shields.io/badge/AI-YOLOv7-blue) ![ESP8266](https://img.shields.io/badge/Hardware-ESP8266-orange) ![TensorFlow.js](https://img.shields.io/badge/ML-TensorFlow.js-yellow)

## 🌟 Features

- **Real-time Weed Detection** with YOLOv7 AI models
- **Dual Camera Support** (Raspberry Pi + Laptop webcam)
- **Precision Laser Control** via ESP8266
- **Web-based Interface** with glassmorphism design
- **Automatic Model Conversion** from .ckpt to TensorFlow.js
- **Agricultural Optimization** with crop/weed specific tuning

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │    ESP8266      │    │  Raspberry Pi   │
│                 │    │                 │    │                 │
│ • UI Dashboard  │◄──►│ • Wi-Fi AP      │    │ • Camera        │
│ • YOLO Models   │    │ • Laser Control │    │ • CV Processing │
│ • TensorFlow.js │    │ • Web Server    │    │ • Video Stream  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Laser Module  │
                    │                 │
                    │ • PWM Control   │
                    │ • Safety Lock   │
                    └─────────────────┘
```

## 🚀 Quick Start

### 1. **Hardware Setup**

#### ESP8266 Configuration
```cpp
// Arduino IDE setup for ESP8266
// 1. Install ESP8266 board package
// 2. Select NodeMCU or ESP8266 board
// 3. Upload web server code with Wi-Fi AP configuration

#define WIFI_SSID "LaserControlESP"
#define WIFI_PASSWORD "12345678"
#define LASER_PIN D1  // PWM pin for laser control
```

#### Raspberry Pi Setup
```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip
pip3 install opencv-python numpy flask

# Install MJPEG streamer
sudo apt install mjpg-streamer

# Start video stream
mjpg_streamer -i "input_raspicam.so -fps 30 -q 50 -x 640 -y 480" -o "output_http.so -p 8080"
```

### 2. **Web Interface Setup**

#### Clone and Navigate
```bash
git clone <your-repo>
cd weedzap
```

#### Install Dependencies (Optional - for model conversion)
```bash
# Python dependencies for model conversion
pip install torch tensorflowjs ultralytics tensorflow

# Node.js for local development server (optional)
npm install -g http-server
```

#### Start Local Development Server
```bash
# Option 1: Python
cd web
python3 -m http.server 8080

# Option 2: Node.js
npx http-server web -p 8080

# Option 3: PHP
php -S localhost:8080 -t web
```

### 3. **Model Conversion**

#### Convert Your YOLOv7 Checkpoints
```bash
cd models

# Single model conversion
python convert_model.py \
  --input custom-models/CropOrWeed2_640px_yolov7-tiny_epoch=37_lr=_batch=48_val_loss=11.115_map=0.592.ckpt \
  --output custom-models/croporweed2-tiny \
  --quantize

# Batch conversion (all .ckpt files)
python batch_convert.py --quantize --generate-index
```

#### Supported Model Formats
- ✅ **YOLOv7 Checkpoints** (.ckpt) - Auto-detected and converted
- ✅ **PyTorch Models** (.pt) - Direct conversion
- ✅ **ONNX Models** (.onnx) - Ready for conversion
- ✅ **TensorFlow.js** (.json + .bin) - Ready to use

## 🌐 Network Architecture

### ESP8266 Wi-Fi Access Point Mode
```
LaserControlESP Network (192.168.4.0/24)
├── ESP8266 Web Server     → 192.168.4.1
├── Raspberry Pi           → 192.168.4.2
├── Your Laptop/Phone      → 192.168.4.x
└── Additional Devices     → 192.168.4.x
```

### API Endpoints

#### ESP8266 Laser Control
```http
GET http://192.168.4.1/
# Main dashboard

GET http://192.168.4.1/set?val=50
# Set laser power to 50%

GET http://192.168.4.1/status
# Get system status
```

#### Raspberry Pi Video Stream
```http
GET http://192.168.4.2:8080/stream
# MJPEG video stream

GET http://192.168.4.2:8080/?action=snapshot
# Single frame capture
```

## 🤖 AI Model Integration

### Automatic Model Detection
The system automatically scans for YOLOv7 checkpoint files and displays them in the interface:

```javascript
// Current detected model:
"YOLOV7-tiny CropOrWeed2 (640px) - ep37, mAP: 59.2%, loss: 11.12"
```

### Model Performance Tiers
- **Tiny**: Fast inference (~50+ FPS) - Real-time operation
- **Base**: Balanced performance (~30 FPS) - Good accuracy
- **X**: High accuracy (~20 FPS) - Precision detection
- **E6**: Premium accuracy (~15 FPS) - Maximum precision

### Supported Detection Classes
```json
{
  "classes": [
    "broadleaf_weed",
    "grass_weed", 
    "crop_plant",
    "soil"
  ]
}
```

## 🎮 Web Interface Usage

### 1. **Connect to Network**
```
Network: LaserControlESP
Password: 12345678
URL: http://192.168.4.1
```

### 2. **Camera Selection**
- **Raspberry Pi**: External camera with CV processing
- **Laptop Camera**: WebRTC with client-side AI processing

### 3. **Model Selection**
- **Simple Color Detection**: Basic green area detection
- **YOLOv7 Models**: AI-powered crop/weed classification
- **Custom Models**: Upload your own trained models

### 4. **Laser Control**
- **Power Slider**: 0-100% adjustable power
- **Quick Controls**: ON (100%) / OFF buttons
- **Safety Features**: Emergency stop (ESC key)

### 5. **Keyboard Shortcuts**
- `ESC`: Emergency laser shutdown
- `SPACE`: Toggle laser on/off
- `F`: Toggle fullscreen video

## 🔧 Development Guide

### File Structure
```
weedzap/
├── web/                    # Web interface
│   ├── index.html         # Main dashboard
│   ├── style.css          # Glassmorphism design
│   ├── script.js          # Main application logic
│   └── js/
│       ├── yolo-detector.js    # AI model inference
│       └── model-manager.js    # Model loading/switching
├── models/                # AI models directory
│   ├── README.md          # Model documentation
│   ├── convert_model.py   # Single model converter
│   ├── batch_convert.py   # Batch converter
│   ├── custom-models/     # Your .ckpt files
│   └── yolov5s-weed/     # Pre-trained models
├── esp8266/               # ESP8266 Arduino code
├── raspberry_pi/          # Raspberry Pi Python code
└── docs/                  # Documentation
```

### Adding New Models

#### 1. Place Model Files
```bash
# Copy your .ckpt file
cp your-model.ckpt models/custom-models/

# Or download pre-trained models
wget https://github.com/your-repo/model.ckpt -O models/custom-models/
```

#### 2. Convert to TensorFlow.js
```bash
cd models
python convert_model.py --input custom-models/your-model.ckpt --output custom-models/your-model --quantize
```

#### 3. Automatic Detection
The web interface will automatically detect and offer conversion options for new .ckpt files.

### Custom ESP8266 Code
```cpp
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <EEPROM.h>

ESP8266WebServer server(80);

void setup() {
  // Initialize Wi-Fi AP
  WiFi.softAP("LaserControlESP", "12345678");
  
  // Setup laser control pin
  pinMode(D1, OUTPUT);
  analogWriteRange(100); // 0-100% power range
  
  // Load last power setting from EEPROM
  EEPROM.begin(512);
  int lastPower = EEPROM.read(0);
  analogWrite(D1, lastPower);
  
  // Setup web routes
  server.on("/", handleRoot);
  server.on("/set", handleSetPower);
  server.on("/status", handleStatus);
  
  server.begin();
}

void loop() {
  server.handleClient();
}
```

## 🔒 Safety Features

### Hardware Safety
- **Emergency Stop**: ESC key or /set?val=0 endpoint
- **Power Persistence**: Last setting saved to EEPROM
- **PWM Control**: Smooth power adjustment (0-100%)
- **Status Monitoring**: Real-time connection/laser status

### Software Safety
- **Input Validation**: All power values clamped to 0-100%
- **Connection Timeout**: Auto-disconnect on network issues
- **Error Recovery**: Automatic fallback to safe states
- **Visual Feedback**: Clear status indicators for all operations

### Operational Safety
```
⚠️  LASER SAFETY WARNING ⚠️
- Always wear appropriate laser safety equipment
- Never point laser at people, animals, or reflective surfaces  
- Ensure proper ventilation in operating area
- Follow local regulations for laser device operation
- Keep emergency stop accessible at all times
```

## 📊 Performance Metrics

### Model Performance (640px input)
| Model | mAP | Inference Time | Memory Usage | Best For |
|-------|-----|----------------|--------------|----------|
| YOLOv7-tiny | 59.2% | ~20ms | ~15MB | Real-time detection |
| YOLOv7-base | ~65% | ~35ms | ~30MB | Balanced performance |
| YOLOv7-x | ~70% | ~50ms | ~60MB | High accuracy |
| YOLOv7-e6 | ~75% | ~80ms | ~120MB | Maximum precision |

### System Requirements
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+
- **Memory**: 4GB RAM recommended for large models
- **Network**: 2.4GHz Wi-Fi for ESP8266 compatibility
- **WebGL**: Required for TensorFlow.js acceleration

## 🐛 Troubleshooting

### Common Issues

#### "Model not loading"
```bash
# Check model file paths
ls -la models/custom-models/
ls -la web/models/

# Verify TensorFlow.js conversion
cd models
python convert_model.py --validate -i your-model.ckpt -o output-dir
```

#### "Camera not detected"
```javascript
// Check browser permissions
navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => console.log('Camera OK'))
  .catch(err => console.error('Camera error:', err));
```

#### "ESP8266 connection failed"
```bash
# Check Wi-Fi connection
ping 192.168.4.1

# Verify ESP8266 is broadcasting
iwlist scan | grep LaserControlESP
```

#### "Slow model inference"
- Use quantized models (--quantize flag)
- Reduce input resolution (640px → 416px)
- Enable WebGL acceleration in browser
- Close other browser tabs

### Debug Commands
```bash
# Check model conversion
python models/convert_model.py --validate

# Test web server locally  
cd web && python3 -m http.server 8080

# Monitor ESP8266 serial output
screen /dev/ttyUSB0 115200

# Check Raspberry Pi video stream
curl -I http://192.168.4.2:8080/stream
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-model`)
3. Test with your hardware setup
4. Submit pull request with hardware compatibility notes

### Model Contributions
- Share trained YOLOv7 models for different crop types
- Include training dataset information and performance metrics
- Provide conversion instructions and optimal settings

### Hardware Contributions
- ESP8266 code improvements and safety features
- Raspberry Pi optimization and camera drivers
- Laser control hardware designs and safety systems

## 📚 Additional Resources

### Documentation
- [Model Training Guide](docs/training.md)
- [Hardware Assembly](docs/hardware.md)
- [API Reference](docs/api.md)
- [Safety Guidelines](docs/safety.md)

### Community
- [GitHub Issues](https://github.com/your-repo/issues) - Bug reports and feature requests
- [Discussions](https://github.com/your-repo/discussions) - Community support
- [Wiki](https://github.com/your-repo/wiki) - Extended documentation

### Citations
```bibtex
@misc{laserweed2025,
  title={LaserWeed: AI-Powered Precision Weed Control System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/weedzap}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Disclaimer

This project involves laser technology. Users are responsible for:
- Compliance with local laws and regulations
- Proper safety equipment and procedures  
- Risk assessment of their specific use case
- Any damage or injury resulting from use

**Use at your own risk. The authors assume no liability for any consequences of using this system.**

---

## 🎯 Quick Command Reference

```bash
# Start everything
cd weedzap
python3 -m http.server 8080 -d web  # Web interface
python models/batch_convert.py      # Convert models
ping 192.168.4.1                   # Test ESP8266

# Convert your model
python models/convert_model.py -i models/custom-models/your-model.ckpt -o models/custom-models/converted --quantize

# Access system
# Wi-Fi: LaserControlESP (password: 12345678)
# URL: http://192.168.4.1
```

**Happy weed zapping! 🌱⚡**