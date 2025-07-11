# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project "LaserWeed" is an integrated IoT system that combines hardware control, computer vision, and web interfaces to identify and target weeds using a laser. The system consists of:

- **ESP8266**: Central hub that creates Wi-Fi AP and hosts web server for laser control
- **Raspberry Pi**: Handles computer vision processing and video streaming
- **Web Interface**: Unified dashboard for live video feed and laser controls

## Architecture

### System Components
1. **ESP8266 (Primary Controller)**
   - Creates Wi-Fi Access Point: `LaserControlESP` (password: `12345678`)
   - Hosts web server at static IP: `192.168.4.1`
   - Controls laser via PWM on pin D1
   - Persists laser settings to EEPROM

2. **Raspberry Pi (Computer Vision)**
   - Connects to ESP8266's Wi-Fi network
   - Runs video capture and CV processing
   - Streams processed video with detection overlays
   - Fallback mode: plays pre-recorded video for demos

3. **Web Interface (Unified Dashboard)**
   - Embedded live video feed from Raspberry Pi
   - Laser control interface (slider + ON/OFF buttons)
   - Real-time updates via `/set?val=...` endpoints
   - CV overlay showing detected weeds in bounding boxes

### Network Flow
```
User Device → ESP8266 Wi-Fi AP → Web Dashboard
                ↓
ESP8266 ← Laser Control Commands
                ↓
Raspberry Pi → Video Stream → Web Dashboard
```

## Development Structure

### Expected File Organization
```
/esp8266/           # ESP8266 Arduino code
  - main.ino        # Main ESP8266 sketch
  - web_server.cpp  # Web server implementation
  - laser_control.cpp # Laser PWM control
  
/raspberry_pi/      # Raspberry Pi Python code
  - cv_processor.py # Computer vision processing
  - video_stream.py # Video streaming service
  - demo_mode.py    # Pre-recorded video fallback
  
/web/              # Web interface assets
  - index.html     # Main dashboard
  - style.css      # UI styling
  - script.js      # Client-side controls
  
/docs/             # Documentation
  - setup.md       # Hardware setup instructions
  - api.md         # API endpoints documentation
```

## Key Technical Requirements

### Performance Targets
- End-to-end video latency: <2 seconds
- Weed detection accuracy: >80%
- Continuous operation: 15+ minutes without crashes

### Hardware Specifications
- ESP8266: Wi-Fi AP mode, PWM laser control
- Raspberry Pi: Camera module, OpenCV processing
- Laser: PWM-controlled power (0-100%)

### Computer Vision
- V1: Color thresholding for green detection
- Real-time bounding box overlay on video stream
- Demo mode with pre-scripted detections

## Development Commands

### ESP8266 Development
```bash
# Flash ESP8266 (typical Arduino IDE workflow)
arduino-cli compile --fqbn esp8266:esp8266:nodemcuv2 esp8266/
arduino-cli upload --fqbn esp8266:esp8266:nodemcuv2 esp8266/
```

### Raspberry Pi Development
```bash
# Install dependencies
pip install opencv-python numpy flask

# Run CV processor
python raspberry_pi/cv_processor.py

# Run video streaming service
python raspberry_pi/video_stream.py

# Demo mode
python raspberry_pi/demo_mode.py
```

### Testing
```bash
# Test ESP8266 web server
curl http://192.168.4.1/

# Test laser control
curl "http://192.168.4.1/set?val=50"  # Set 50% power
curl "http://192.168.4.1/set?val=0"   # Turn off

# Test video stream
curl http://[raspberry_pi_ip]:8080/stream
```

## Important Notes

### Security Considerations
- This is a hardware demonstration project with local Wi-Fi network
- No external internet connectivity required
- Laser safety protocols must be followed during operation

### Development Workflow
1. Develop ESP8266 and Raspberry Pi components independently
2. Test each component in isolation before integration
3. Use demo mode for reliable demonstration scenarios
4. Ensure EEPROM persistence for laser settings across reboots

### Demo Requirements
- System must be operational within 3 minutes of power-on
- Web interface must be intuitive for new users
- Fallback demo mode available for critical presentations
- Visual feedback for all user actions (laser power, detection status)