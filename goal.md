Of course. This is an excellent project that combines hardware, software, and machine learning. A Product Requirements Document (PRD) is the perfect way to organize the development effort.

Here is a comprehensive PRD based on the code, chat logs, and audio clips you provided. It's designed to be a clear blueprint for your team.

---

## **Product Requirements Document: Project "LaserWeed"**

**Version:** 1.0
**Date:** July 11, 2025
**Author:** Daur

### 1. Introduction & Overview

Project "LaserWeed" is an integrated system designed to identify and target weeds using a web-controlled laser. The system leverages a Raspberry Pi for computer vision processing and an ESP8266 for real-time laser control.

The final product will be a single, cohesive web interface where a user can view a live video feed, see detected weeds highlighted by the system in real-time, and manually operate a laser to eliminate them. This document outlines the requirements for a functional prototype intended for a technical demonstration.

### 2. Goals & Objectives

*   **Primary Goal:** To create a functional proof-of-concept that successfully integrates a live video feed, a computer vision weed-detection model, and a web-based laser control interface into a single, user-friendly application.
*   **Secondary Goal:** To deliver a polished and impressive user experience ("The website must shine," as noted in the team discussion) for a successful demonstration.
*   **Tertiary Goal:** To build a modular system where the computer vision and hardware control components can be developed and tested independently before final integration.

### 3. Success Metrics

*   **Functionality:** The system successfully streams video, detects weeds with >80% accuracy on a test set, and fires the laser in response to user commands.
*   **Performance:** End-to-end latency from camera capture to the video displaying on the web UI is under 2 seconds.
*   **Usability:** A new user can understand and operate the entire system via the web interface in under 3 minutes without verbal instructions.
*   **Reliability:** The system can run continuously for a 15-minute demonstration without crashes or significant performance degradation.

### 4. Target Audience & User Persona

*   **Persona:** The "System Operator" (e.g., a demo presenter, a hobbyist, or a technician).
*   **Needs & Goals:**
    *   Wants an easy and quick way to set up and start the system.
    *   Needs a clear, real-time visual of what the system is seeing.
    *   Requires simple and responsive controls to operate the laser.
    *   Needs confidence that the system is working as expected through clear visual feedback.

### 5. Core Features & Requirements

#### **5.1. System Hardware & Networking**
*   **HRD-1: Components:** The system will use an ESP8266 for web serving and laser control, and a Raspberry Pi (or similar single-board computer) with a camera module for video processing.
*   **HRD-2: Wi-Fi Access Point:** The ESP8266 will create its own Wi-Fi Access Point (`WIFI_AP`).
    *   **SSID:** `LaserControlESP`
    *   **Password:** `12345678`
    *   All devices (Raspberry Pi, user's phone/laptop) will connect to this network for operation.
*   **HRD-3: Static IP:** The ESP8266 web server will be accessible at the static IP address `http://192.168.4.1`.

#### **5.2. Web Interface (The "Unified Website")**
*   **UI-1: Main Dashboard:** The web application served by the ESP8266 will be the single point of interaction for the user.
*   **UI-2: Live Video Feed:** The dashboard must embed and display a live video stream from the Raspberry Pi.
*   **UI-3: Laser Controls:** The interface must provide intuitive controls for the laser:
    *   An intensity/power slider, displaying the power as a percentage (0-100%).
    *   A dedicated "ON (100%)" button to set the laser to full power.
    *   A dedicated "OFF" button to immediately turn the laser off (set power to 0).
*   **UI-4: Real-time Updates:** Slider movements and button presses must send commands to the ESP8266 (`/set?val=...`) and update the laser's power without reloading the page (as implemented in the provided code).
*   **UI-5: CV Overlay:** The video feed must display the output of the computer vision model, with detected weeds clearly enclosed in a colored bounding box (e.g., green or red).

#### **5.3. Computer Vision (CV) System**
*   **CV-1: Video Streaming:** The Raspberry Pi will run a service (e.g., MJPEG-Streamer) to capture video from its camera and stream it over the network.
*   **CV-2: Weed Detection:** A script (likely Python with OpenCV) on the Raspberry Pi will process the video frames in real-time.
    *   For V1, detection can be based on color thresholding (detecting the color green).
    *   The script will draw bounding boxes around detected objects directly onto the video frames before they are streamed.
*   **CV-3: Demo/Contingency Mode:** As discussed, a fallback mode is required for the demo.
    *   The system must have a mode where it can play a pre-recorded video file instead of using the live camera feed.
    *   In this mode, the bounding box "detections" can be pre-scripted to appear at specific timestamps to ensure a flawless demonstration.

#### **5.4. Laser Control System**
*   **LC-1: Hardware Interface:** The ESP8266 will control the laser module via Pulse Width Modulation (PWM) on pin `D1` using `analogWrite()`.
*   **LC-2: Persistence:** The laser's last brightness setting must be saved to the ESP8266's EEPROM. When the device reboots, it must automatically load this value and set the laser to the last known intensity.

### 6. System Architecture & User Flow

#### **Architecture:**
1.  **ESP8266:** Acts as the central hub. It creates the Wi-Fi network and hosts the main web server for user control.
2.  **Raspberry Pi:** Connects to the ESP8266's Wi-Fi. It runs the CV script, processes the video, and streams the result to a specific port on its own IP address.
3.  **User Device (Laptop/Phone):** Connects to the ESP8266's Wi-Fi. The user opens a browser to the ESP8266's IP (`192.168.4.1`). The HTML page loaded from the ESP8266 will contain an `<img>` or `<video>` tag that points to the Raspberry Pi's video stream IP address.

#### **User Flow:**
1.  User powers on the ESP8266 and Raspberry Pi.
2.  User connects their laptop/phone to the "LaserControlESP" Wi-Fi network.
3.  User navigates to `http://192.168.4.1` in their web browser.
4.  The LaserWeed dashboard loads, showing the live video feed with detected weeds highlighted.
5.  User adjusts the laser power using the slider or presses the ON/OFF buttons.
6.  The laser responds instantly to the user's commands.

### 7. Out of Scope (For Version 1.0)

*   **Automatic Targeting & Firing:** The system will only *detect* weeds. The user is responsible for aiming the laser and firing.
*   **Physical Movement:** The system is assumed to be stationary. Integration with a mobile robot or gimbal is not part of V1.
*   **Advanced ML Models:** V1 will use simpler, faster detection methods (like color thresholding). Full-fledged neural network models are a future consideration.
*   **Cloud Connectivity:** The system will operate entirely offline on its own local network.

---