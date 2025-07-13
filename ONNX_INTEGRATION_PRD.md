# Product Requirements Document: ONNX Model Integration for LaserWeed System

## Document Information
- **Project**: LaserWeed Control System
- **Feature**: ONNX Model Integration & Browser Testing
- **Version**: 1.0
- **Date**: July 13, 2025
- **Status**: In Progress

## Executive Summary

The LaserWeed Control System requires complete integration of ONNX models for real-time crop and weed detection in web browsers. This PRD outlines the requirements to enable ONNX models to load, run inference, and provide accurate detection results while ensuring seamless switching between YOLO and ONNX detection modes.

## Problem Statement

Currently, the LaserWeed system has ONNX models that are detected and appear to load successfully, but fail during the inference post-processing stage with tensor data structure errors. The system needs robust ONNX model support with proper error handling, performance optimization, and browser compatibility.

## Goals & Objectives

### Primary Goals
1. **Complete ONNX Integration**: Enable full ONNX model functionality in the web interface
2. **Reliable Inference**: Fix post-processing errors and ensure accurate detection results
3. **Seamless Model Switching**: Allow users to switch between YOLO and ONNX models without issues
4. **Browser Compatibility**: Ensure ONNX models work across all major browsers

### Success Metrics
- ONNX models load without errors in 100% of test cases
- Inference completes successfully with detection results displayed
- Model switching works between all available models (Simple, YOLO, ONNX)
- Performance meets or exceeds 20 FPS on modern hardware
- Zero critical errors in browser console during normal operation

## User Stories

### As a LaserWeed System User
- **US1**: I want to select ONNX models from the dropdown and have them load successfully
- **US2**: I want to see real-time weed detection using ONNX models with bounding boxes
- **US3**: I want to switch between different detection models (Simple, YOLO, ONNX) seamlessly
- **US4**: I want the system to automatically select the best available model
- **US5**: I want clear feedback when models are loading, loaded, or encountering errors

### As a Developer
- **US6**: I want comprehensive debugging information for ONNX tensor processing
- **US7**: I want modular code that separates ONNX and YOLO detection logic
- **US8**: I want robust error handling that doesn't crash the application

## Technical Requirements

### Functional Requirements

#### FR1: ONNX Model Loading
- **FR1.1**: System shall load ONNX models from `/web/models/custom-models/` directory
- **FR1.2**: System shall parse model configuration from corresponding JSON files
- **FR1.3**: System shall validate model compatibility before loading
- **FR1.4**: System shall provide loading progress feedback to users

#### FR2: ONNX Inference Engine
- **FR2.1**: System shall preprocess camera input to match ONNX model requirements
- **FR2.2**: System shall execute ONNX model inference using ONNX.js runtime
- **FR2.3**: System shall post-process model outputs to extract bounding boxes and classifications
- **FR2.4**: System shall handle different ONNX output tensor formats (YOLOv7, YOLOv5, custom)

#### FR3: Detection Results Processing
- **FR3.1**: System shall apply confidence thresholding to filter low-confidence detections
- **FR3.2**: System shall apply Non-Maximum Suppression (NMS) to remove duplicate detections
- **FR3.3**: System shall scale detection coordinates to match original image dimensions
- **FR3.4**: System shall classify detections into predefined categories (crop, weed, etc.)

#### FR4: User Interface Integration
- **FR4.1**: System shall display ONNX models in the model selection dropdown
- **FR4.2**: System shall show detection bounding boxes on video feed
- **FR4.3**: System shall update weed count based on ONNX detection results
- **FR4.4**: System shall provide model status information (loading, loaded, error)

#### FR5: Model Management
- **FR5.1**: System shall support switching between detection modes without page reload
- **FR5.2**: System shall maintain separate configurations for each model type
- **FR5.3**: System shall automatically select recommended models when available
- **FR5.4**: System shall handle model loading failures gracefully

### Non-Functional Requirements

#### NFR1: Performance
- **NFR1.1**: ONNX inference shall complete within 50ms per frame on modern hardware
- **NFR1.2**: System shall maintain 20+ FPS during real-time detection
- **NFR1.3**: Memory usage shall not exceed 500MB during normal operation
- **NFR1.4**: Model loading shall complete within 10 seconds

#### NFR2: Reliability
- **NFR2.1**: System shall handle network interruptions without crashing
- **NFR2.2**: Detection errors shall not prevent continued operation
- **NFR2.3**: System shall recover from model loading failures
- **NFR2.4**: Browser refresh shall restore previous model selection

#### NFR3: Compatibility
- **NFR3.1**: System shall work on Chrome 90+, Firefox 88+, Safari 14+
- **NFR3.2**: System shall support both desktop and mobile browsers
- **NFR3.3**: System shall work with WebGL and CPU inference backends
- **NFR3.4**: System shall handle different camera resolutions and formats

#### NFR4: Usability
- **NFR4.1**: Model selection shall be intuitive with clear labeling
- **NFR4.2**: Error messages shall be user-friendly and actionable
- **NFR4.3**: Detection visualization shall be clear and responsive
- **NFR4.4**: System shall provide adequate feedback for all user actions

## Current State Analysis

### Completed Features ✅
- ONNX model files are present and accessible
- Model configuration JSON files are correctly formatted
- ONNXDetector class implementation with proper structure
- Model type routing (ONNX models go to ONNXDetector)
- Basic ONNX model loading functionality
- Model selection UI integration
- Browser MCP testing infrastructure

### Issues Identified 🚨

#### Critical Issues
1. **Tensor Post-Processing Error**: `TypeError: Cannot read properties of undefined (reading 'data')` in line 202 of `onnx-detector.js`
2. **Browser Cache Issues**: Updated code not loading despite hard refresh attempts
3. **Output Tensor Format Mismatch**: ONNX output format not matching expected structure

#### High Priority Issues
1. **Debug Output Visibility**: Debug console.log statements not appearing in browser
2. **Inference Pipeline Validation**: Need to verify complete inference flow
3. **Error Recovery**: System should handle ONNX errors gracefully

#### Medium Priority Issues
1. **Performance Optimization**: ONNX inference speed optimization needed
2. **Memory Management**: Proper disposal of ONNX sessions
3. **Model Validation**: Better validation of ONNX model compatibility

### Dependencies
- **ONNX.js Runtime**: Web-compatible ONNX inference engine
- **Browser WebGL Support**: For accelerated inference
- **Model Files**: Pre-trained ONNX models and configurations
- **Camera API**: WebRTC for video input

## Implementation Plan

### Phase 1: Critical Bug Fixes (Priority: High)
**Timeline**: 2-3 days

#### Task 1.1: Resolve Browser Cache Issues
- Force browser to load latest code versions
- Implement cache-busting techniques
- Verify debug output visibility

#### Task 1.2: Fix Tensor Post-Processing
- Analyze actual ONNX output tensor structure
- Update postProcess method to handle correct tensor format
- Add comprehensive tensor format debugging

#### Task 1.3: Validate ONNX Inference Pipeline
- Test complete flow from image input to detection output
- Verify tensor preprocessing and postprocessing
- Ensure coordinate scaling works correctly

### Phase 2: Robustness & Error Handling (Priority: High)
**Timeline**: 2-3 days

#### Task 2.1: Implement Robust Error Handling
- Add try-catch blocks around ONNX operations
- Provide fallback to simple detection on ONNX failure
- Display user-friendly error messages

#### Task 2.2: Memory Management
- Implement proper ONNX session disposal
- Add memory usage monitoring
- Prevent memory leaks during model switching

#### Task 2.3: Model Validation
- Validate ONNX model compatibility before loading
- Check tensor input/output specifications
- Provide clear feedback for incompatible models

### Phase 3: Performance & Optimization (Priority: Medium)
**Timeline**: 3-4 days

#### Task 3.1: Performance Optimization
- Optimize tensor preprocessing operations
- Enable WebGL acceleration where possible
- Reduce memory allocations in hot paths

#### Task 3.2: Multi-Model Support
- Support different ONNX model architectures
- Handle varying output formats automatically
- Add model-specific configuration options

#### Task 3.3: Advanced Features
- Implement model warmup for faster first inference
- Add performance metrics and monitoring
- Optimize for mobile browser performance

### Phase 4: Testing & Documentation (Priority: Medium)
**Timeline**: 2-3 days

#### Task 4.1: Comprehensive Testing
- Browser compatibility testing (Chrome, Firefox, Safari)
- Performance testing on different hardware
- Stress testing with continuous operation

#### Task 4.2: Documentation Updates
- Update user documentation with ONNX instructions
- Create troubleshooting guide for common issues
- Document model conversion and customization process

#### Task 4.3: Code Quality
- Add comprehensive code comments
- Implement unit tests for critical functions
- Review and refactor code for maintainability

## Testing Strategy

### Unit Testing
- ONNX model loading functions
- Tensor preprocessing operations
- Post-processing and NMS algorithms
- Model switching functionality

### Integration Testing
- Complete inference pipeline testing
- Model manager integration
- UI component integration
- Camera input processing

### Browser Testing
- Chrome (latest and -2 versions)
- Firefox (latest and -2 versions)
- Safari (latest and -1 versions)
- Edge (latest version)

### Performance Testing
- Inference speed benchmarking
- Memory usage profiling
- CPU utilization monitoring
- Battery impact assessment (mobile)

### User Acceptance Testing
- Model selection and switching
- Real-time detection accuracy
- Error handling and recovery
- Overall user experience

## Risk Assessment

### High Risk
1. **ONNX.js Compatibility**: Different browsers may have varying ONNX.js support
2. **Model Format Variations**: ONNX models may have incompatible output formats
3. **Performance Issues**: ONNX inference may be slower than expected

### Medium Risk
1. **Browser Caching**: Cache issues may persist across different browsers
2. **Memory Leaks**: Improper tensor disposal may cause memory issues
3. **Model Size**: Large ONNX models may cause loading delays

### Low Risk
1. **UI Integration**: Model selection UI is already implemented
2. **Configuration Parsing**: JSON config parsing is straightforward
3. **Fallback Options**: Simple detection provides reliable fallback

## Success Criteria

### Must Have (P0)
- [x] ONNX models load without errors
- [ ] ONNX inference completes successfully
- [ ] Detection results are displayed correctly
- [ ] Model switching works between all types
- [ ] No critical errors in browser console

### Should Have (P1)
- [ ] Performance meets 20+ FPS target
- [ ] Memory usage stays under 500MB
- [ ] Works on all target browsers
- [ ] Error recovery functions properly
- [ ] User feedback is clear and helpful

### Could Have (P2)
- [ ] Model warmup for faster startup
- [ ] Advanced performance metrics
- [ ] Mobile browser optimization
- [ ] Multiple ONNX model format support
- [ ] Automatic model recommendation

## Deliverables

1. **Fixed ONNX Integration**: Fully functional ONNX model support
2. **Updated Documentation**: User guides and technical documentation
3. **Test Suite**: Comprehensive testing for ONNX functionality
4. **Performance Benchmarks**: Performance metrics and optimization results
5. **Deployment Guide**: Instructions for deploying ONNX-enabled system

## Appendix

### File Structure
```
weedzap/
├── web/
│   ├── js/
│   │   ├── onnx-detector.js      # ONNX detection engine
│   │   ├── yolo-detector.js      # TensorFlow.js detection
│   │   ├── model-manager.js      # Model management
│   │   └── script.js             # Main application
│   ├── models/
│   │   └── custom-models/
│   │       ├── tiny_model_680_final.onnx
│   │       ├── tiny_model_680_final.json
│   │       ├── tiny_model_680.onnx
│   │       └── tiny_model_680.json
│   ├── index.html               # Main interface
│   └── test-onnx-debug.html     # Debug interface
├── ONNX_INTEGRATION_PRD.md      # This document
└── README.md                    # Project documentation
```

### Key Dependencies
- **ONNX.js**: Web runtime for ONNX model inference
- **TensorFlow.js**: Fallback for YOLO models
- **WebGL**: Hardware acceleration
- **WebRTC**: Camera input
- **ES6 Modules**: Modern JavaScript features

### Contact Information
- **Project Lead**: LaserWeed Development Team
- **Technical Lead**: ONNX Integration Specialist
- **QA Lead**: Browser Compatibility Team
- **Documentation**: Technical Writing Team

---

**Last Updated**: July 13, 2025  
**Next Review**: Upon completion of Phase 1 tasks
