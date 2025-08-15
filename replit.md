# iPhone Unknown Part Detector

## Overview

This is a single-file machine learning application for detecting "Unknown Part" versus "Genuine" components in iPhone images. The application combines deep learning model training, evaluation, prediction capabilities, and a Streamlit-based web interface for both inference and data annotation. It's designed as a complete end-to-end solution for building and deploying a binary image classification system.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (August 2025)

### Enhanced Annotation System
- **Bulk Image Annotation**: Added comprehensive bulk annotation tool for processing multiple image URLs
- **Area Marking**: Integrated interactive drawing canvas (streamlit-drawable-canvas) for marking specific areas/regions in images
- **Auto-progression**: After saving annotations, automatically moves to the next image for efficient workflow
- **Data Export**: Enhanced annotation export with bounding box coordinates and metadata
- **Progress Tracking**: Real-time progress indicators and navigation between images
- **Persistent Storage**: Annotations saved both in session state and persistent files

### Removed Test Data
- Removed dataset/test directory and all test images as requested by user
- Streamlined dataset structure to focus on train/validation splits

## System Architecture

### Single-File Architecture
The entire application is contained in a single Python file (`unknown_part.py`) that includes:
- Model training and evaluation logic
- Streamlit web interface for predictions and annotations
- Data management and preprocessing utilities
- Image download and caching mechanisms

### Machine Learning Pipeline
- **Framework**: PyTorch with torchvision for computer vision tasks
- **Model Architecture**: Transfer learning using pre-trained models (ResNet, EfficientNet, etc.)
- **Training Features**: 
  - Early stopping and learning rate scheduling
  - Mixed precision training for efficiency
  - Class imbalance handling with weighted loss functions
  - Data augmentation for robustness
- **Model Persistence**: Saves best and last model weights, TorchScript exports, and training history

### Data Management
- **Dataset Structure**: Standard train/validation/test split with class-based folder organization
- **Image Processing**: EXIF orientation correction, PIL-based image handling
- **Caching Strategy**: SHA256-based filename generation for downloaded images to avoid re-downloading
- **Annotation System**: Interactive canvas-based bounding box annotation integrated into Streamlit UI

### User Interface
- **Framework**: Streamlit for rapid web UI development
- **Interactive Components**: 
  - Image upload and URL input for predictions
  - Drawable canvas for annotation (streamlit-drawable-canvas)
  - Batch processing capabilities
  - Real-time prediction display with confidence scores

### Image Processing Pipeline
- **Download Handling**: Robust URL fetching with timeouts, retries, and content-type validation
- **Image Transformations**: Standard computer vision preprocessing (resize, normalize, augment)
- **Format Support**: PIL-compatible formats with automatic orientation correction

## External Dependencies

### Core ML Libraries
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities and pre-trained models
- **PIL (Pillow)**: Image processing and manipulation

### Web Interface
- **streamlit**: Web application framework
- **streamlit-drawable-canvas**: Interactive drawing component for annotations

### Data Processing
- **requests**: HTTP client for image downloads
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization

### Utilities
- **pathlib**: Modern path handling
- **hashlib**: SHA256 hashing for image caching
- **csv**: Dataset annotation export
- **json**: Configuration and metadata storage

### System Integration
- Standard Python libraries: os, sys, threading, signal, time, logging, dataclasses, datetime, textwrap, re, shutil, io, math, random