from __future__ import annotations
import logging
import math
import os
import json
import io
import random
import re
import shutil
import signal
import sys
import textwrap
import threading
import time
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Third-party
import requests
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# UI
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# OCR and CV
import cv2
import pytesseract

# Optional plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Utility: SHA256 of text (for file naming)
def sha256_of_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ========== Global Constants & Defaults ==========
APP_NAME = "iPhone Unknown Part Detector"
DEFAULT_MODEL_DIR = Path("artifacts")
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH_BEST = DEFAULT_MODEL_DIR / "model_best.pth"
MODEL_PATH_LAST = DEFAULT_MODEL_DIR / "model_last.pth"
MODEL_TORCHSCRIPT = DEFAULT_MODEL_DIR / "model_scripted.pt"
LABELS_PATH = DEFAULT_MODEL_DIR / "class_names.json"
HISTORY_PATH = DEFAULT_MODEL_DIR / "training_history.json"
METRICS_PATH = DEFAULT_MODEL_DIR / "metrics.json"
RUNTIME_CONFIG_PATH = DEFAULT_MODEL_DIR / "runtime_config.json"
IMAGE_CACHE_DIR = DEFAULT_MODEL_DIR / "image_cache"
ANNOTATIONS_CSV = DEFAULT_MODEL_DIR / "annotations.csv"
FEEDBACK_CSV = DEFAULT_MODEL_DIR / "feedback.csv"
LOG_DIR = DEFAULT_MODEL_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATASET_DIR = Path("dataset")

# ========== Logging Setup ==========
LOGGER = logging.getLogger(APP_NAME)
LOGGER.setLevel(logging.DEBUG)

class _StreamlitHandler(logging.Handler):
    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.capacity = capacity
        self.buffer: List[str] = []
    
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.buffer.append(msg)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

STREAMLIT_HANDLER = _StreamlitHandler(capacity=2000)
STREAMLIT_HANDLER.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
LOGGER.addHandler(STREAMLIT_HANDLER)

from logging.handlers import RotatingFileHandler
try:
    FILE_HANDLER = RotatingFileHandler(LOG_FILE, maxBytes=100_000, backupCount=1, encoding="utf-8")
    FILE_HANDLER.setLevel(logging.WARNING)  # Only log warnings and errors to file
    FILE_HANDLER.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(FILE_HANDLER)
except Exception:
    # If file handler fails, continue without it
    pass

CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
CONSOLE_HANDLER.setLevel(logging.INFO)
CONSOLE_HANDLER.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
LOGGER.addHandler(CONSOLE_HANDLER)

# ========== Dataclasses & Helper Classes ==========
SEED = 42
random.seed(SEED)

def set_torch_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    try:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

@dataclass
class TrainConfig:
    data_dir: str = str(DEFAULT_DATASET_DIR)
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    model_name: str = "resnet50"
    freeze_backbone: bool = False
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    amp: bool = True
    num_workers: int = 0

@dataclass
class AppConfig:
    request_timeout: int = 15
    max_image_size_mb: int = 10
    allowed_content_types: tuple = ("image/jpeg", "image/png", "image/webp")
    retry_attempts: int = 3
    retry_backoff_sec: float = 1.5

DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_APP_CONFIG = AppConfig()

class ImageFetcher:
    def __init__(self, app_cfg: AppConfig):
        self.app_cfg = app_cfg
    
    def fetch_bytes(self, url: str) -> bytes:
        last_exc = None
        for attempt in range(self.app_cfg.retry_attempts):
            try:
                resp = requests.get(url, timeout=self.app_cfg.request_timeout)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if not any(ctype.startswith(t) for t in self.app_cfg.allowed_content_types):
                    raise ValueError(f"Content-Type {ctype} not allowed")
                if int(resp.headers.get("content-length", 0)) > self.app_cfg.max_image_size_mb * 1024 * 1024:
                    raise ValueError("Image too large")
                return resp.content
            except Exception as e:
                last_exc = e
                time.sleep(self.app_cfg.retry_backoff_sec * (attempt + 1))
        raise RuntimeError(f"Failed to fetch image after retries: {last_exc}")
    
    def fetch(self, url: str) -> Image.Image:
        data = self.fetch_bytes(url)
        try:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            LOGGER.exception("Failed to parse image: %s", e)
            raise
        return img

# ========== Dataset Helpers ==========
@dataclass
class DatasetSummary:
    class_counts: Dict[str, int]
    total_images: int
    num_classes: int

def summarize_imagefolder(root: str) -> DatasetSummary:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    
    class_counts: Dict[str, int] = {}
    for phase in ["train", "val", "test"]:
        phase_dir = root_path / phase
        if not phase_dir.exists():
            continue
        try:
            ds = datasets.ImageFolder(phase_dir)
            counts = {ds.classes[i]: 0 for i in range(len(ds.classes))}
            for _, label in ds.samples:
                class_name = ds.classes[label]
                counts[class_name] += 1
            LOGGER.info("Phase '%s' counts: %s", phase, counts)
            for k, v in counts.items():
                class_counts[k] = class_counts.get(k, 0) + v
        except Exception as e:
            LOGGER.warning("Could not process phase %s: %s", phase, e)
    
    total = sum(class_counts.values())
    return DatasetSummary(class_counts=class_counts, total_images=total, num_classes=len(class_counts))

# ========== Model Factory ==========
class ModelFactory:
    @staticmethod
    def create_backbone(name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
        name = name.lower()
        if name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = model.fc.in_features
            if freeze_backbone:
                for p in model.parameters():
                    p.requires_grad = False
            model.fc = nn.Linear(in_features, num_classes)
            return model
        elif name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            if freeze_backbone:
                for p in model.parameters():
                    p.requires_grad = False
            model.fc = nn.Linear(in_features, num_classes)
            return model
        elif name == "efficientnet_v2_s":
            try:
                model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
                in_features = model.classifier[-1].in_features
                if freeze_backbone:
                    for p in model.parameters():
                        p.requires_grad = False
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                return model
            except Exception as e:
                LOGGER.warning("EfficientNet V2 not available, fallback to resnet50: %s", e)
                return ModelFactory.create_backbone("resnet50", num_classes, freeze_backbone)
        else:
            LOGGER.warning("Unknown model '%s', fallback to resnet50", name)
            return ModelFactory.create_backbone("resnet50", num_classes, freeze_backbone)

# ========== Augmentations ==========
def build_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),  # Slightly larger for cropping
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Minimal vertical flip for phone components
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.RandomAffine(
                degrees=15, 
                translate=(0.1, 0.1), 
                scale=(0.9, 1.1), 
                shear=8,
                fill=128  # Gray fill for rotations
            ),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Random erasing to simulate occlusions
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# ========== Mixup / CutMix ==========
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# ========== Training & Evaluation ==========
class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_torch_seed(SEED)
        
    def train(self, progress_callback=None):
        LOGGER.info("Starting training with config: %s", asdict(self.config))
        
        # Load dataset
        try:
            summary = summarize_imagefolder(self.config.data_dir)
            LOGGER.info("Dataset summary: %s", summary)
            
            if summary.num_classes == 0:
                raise ValueError("No classes found in dataset")
                
        except Exception as e:
            LOGGER.error("Dataset loading failed: %s", e)
            return False
        
        # Create data loaders
        train_transform = build_transforms(self.config.img_size, is_train=True)
        val_transform = build_transforms(self.config.img_size, is_train=False)
        
        train_dataset = datasets.ImageFolder(
            os.path.join(self.config.data_dir, "train"), 
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(self.config.data_dir, "val"), 
            transform=val_transform
        )
        
        # Handle severe class imbalance with weighted sampling
        class_counts = [0] * len(train_dataset.classes)
        for _, class_idx in train_dataset.samples:
            class_counts[class_idx] += 1
        
        LOGGER.info(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")
        
        # Calculate class weights (inverse frequency)
        total_samples = sum(class_counts)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        
        # Create sample weights for weighted sampling
        sample_weights = [class_weights[class_idx] for _, class_idx in train_dataset.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            sampler=sampler,  # Use weighted sampling instead of shuffle
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers
        )
        
        # Save class names and model config
        class_names = train_dataset.classes
        model_config = {
            "class_names": class_names,
            "model_name": self.config.model_name,
            "num_classes": len(class_names),
            "img_size": self.config.img_size
        }
        with open(LABELS_PATH, "w") as f:
            json.dump(model_config, f, indent=2)
        LOGGER.info("Saved model config: %s", model_config)
        
        # Create model
        model = ModelFactory.create_backbone(
            self.config.model_name, 
            len(class_names), 
            self.config.freeze_backbone
        )
        model = model.to(self.device)
        
        # Loss and optimizer with class balancing
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay,
            eps=1e-8,  # For numerical stability
            betas=(0.9, 0.999)
        )
        
        # Use ReduceLROnPlateau for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
        scaler = torch.cuda.amp.GradScaler() if self.config.amp and torch.cuda.is_available() else None
        
        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if progress_callback:
                    progress = (batch_idx + 1) / len(train_loader)
                    progress_callback(epoch, progress, "training")
            
            train_acc = 100 * train_correct / train_total
            train_loss_avg = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss_avg = val_loss / len(val_loader)
            
            # Update history
            history["train_loss"].append(train_loss_avg)
            history["val_loss"].append(val_loss_avg)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            
            # Log progress
            LOGGER.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                       f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH_BEST)
                patience_counter = 0
                LOGGER.info(f"New best model saved with val_acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stop_patience:
                LOGGER.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            scheduler.step(val_acc)  # Step with validation accuracy
        
        # Save final model and history
        torch.save(model.state_dict(), MODEL_PATH_LAST)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
        
        # Export to TorchScript
        try:
            model.eval()
            example_input = torch.randn(1, 3, self.config.img_size, self.config.img_size).to(self.device)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(MODEL_TORCHSCRIPT))
            LOGGER.info("Model exported to TorchScript")
        except Exception as e:
            LOGGER.warning("TorchScript export failed: %s", e)
        
        LOGGER.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        return True

# ========== Model Loading & Inference ==========
class Predictor:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        
    def load_model(self, model_path: Path = MODEL_PATH_BEST):
        """Load trained model for inference"""
        try:
            # Load model config
            if not LABELS_PATH.exists():
                LOGGER.info("No trained model found. Train a model first.")
                return False
            
            with open(LABELS_PATH, "r") as f:
                model_config = json.load(f)
            
            # Handle both old and new config formats
            if isinstance(model_config, list):
                # Old format - just class names
                self.class_names = model_config
                model_name = "resnet50"  # default for old models
            else:
                # New format - full config
                self.class_names = model_config["class_names"]
                model_name = model_config.get("model_name", "resnet50")
            
            # Create model architecture with correct model type
            self.model = ModelFactory.create_backbone(model_name, len(self.class_names), False)
            
            # Load weights
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.transform = build_transforms(224, is_train=False)
                LOGGER.info("Model loaded successfully")
                return True
            else:
                LOGGER.info("Model file not found. Train a model first.")
                return False
                
        except Exception as e:
            LOGGER.error("Failed to load model: %s", e)
            return False
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Enhanced prediction with Test-Time Augmentation (TTA)"""
        try:
            if self.model is None or self.transform is None:
                raise RuntimeError("Model not loaded")
            
            LOGGER.debug(f"Starting prediction with image size: {image.size}")
            
            # Apply multiple transformations (Test-Time Augmentation)
            tta_transforms = [
                # Original
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                # Slight rotations
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(degrees=5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                # Color adjustment
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            ]
            
            all_predictions = []
            
            # Run predictions with different augmentations
            with torch.no_grad():
                for transform in tta_transforms:
                    try:
                        input_tensor = transform(image).unsqueeze(0).to(self.device)
                        outputs = self.model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        all_predictions.append(probabilities.cpu())
                    except Exception as e:
                        LOGGER.warning(f"TTA transform failed: {e}")
                        continue
                
                if not all_predictions:
                    # Fallback to single prediction
                    input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    all_predictions = [probabilities.cpu()]
                
                # Average predictions
                avg_probabilities = torch.mean(torch.stack(all_predictions), dim=0)
                confidence, predicted_idx = torch.max(avg_probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                # Apply confidence calibration based on class
                confidence_score = self.calibrate_confidence(predicted_class, confidence_score, avg_probabilities[0])
                
                LOGGER.info(f"TTA Prediction: {predicted_class} with confidence {confidence_score:.3f}")
            
            return predicted_class, confidence_score
            
        except Exception as e:
            LOGGER.error(f"Prediction failed: {e}")
            LOGGER.exception("Full prediction error:")
            raise
    
    def calibrate_confidence(self, predicted_class: str, raw_confidence: float, probabilities: torch.Tensor) -> float:
        """Calibrate confidence scores based on class characteristics and probability distribution"""
        
        # Calculate entropy of the prediction (uncertainty measure)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
        max_entropy = -math.log(1.0 / len(self.class_names))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy
        
        # Adjust confidence based on entropy (lower entropy = higher confidence)
        entropy_adjustment = 1.0 - (normalized_entropy * 0.3)
        
        # Class-specific calibration factors (based on typical accuracy per class)
        class_factors = {
            "Genuine": 0.95,      # Usually easier to identify
            "Unknown Part": 0.85,  # Medium difficulty
            "Service": 0.80,       # Can be ambiguous
            "Data not correct": 0.70  # Often uncertain
        }
        
        class_factor = class_factors.get(predicted_class, 0.8)
        
        # Calculate calibrated confidence
        calibrated_confidence = raw_confidence * entropy_adjustment * class_factor
        
        # Ensure minimum confidence for very certain predictions
        if raw_confidence > 0.9 and normalized_entropy < 0.3:
            calibrated_confidence = max(calibrated_confidence, 0.75)
        
        return min(0.99, max(0.1, calibrated_confidence))

# ========== Image Quality Assessment ==========
class ImageQualityAssessor:
    @staticmethod
    def assess_image_quality(image: Image.Image) -> Dict[str, float]:
        """Assess various quality metrics of the input image"""
        try:
            img_array = np.array(image)
            
            # Convert to grayscale for some assessments
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            metrics = {}
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = min(1.0, laplacian_var / 500.0)  # Normalize
            
            # 2. Brightness
            brightness = np.mean(gray) / 255.0
            metrics['brightness'] = brightness
            
            # 3. Contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            metrics['contrast'] = contrast
            
            # 4. Image size appropriateness
            width, height = image.size
            total_pixels = width * height
            metrics['resolution_score'] = min(1.0, total_pixels / (224 * 224))
            
            # 5. Color distribution (for colored images)
            if len(img_array.shape) == 3:
                # Check if image is too monochromatic
                rgb_std = np.std(img_array.reshape(-1, 3), axis=0)
                color_variety = np.mean(rgb_std) / 255.0
                metrics['color_variety'] = color_variety
            else:
                metrics['color_variety'] = 0.0
            
            # 6. Noise estimation (high-frequency content)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = np.mean(np.sqrt(sobel_x**2 + sobel_y**2)) / 255.0
            metrics['edge_density'] = edge_density
            
            # 7. Overall quality score
            quality_weights = {
                'sharpness': 0.3,
                'contrast': 0.2,
                'brightness': 0.15,
                'resolution_score': 0.15,
                'color_variety': 0.1,
                'edge_density': 0.1
            }
            
            # Penalize extreme brightness
            brightness_penalty = 1.0
            if brightness < 0.1 or brightness > 0.9:
                brightness_penalty = 0.7
            
            overall_score = sum(metrics[key] * weight for key, weight in quality_weights.items())
            overall_score *= brightness_penalty
            
            metrics['overall_quality'] = min(1.0, max(0.0, overall_score))
            
            return metrics
            
        except Exception as e:
            LOGGER.warning(f"Image quality assessment failed: {e}")
            return {
                'sharpness': 0.5, 'brightness': 0.5, 'contrast': 0.5,
                'resolution_score': 0.5, 'color_variety': 0.5, 
                'edge_density': 0.5, 'overall_quality': 0.5
            }

# ========== Ensemble Predictor ==========
class EnsemblePredictor:
    def __init__(self):
        self.cnn_predictor = Predictor()
        self.ocr_detector = iPhoneServiceHistoryDetector()
        self.quality_assessor = ImageQualityAssessor()
        
    def load_model(self):
        """Load the CNN model"""
        return self.cnn_predictor.load_model()
    
    def predict_with_ensemble(self, image: Image.Image, use_ocr: bool = True) -> Tuple[str, float, Dict[str, Any]]:
        """
        Comprehensive prediction using ensemble of CNN + OCR + Quality Assessment
        Returns: (predicted_class, confidence, details)
        """
        details = {
            'cnn_prediction': None,
            'cnn_confidence': 0.0,
            'ocr_prediction': None,
            'ocr_confidence': 0.0,
            'ocr_text': '',
            'ensemble_method': 'cnn_only',
            'agreement': False,
            'quality_metrics': {},
            'quality_penalty': 1.0,
            'final_reasoning': ''
        }
        
        try:
            # 1. Assess image quality first
            quality_metrics = self.quality_assessor.assess_image_quality(image)
            details['quality_metrics'] = quality_metrics
            
            # Apply quality-based confidence penalty
            quality_penalty = 1.0
            if quality_metrics['overall_quality'] < 0.3:
                quality_penalty = 0.6
                details['final_reasoning'] = "Low image quality detected. "
            elif quality_metrics['overall_quality'] < 0.5:
                quality_penalty = 0.8
                details['final_reasoning'] = "Moderate image quality. "
            
            details['quality_penalty'] = quality_penalty
            
            # 2. Get CNN prediction
            cnn_class, cnn_conf = self.cnn_predictor.predict(image)
            details['cnn_prediction'] = cnn_class
            details['cnn_confidence'] = cnn_conf
            
            final_class = cnn_class
            final_confidence = cnn_conf * quality_penalty
            
            # 3. OCR Analysis (if enabled)
            if use_ocr:
                try:
                    # Get OCR prediction
                    ocr_class, ocr_conf, ocr_text = self.ocr_detector.detect_service_status(image)
                    details['ocr_prediction'] = ocr_class
                    details['ocr_confidence'] = ocr_conf
                    details['ocr_text'] = ocr_text[:200] + '...' if len(ocr_text) > 200 else ocr_text
                    
                    # Check if OCR found valid service history
                    if ocr_class not in ["No Service History Found", "Data not correct"] and ocr_conf > 0.3:
                        details['ensemble_method'] = 'ensemble'
                        
                        # Check agreement between CNN and OCR
                        agreement_score = self.calculate_agreement(cnn_class, ocr_class)
                        details['agreement'] = agreement_score > 0.5
                        
                        if agreement_score > 0.8:
                            # Strong agreement - high confidence boost
                            weight_cnn = 0.55
                            weight_ocr = 0.45
                            confidence_boost = 1.3
                            final_confidence = (cnn_conf * weight_cnn + ocr_conf * weight_ocr) * confidence_boost
                            final_confidence = min(0.95, final_confidence * quality_penalty)
                            
                            # Use higher confidence prediction
                            final_class = cnn_class if cnn_conf > ocr_conf else ocr_class
                            details['final_reasoning'] += f"Strong agreement between CNN and OCR ({agreement_score:.2f}). "
                            
                        elif agreement_score > 0.5:
                            # Moderate agreement
                            weight_cnn = 0.6
                            weight_ocr = 0.4
                            final_confidence = (cnn_conf * weight_cnn + ocr_conf * weight_ocr) * 1.15
                            final_confidence = min(0.90, final_confidence * quality_penalty)
                            
                            final_class = cnn_class if cnn_conf > ocr_conf else ocr_class
                            details['final_reasoning'] += f"Moderate agreement between predictions. "
                            
                        else:
                            # Disagreement - use more confident but penalize
                            if cnn_conf > ocr_conf:
                                final_class = cnn_class
                                final_confidence = cnn_conf * 0.75 * quality_penalty
                                details['final_reasoning'] += "CNN prediction preferred due to higher confidence. "
                            else:
                                final_class = ocr_class
                                final_confidence = ocr_conf * 0.75 * quality_penalty
                                details['final_reasoning'] += "OCR prediction preferred due to higher confidence. "
                    
                    # Special case: OCR strongly indicates service history screen
                    elif "parts and service history" in ocr_text.lower():
                        details['ensemble_method'] = 'ocr_informed'
                        # If OCR detects service screen but no clear status, prefer visual assessment
                        if cnn_class in ["Unknown Part", "Service"]:
                            final_confidence *= 1.1  # Slight boost for consistency
                            details['final_reasoning'] += "Service history screen detected via OCR. "
                        else:
                            # CNN says genuine/incorrect but OCR sees service screen
                            final_confidence *= 0.9  # Slight penalty for inconsistency
                            details['final_reasoning'] += "Inconsistency: Service screen detected but visual classification differs. "
                    
                    else:
                        # OCR didn't find valid results, rely on CNN
                        details['ensemble_method'] = 'cnn_primary'
                        details['final_reasoning'] += "No service history text detected, relying on visual analysis. "
                        
                except Exception as e:
                    LOGGER.warning(f"OCR processing failed: {e}")
                    details['ensemble_method'] = 'cnn_fallback'
                    details['final_reasoning'] += "OCR analysis failed, using visual analysis only. "
            
            # 4. Final quality and context checks
            final_confidence = self.apply_final_calibration(final_class, final_confidence, quality_metrics)
            
            # 5. Logging and return
            LOGGER.info(f"Ensemble prediction: {final_class} (conf: {final_confidence:.3f}, method: {details['ensemble_method']}, quality: {quality_metrics['overall_quality']:.2f})")
            
            return final_class, final_confidence, details
            
        except Exception as e:
            LOGGER.error(f"Ensemble prediction failed: {e}")
            return "Data not correct", 0.1, details
    
    def apply_final_calibration(self, predicted_class: str, confidence: float, quality_metrics: Dict[str, float]) -> float:
        """Apply final confidence calibration based on class and quality"""
        
        # Class-specific calibration
        class_calibration = {
            "Genuine": 1.0,         # Usually most reliable
            "Unknown Part": 0.95,   # Fairly reliable
            "Service": 0.90,        # Can be ambiguous  
            "Data not correct": 0.85  # Often uncertain
        }
        
        calibrated = confidence * class_calibration.get(predicted_class, 0.85)
        
        # Additional quality-based adjustments
        if quality_metrics['sharpness'] < 0.3:
            calibrated *= 0.9  # Blurry images are less reliable
        
        if quality_metrics['brightness'] < 0.2 or quality_metrics['brightness'] > 0.8:
            calibrated *= 0.95  # Extreme lighting reduces reliability
        
        # Ensure reasonable bounds
        return min(0.98, max(0.05, calibrated))
    
    def calculate_agreement(self, cnn_class: str, ocr_class: str) -> float:
        """Calculate agreement score between CNN and OCR predictions"""
        
        # Direct match
        if cnn_class == ocr_class:
            return 1.0
        
        # Compatible mappings (partial agreement)
        compatible_mappings = {
            ("Unknown Part", "Service"): 0.7,  # Both indicate non-genuine
            ("Service", "Unknown Part"): 0.7,
            ("Genuine", "Data not correct"): 0.3,  # OCR might miss genuine indicators
            ("Data not correct", "Genuine"): 0.3,
        }
        
        pair = (cnn_class, ocr_class)
        if pair in compatible_mappings:
            return compatible_mappings[pair]
        
        # Reverse pair
        reverse_pair = (ocr_class, cnn_class)
        if reverse_pair in compatible_mappings:
            return compatible_mappings[reverse_pair]
        
        # No agreement
        return 0.2
    
    def predict_batch(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        """Predict classes and confidences for multiple images"""
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded")
        
        results = []
        for img in images:
            try:
                result = self.predict(img)
                results.append(result)
            except Exception as e:
                LOGGER.warning("Prediction failed for image: %s", e)
                results.append(("Error", 0.0))
        
        return results

# ========== OCR iPhone Service History Detector ==========
class iPhoneServiceHistoryDetector:
    def __init__(self):
        # Known iPhone service history patterns and keywords
        self.service_patterns = [
            r"service[d]?",
            r"replaced",
            r"repaired",
            r"serviced",
            r"apple certified",
            r"genuine apple"
        ]
        
        self.unknown_part_patterns = [
            r"unknown part",
            r"non-genuine",
            r"third[- ]?party",
            r"aftermarket",
            r"not verified",
            r"replacement part"
        ]
        
        self.genuine_patterns = [
            r"genuine",
            r"original",
            r"apple certified",
            r"verified",
            r"authentic"
        ]
        
        # Component keywords to identify iPhone parts
        self.component_keywords = [
            "battery", "screen", "display", "camera", "speaker", "microphone",
            "home button", "touch id", "face id", "charging port", "lightning",
            "wireless charging", "logic board", "antenna", "vibrator",
            "proximity sensor", "ambient light sensor", "accelerometer"
        ]
    
    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing for better OCR accuracy"""
        try:
            # Convert to numpy array
            img_np = np.array(image)
            
            # Convert to grayscale
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Adaptive thresholding for better text separation
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((2,2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            return Image.fromarray(processed)
            
        except Exception as e:
            LOGGER.warning(f"Image preprocessing failed, using original: {e}")
            return image
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Enhanced text extraction with multiple OCR strategies"""
        try:
            # Strategy 1: Preprocess and extract from enhanced image
            processed_img = self.preprocess_image_for_ocr(image)
            
            # Custom OCR configuration for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,():/-+ '
            
            # Extract text with custom config
            processed_text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Strategy 2: Extract from original image as fallback
            original_text = pytesseract.image_to_string(image)
            
            # Combine both results
            combined_text = f"{processed_text}\n{original_text}"
            
            # Clean up the text
            cleaned_text = self.clean_extracted_text(combined_text)
            
            LOGGER.debug(f"Extracted text length: {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            LOGGER.error(f"OCR extraction failed: {e}")
            return ""
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common OCR mistakes
        corrections = {
            'Servicé': 'Service',
            'Serviced': 'Service',
            'Génuine': 'Genuine',
            'Genuíne': 'Genuine',
            'Unknówn': 'Unknown',
            'Pàrt': 'Part',
            '0enuine': 'Genuine',  # OCR often mistakes G with 0
            'Servlce': 'Service',   # OCR mistakes i with l
        }
        
        for mistake, correction in corrections.items():
            text = text.replace(mistake, correction)
        
        return text.strip()
    
    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Advanced pattern analysis of extracted text"""
        text_lower = text.lower()
        analysis = {
            'has_service_history_header': False,
            'service_matches': [],
            'unknown_part_matches': [],
            'genuine_matches': [],
            'component_mentions': [],
            'confidence_indicators': []
        }
        
        # Check for service history header
        header_patterns = [
            r"parts?\s+and\s+service\s+history",
            r"service\s+history",
            r"parts?\s+history",
            r"repair\s+history"
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, text_lower):
                analysis['has_service_history_header'] = True
                break
        
        # Find service-related matches
        for pattern in self.service_patterns:
            matches = re.findall(pattern, text_lower)
            analysis['service_matches'].extend(matches)
        
        # Find unknown part matches
        for pattern in self.unknown_part_patterns:
            matches = re.findall(pattern, text_lower)
            analysis['unknown_part_matches'].extend(matches)
        
        # Find genuine part matches
        for pattern in self.genuine_patterns:
            matches = re.findall(pattern, text_lower)
            analysis['genuine_matches'].extend(matches)
        
        # Find component mentions
        for component in self.component_keywords:
            if component.lower() in text_lower:
                analysis['component_mentions'].append(component)
        
        # Look for confidence indicators
        confidence_patterns = [
            (r"apple\s+certified", 0.9),
            (r"verified\s+genuine", 0.9),
            (r"original\s+apple", 0.9),
            (r"third[- ]?party", 0.8),
            (r"aftermarket", 0.8),
            (r"non[- ]?genuine", 0.8),
        ]
        
        for pattern, confidence in confidence_patterns:
            if re.search(pattern, text_lower):
                analysis['confidence_indicators'].append((pattern, confidence))
        
        return analysis
    
    def calculate_advanced_confidence(self, analysis: Dict[str, Any], predicted_class: str) -> float:
        """Calculate confidence score based on multiple factors"""
        base_confidence = 0.3
        
        # Header presence boost
        if analysis['has_service_history_header']:
            base_confidence += 0.2
        
        # Component mentions boost (indicates it's actually a service screen)
        component_boost = min(0.2, len(analysis['component_mentions']) * 0.05)
        base_confidence += component_boost
        
        # Pattern match scoring
        if predicted_class == "Unknown Part":
            unknown_count = len(analysis['unknown_part_matches'])
            if unknown_count > 0:
                base_confidence += min(0.4, unknown_count * 0.2)
                
        elif predicted_class == "Service":
            service_count = len(analysis['service_matches'])
            # Subtract header mentions (usually contains "service")
            adjusted_service_count = max(0, service_count - 2)
            if adjusted_service_count > 0:
                base_confidence += min(0.3, adjusted_service_count * 0.15)
                
        elif predicted_class == "Genuine":
            genuine_count = len(analysis['genuine_matches'])
            if genuine_count > 0:
                base_confidence += min(0.4, genuine_count * 0.2)
        
        # Confidence indicators boost
        for pattern, conf_boost in analysis['confidence_indicators']:
            if predicted_class.lower() in pattern:
                base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def detect_service_status(self, image: Image.Image) -> Tuple[str, float, str]:
        """Enhanced service status detection with pattern matching"""
        
        extracted_text = self.extract_text_from_image(image)
        
        if not extracted_text.strip():
            return "Data not correct", 0.1, "No text could be extracted from image"
        
        # Perform advanced text analysis
        analysis = self.analyze_text_patterns(extracted_text)
        
        LOGGER.debug(f"Text analysis: {analysis}")
        
        # Decision logic based on pattern analysis
        unknown_part_score = len(analysis['unknown_part_matches'])
        service_score = max(0, len(analysis['service_matches']) - 2)  # Subtract header mentions
        genuine_score = len(analysis['genuine_matches'])
        
        # Determine prediction
        predicted_class = "Data not correct"
        
        if unknown_part_score > 0:
            predicted_class = "Unknown Part"
        elif genuine_score > service_score and genuine_score > 0:
            predicted_class = "Genuine"
        elif service_score > 0:
            predicted_class = "Service"
        elif analysis['has_service_history_header']:
            # If we have the header but no clear indicators, check for component mentions
            if analysis['component_mentions']:
                predicted_class = "Genuine"  # Assume genuine if components listed but no issues
            else:
                predicted_class = "Data not correct"  # Header present but unclear content
        
        # Calculate confidence
        confidence = self.calculate_advanced_confidence(analysis, predicted_class)
        
        LOGGER.info(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
        LOGGER.debug(f"Scores - Unknown: {unknown_part_score}, Service: {service_score}, Genuine: {genuine_score}")
        
        return predicted_class, confidence, extracted_text

# ========== Feedback Management ==========
class FeedbackManager:
    def __init__(self):
        self.feedback_file = FEEDBACK_CSV
        self.ensure_feedback_file()
    
    def ensure_feedback_file(self):
        """Create feedback CSV if it doesn't exist"""
        if not self.feedback_file.exists():
            with open(self.feedback_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'image_url', 'predicted_class', 'confidence', 'user_feedback', 'correct_class', 'extracted_text'])
    
    def save_feedback(self, image_url: str, predicted_class: str, confidence: float, user_feedback: bool, correct_class: str, extracted_text: str):
        """Save user feedback to CSV"""
        try:
            with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    image_url,
                    predicted_class,
                    confidence,
                    user_feedback,
                    correct_class,
                    extracted_text.replace('\n', ' ').replace('\r', ' ')[:500]  # Limit text length
                ])
            LOGGER.info(f"Feedback saved for prediction: {predicted_class} -> {correct_class}")
        except Exception as e:
            LOGGER.error(f"Failed to save feedback: {e}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            import pandas as pd
            if self.feedback_file.exists():
                df = pd.read_csv(self.feedback_file)
                total_feedback = len(df)
                correct_predictions = len(df[df['user_feedback'] == True])
                accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0
                
                return {
                    'total_feedback': total_feedback,
                    'correct_predictions': correct_predictions,
                    'accuracy': accuracy,
                    'class_distribution': df['predicted_class'].value_counts().to_dict() if total_feedback > 0 else {}
                }
            else:
                return {'total_feedback': 0, 'correct_predictions': 0, 'accuracy': 0, 'class_distribution': {}}
        except Exception as e:
            LOGGER.error(f"Failed to get feedback stats: {e}")
            return {'total_feedback': 0, 'correct_predictions': 0, 'accuracy': 0, 'class_distribution': {}}

# ========== Streamlit UI Components ==========
def ui_sidebar(tcfg: TrainConfig, appcfg: AppConfig) -> Tuple[TrainConfig, AppConfig]:
    """Sidebar configuration"""
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("Training Config", expanded=False):
        tcfg.data_dir = st.text_input("Dataset Directory", value=tcfg.data_dir)
        tcfg.img_size = st.selectbox("Image Size", [224, 256, 384], index=0)
        tcfg.batch_size = st.slider("Batch Size", 4, 64, tcfg.batch_size)
        tcfg.epochs = st.slider("Epochs", 1, 50, tcfg.epochs)
        tcfg.lr = st.number_input("Learning Rate", 1e-6, 1e-1, tcfg.lr, format="%.6f")
        tcfg.model_name = st.selectbox("Model", ["resnet18", "resnet50", "efficientnet_v2_s"])
        tcfg.freeze_backbone = st.checkbox("Freeze Backbone", tcfg.freeze_backbone)
        tcfg.amp = st.checkbox("Mixed Precision", tcfg.amp)
    
    with st.sidebar.expander("App Config", expanded=False):
        appcfg.request_timeout = st.slider("Request Timeout (s)", 5, 30, appcfg.request_timeout)
        appcfg.max_image_size_mb = st.slider("Max Image Size (MB)", 1, 50, appcfg.max_image_size_mb)
        appcfg.retry_attempts = st.slider("Retry Attempts", 1, 5, appcfg.retry_attempts)
    
    return tcfg, appcfg

def ui_predict_single(appcfg: AppConfig, tcfg: TrainConfig):
    """Single image prediction UI"""
    st.header("🔍 Single Image Prediction")
    
    # Initialize ensemble predictor in session state
    if 'predictor' not in st.session_state:
        st.session_state['predictor'] = EnsemblePredictor()
        st.session_state['predictor_loaded'] = False
        st.session_state['uploaded_image'] = None
        st.session_state['fetched_image'] = None
    
    predictor = st.session_state['predictor']
    
    # Load model section
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 Load Model", key="load_single"):
            try:
                with st.spinner("Loading model..."):
                    if predictor.load_model():
                        st.success("✅ Model loaded!")
                        st.session_state['predictor_loaded'] = True
                    else:
                        st.error("❌ Failed to load model. Train first.")
                        st.session_state['predictor_loaded'] = False
            except Exception as e:
                st.error(f"Load error: {e}")
                st.session_state['predictor_loaded'] = False
    
    with col2:
        if st.session_state.get('predictor_loaded', False):
            st.success("✅ Model ready for predictions")
        else:
            st.warning("⚠️ Model not loaded")
    
    st.divider()
    
    # Image input options
    input_method = st.radio("Choose input method:", ["Upload Image", "Image URL"])
    
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'webp'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state['uploaded_image'] = image
            st.session_state['fetched_image'] = None
        elif 'uploaded_image' in st.session_state:
            image = st.session_state['uploaded_image']
    else:
        url = st.text_input("Enter image URL:", key="image_url_input")
        if url and st.button("🔗 Fetch Image", key="fetch_btn"):
            try:
                fetcher = ImageFetcher(appcfg)
                with st.spinner("Fetching image..."):
                    image = fetcher.fetch(url)
                st.success("✅ Image fetched successfully")
                st.session_state['fetched_image'] = image
                st.session_state['uploaded_image'] = None
            except Exception as e:
                st.error(f"❌ Failed to fetch image: {e}")
        elif 'fetched_image' in st.session_state:
            image = st.session_state['fetched_image']
    
    # Display and predict
    if image:
        st.divider()
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        
        with col2:
            if not st.session_state.get('predictor_loaded', False):
                st.warning("⚠️ Load the model first")
            else:
                if st.button("🔮 Predict", key="predict_single", type="primary"):
                    prediction_container = st.container()
                    with prediction_container:
                        try:
                            LOGGER.info("Starting prediction...")
                            
                            # Show progress
                            progress_text = st.empty()
                            progress_text.text("🔄 Processing image...")
                            
                            with st.spinner("Analyzing image..."):
                                # Use ensemble prediction with detailed analysis
                                use_ocr = st.checkbox("Enable OCR Analysis", value=True, key="use_ocr_single")
                                predicted_class, confidence, details = predictor.predict_with_ensemble(image, use_ocr=use_ocr)
                            
                            progress_text.empty()
                            
                            # Display main results
                            st.subheader("🎯 Prediction Results")
                            
                            # Color-coded result with enhanced display
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                if predicted_class == "Unknown Part":
                                    st.error(f"🚨 **{predicted_class}**")
                                elif predicted_class == "Service":
                                    st.warning(f"⚠️ **{predicted_class}**")
                                elif predicted_class == "Genuine":
                                    st.success(f"✅ **{predicted_class}**")
                                elif predicted_class == "Data not correct":
                                    st.info(f"❌ **{predicted_class}**")
                                else:
                                    st.write(f"❓ **{predicted_class}**")
                            
                            with col2:
                                st.metric("Confidence", f"{confidence:.1%}")
                                st.progress(confidence)
                            
                            with col3:
                                quality_score = details.get('quality_metrics', {}).get('overall_quality', 0)
                                st.metric("Image Quality", f"{quality_score:.1%}")
                                st.progress(quality_score)
                            
                            # Analysis details in expandable section
                            with st.expander("🔬 Detailed Analysis", expanded=confidence < 0.6):
                                
                                # Method used
                                method = details.get('ensemble_method', 'unknown')
                                st.write(f"**Analysis Method:** {method.replace('_', ' ').title()}")
                                
                                # Reasoning
                                if details.get('final_reasoning'):
                                    st.write(f"**Reasoning:** {details['final_reasoning']}")
                                
                                # CNN vs OCR comparison
                                if details.get('cnn_prediction') and details.get('ocr_prediction'):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Visual Analysis (CNN)**")
                                        st.write(f"Prediction: {details['cnn_prediction']}")
                                        st.write(f"Confidence: {details['cnn_confidence']:.1%}")
                                    
                                    with col2:
                                        st.write("**Text Analysis (OCR)**")
                                        st.write(f"Prediction: {details['ocr_prediction']}")
                                        st.write(f"Confidence: {details['ocr_confidence']:.1%}")
                                        
                                    # Agreement indicator
                                    if details.get('agreement'):
                                        st.success("✅ Methods agree - high confidence")
                                    else:
                                        st.warning("⚠️ Methods disagree - moderate confidence")
                                
                                # Quality metrics breakdown
                                quality_metrics = details.get('quality_metrics', {})
                                if quality_metrics:
                                    st.write("**Image Quality Breakdown:**")
                                    metrics_cols = st.columns(3)
                                    
                                    with metrics_cols[0]:
                                        st.write(f"Sharpness: {quality_metrics.get('sharpness', 0):.2f}")
                                        st.write(f"Contrast: {quality_metrics.get('contrast', 0):.2f}")
                                    
                                    with metrics_cols[1]:
                                        st.write(f"Brightness: {quality_metrics.get('brightness', 0):.2f}")
                                        st.write(f"Resolution: {quality_metrics.get('resolution_score', 0):.2f}")
                                    
                                    with metrics_cols[2]:
                                        st.write(f"Color Variety: {quality_metrics.get('color_variety', 0):.2f}")
                                        st.write(f"Edge Density: {quality_metrics.get('edge_density', 0):.2f}")
                                
                                # OCR text preview
                                if details.get('ocr_text') and len(details['ocr_text']) > 10:
                                    st.write("**Extracted Text Preview:**")
                                    st.text_area("OCR Output", details['ocr_text'], height=100, disabled=True)
                            
                            # Confidence warnings
                            if confidence < 0.3:
                                st.error("⚠️ Very low confidence - results may be unreliable")
                            elif confidence < 0.5:
                                st.warning("⚠️ Low confidence - consider retaking the image")
                            elif confidence > 0.8:
                                st.success("✅ High confidence prediction")
                            
                            LOGGER.info(f"Enhanced prediction completed: {predicted_class} ({confidence:.1%}, method: {details.get('ensemble_method', 'unknown')})")
                            
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
                            LOGGER.error(f"Prediction error: {e}")
                            
                            with st.expander("🔍 Debug Information", expanded=False):
                                st.write(f"Model loaded: {predictor.model is not None}")
                                st.write(f"Transform loaded: {predictor.transform is not None}")
                                st.write(f"Class names: {predictor.class_names}")
                                st.write(f"Image type: {type(image)}")
                                if hasattr(image, 'size'):
                                    st.write(f"Image size: {image.size}")
                                st.write(f"Device: {predictor.device}")
    else:
        st.info("📷 Please upload an image or fetch from URL to start prediction")

def ui_predict_batch(appcfg: AppConfig, tcfg: TrainConfig):
    """Batch prediction UI"""
    st.header("📊 Batch Prediction")
    
    # URL input
    urls_text = st.text_area(
        "Enter image URLs (one per line):", 
        height=150,
        placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg\n..."
    )
    
    if st.button("🚀 Process Batch") and urls_text.strip():
        urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
        
        if not urls:
            st.warning("No valid URLs provided")
            return
        
        predictor = EnsemblePredictor()
        if not predictor.load_model():
            st.error("❌ Failed to load model. Train a model first.")
            return
        
        fetcher = ImageFetcher(appcfg)
        results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, url in enumerate(urls):
            try:
                status_text.text(f"Processing {i+1}/{len(urls)}: {url[:50]}...")
                
                # Fetch image
                image = fetcher.fetch(url)
                
                # Predict with ensemble
                predicted_class, confidence, _ = predictor.predict_with_ensemble(image)
                
                results.append({
                    "URL": url,
                    "Predicted Class": predicted_class,
                    "Confidence": f"{confidence:.1%}",
                    "Status": "✅ Success"
                })
                
            except Exception as e:
                LOGGER.warning("Batch prediction failed for %s: %s", url, e)
                results.append({
                    "URL": url,
                    "Predicted Class": "Error",
                    "Confidence": "0%",
                    "Status": f"❌ {str(e)[:50]}"
                })
            
            progress_bar.progress((i + 1) / len(urls))
        
        status_text.text("✅ Batch processing completed!")
        
        # Display results
        if results:
            st.subheader("📋 Results Summary")
            
            # Summary statistics
            success_count = sum(1 for r in results if r["Status"] == "✅ Success")
            unknown_count = sum(1 for r in results if r["Predicted Class"] == "Unknown Part")
            genuine_count = sum(1 for r in results if r["Predicted Class"] == "Genuine")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Processed", len(results))
            col2.metric("Success Rate", f"{success_count/len(results):.1%}")
            col3.metric("Unknown Parts", unknown_count)
            col4.metric("Genuine", genuine_count)
            
            # Results table
            st.dataframe(results, use_container_width=True)
            
            # Download option
            csv_data = "\n".join([
                ",".join(results[0].keys()),
                *[",".join(f'"{v}"' for v in r.values()) for r in results]
            ])
            
            st.download_button(
                "📥 Download Results (CSV)",
                csv_data,
                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def ui_annotate(appcfg: AppConfig, tcfg: TrainConfig):
    """Enhanced Bulk Annotation Tool with Area Marking"""
    st.header("✏️ Bulk Image Annotation Tool")
    
    # Initialize session state for annotation
    if 'annotation_urls' not in st.session_state:
        st.session_state.annotation_urls = []
    if 'current_annotation_idx' not in st.session_state:
        st.session_state.current_annotation_idx = 0
    if 'annotated_data' not in st.session_state:
        st.session_state.annotated_data = {}
    
    st.info("📋 Paste your image URLs below (one per line) for bulk annotation with area marking capabilities")
    
    # URL input section
    url_text = st.text_area(
        "Paste image URLs (one per line):", 
        height=120, 
        placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg\n...",
        key="bulk_annotate_url_text_area"
    )
    
    # Load URLs button
    if st.button("🔄 Load URLs", key="load_urls"):
        urls = [u.strip() for u in url_text.split('\n') if u.strip()]
        if urls:
            st.session_state.annotation_urls = urls
            st.session_state.current_annotation_idx = 0
            st.session_state.annotated_data = {}
            st.success(f"✅ Loaded {len(urls)} URLs for annotation")
            st.rerun()
    
    if not st.session_state.annotation_urls:
        st.markdown("""
        ### 🎯 How to use this tool:
        1. **Paste URLs**: Add your image URLs above (one per line)
        2. **Load URLs**: Click "Load URLs" to start the annotation process
        3. **Mark Areas**: Use the drawing canvas to mark important areas in each image
        4. **Classify**: Select the correct classification for each image
        5. **Auto-Progress**: After annotation, automatically moves to the next image
        6. **Export Data**: Download your annotations for training
        """)
        return
    
    # Progress tracking
    total_images = len(st.session_state.annotation_urls)
    current_idx = st.session_state.current_annotation_idx
    completed = len(st.session_state.annotated_data)
    
    # Progress bar and navigation
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        progress = (current_idx + 1) / total_images if total_images > 0 else 0
        st.progress(progress, f"Image {current_idx + 1} of {total_images} | Annotated: {completed}")
    
    with col2:
        if st.button("⬅️ Previous", disabled=current_idx <= 0):
            st.session_state.current_annotation_idx = max(0, current_idx - 1)
            st.rerun()
    
    with col3:
        if st.button("➡️ Next", disabled=current_idx >= total_images - 1):
            st.session_state.current_annotation_idx = min(total_images - 1, current_idx + 1)
            st.rerun()
    
    if current_idx >= total_images:
        st.success("🎉 All images have been processed!")
        st.balloons()
        return
    
    # Current image processing
    current_url = st.session_state.annotation_urls[current_idx]
    st.write(f"**Current Image:** {current_url}")
    
    # Load and display image
    try:
        fetcher = ImageFetcher(appcfg)
        with st.spinner("Loading image..."):
            image = fetcher.fetch(current_url)
            
        # Convert PIL image to numpy array for canvas
        import numpy as np
        image_np = np.array(image)
        
        # Main annotation interface
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("🖼️ Image & Area Marking")
            
            # Display the image first
            st.image(image, caption=f"Image {current_idx + 1}", use_container_width=True)
            
            # Drawing canvas for area annotation (without background image to avoid compatibility issues)
            canvas_width = min(600, image.width)
            canvas_height = int(canvas_width * image.height / image.width)
            
            # Try to create canvas with error handling for compatibility issues
            canvas_result = None
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.2)",  # Semi-transparent red
                    stroke_width=3,
                    stroke_color="#FF0000",  # Red stroke
                    update_streamlit=True,
                    width=canvas_width,
                    height=canvas_height,
                    drawing_mode="rect",  # Rectangle drawing mode
                    point_display_radius=0,
                    key=f"canvas_{current_idx}",
                )
                st.write("💡 **Instructions**: Draw rectangles on the canvas above to mark important areas. The canvas dimensions match the image proportions.")
            except Exception as e:
                st.warning(f"⚠️ Drawing canvas temporarily unavailable. You can still classify images.")
                st.info("Canvas error: This may be due to library compatibility. Image classification still works normally.")
                canvas_result = None
            
        with col2:
            st.subheader("📝 Annotation Details")
            
            # Classification selection
            classification = st.selectbox(
                "Select Classification:",
                ["Unknown Part", "Genuine", "Service", "Data not correct"],
                key=f"classification_{current_idx}"
            )
            
            # Additional notes
            notes = st.text_area(
                "Additional Notes (optional):",
                placeholder="Any additional observations about this image...",
                height=100,
                key=f"notes_{current_idx}"
            )
            
            # Area annotations info
            if canvas_result and canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects:
                    st.write(f"📐 **Marked Areas**: {len(objects)} rectangle(s)")
                    for i, obj in enumerate(objects):
                        if obj["type"] == "rect":
                            st.write(f"• Area {i+1}: {obj['width']:.0f}×{obj['height']:.0f}px at ({obj['left']:.0f}, {obj['top']:.0f})")
            elif canvas_result is None:
                st.info("📐 Area marking unavailable - canvas compatibility issue")
            
            st.divider()
            
            # Save annotation
            if st.button("💾 Save Annotation", key=f"save_{current_idx}", type="primary"):
                # Extract bounding box data
                bounding_boxes = []
                if canvas_result and canvas_result.json_data is not None:
                    for obj in canvas_result.json_data["objects"]:
                        if obj["type"] == "rect":
                            bounding_boxes.append({
                                "x": obj["left"],
                                "y": obj["top"], 
                                "width": obj["width"],
                                "height": obj["height"]
                            })
                
                # Save annotation data
                annotation_data = {
                    "url": current_url,
                    "classification": classification,
                    "notes": notes,
                    "bounding_boxes": bounding_boxes,
                    "annotated_at": datetime.now().isoformat(),
                    "image_size": {"width": image.width, "height": image.height}
                }
                
                st.session_state.annotated_data[current_idx] = annotation_data
                
                # Save to CSV file for persistence
                save_annotation_to_file(annotation_data, current_idx)
                
                st.success("✅ Annotation saved!")
                
                # Auto-progress to next image
                if current_idx < total_images - 1:
                    st.session_state.current_annotation_idx += 1
                    time.sleep(1)  # Brief pause to show success message
                    st.rerun()
                else:
                    st.success("🎉 All images completed!")
            
            # Skip current image
            if st.button("⏭️ Skip This Image", key=f"skip_{current_idx}"):
                if current_idx < total_images - 1:
                    st.session_state.current_annotation_idx += 1
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Failed to load image: {e}")
        if st.button("⏭️ Skip to Next", key="skip_error"):
            if current_idx < total_images - 1:
                st.session_state.current_annotation_idx += 1
                st.rerun()
    
    # Annotation summary
    st.divider()
    st.subheader("📊 Annotation Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", total_images)
    col2.metric("Completed", completed)
    col3.metric("Remaining", total_images - completed)
    col4.metric("Progress", f"{(completed/total_images*100):.1f}%" if total_images > 0 else "0%")
    
    # Export annotations
    if st.session_state.annotated_data:
        if st.button("📥 Export Annotations", key="export_annotations"):
            export_annotations_to_csv()
            st.success("✅ Annotations exported to CSV!")
    
    # Clear all annotations
    if st.button("🗑️ Clear All Annotations", key="clear_annotations"):
        st.session_state.annotated_data = {}
        st.session_state.current_annotation_idx = 0
        st.success("✅ All annotations cleared!")
        st.rerun()

def save_annotation_to_file(annotation_data, idx):
    """Save individual annotation to file"""
    try:
        # Ensure annotations directory exists
        annotations_dir = DEFAULT_MODEL_DIR / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual annotation
        annotation_file = annotations_dir / f"annotation_{idx}.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
            
    except Exception as e:
        LOGGER.error(f"Failed to save annotation to file: {e}")

def export_annotations_to_csv():
    """Export all annotations to CSV format"""
    try:
        if not st.session_state.annotated_data:
            return
        
        # Prepare CSV data
        csv_rows = []
        for idx, data in st.session_state.annotated_data.items():
            # Basic info
            base_row = {
                "image_id": idx,
                "url": data["url"],
                "classification": data["classification"],
                "notes": data["notes"],
                "annotated_at": data["annotated_at"],
                "image_width": data["image_size"]["width"],
                "image_height": data["image_size"]["height"],
                "num_bounding_boxes": len(data["bounding_boxes"])
            }
            
            # Add bounding box info
            if data["bounding_boxes"]:
                for i, bbox in enumerate(data["bounding_boxes"]):
                    row = base_row.copy()
                    row.update({
                        "bbox_id": i,
                        "bbox_x": bbox["x"],
                        "bbox_y": bbox["y"],
                        "bbox_width": bbox["width"],
                        "bbox_height": bbox["height"]
                    })
                    csv_rows.append(row)
            else:
                csv_rows.append(base_row)
        
        # Convert to pandas DataFrame and save
        import pandas as pd
        df = pd.DataFrame(csv_rows)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = DEFAULT_MODEL_DIR / f"bulk_annotations_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Also save to session state for download
        csv_content = df.to_csv(index=False)
        st.download_button(
            "📥 Download Annotations CSV",
            csv_content,
            file_name=f"bulk_annotations_{timestamp}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        LOGGER.error(f"Failed to export annotations: {e}")
        st.error(f"❌ Failed to export annotations: {e}")

def organize_annotations_to_dataset():
    """Organize annotated data into training dataset structure"""
    if not ANNOTATIONS_CSV.exists():
        return
    
    try:
        # Create dataset directories
        dataset_dir = Path(DEFAULT_TRAIN_CONFIG.data_dir)
        for split in ["train", "val"]:
            for class_name in ["Genuine", "Service", "Unknown Part", "Data not correct"]:
                (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Read annotations
        with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            annotations = list(reader)
        
        # Download and organize images
        fetcher = ImageFetcher(DEFAULT_APP_CONFIG)
        organized_count = 0
        
        for annotation in annotations:
            try:
                url = annotation["url"]
                label = annotation["label"]
                
                # Generate filename from URL hash
                filename = f"{sha256_of_text(url)}.jpg"
                
                # Determine split (80% train, 20% val)
                split = "train" if hash(url) % 5 != 0 else "val"
                
                # Target path
                target_path = dataset_dir / split / label / filename
                
                # Skip if already exists
                if target_path.exists():
                    continue
                
                # Download and save image
                image = fetcher.fetch(url)
                image.save(target_path, "JPEG", quality=95)
                organized_count += 1
                
            except Exception as e:
                LOGGER.warning("Failed to organize annotation %s: %s", annotation, e)
        
        if organized_count > 0:
            st.success(f"✅ Organized {organized_count} images into dataset structure")
            LOGGER.info("Organized %d images into dataset", organized_count)
    
    except Exception as e:
        st.error(f"Failed to organize dataset: {e}")
        LOGGER.exception("Dataset organization failed")

def setup_training_data_from_urls(urls: List[str], label: str = "Unknown Part"):
    """Helper function to create training dataset from a list of URLs"""
    try:
        # Create dataset directories
        dataset_dir = Path(DEFAULT_TRAIN_CONFIG.data_dir)
        for split in ["train", "val"]:
            for class_name in ["Genuine", "Service", "Unknown Part", "Data not correct"]:
                (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        fetcher = ImageFetcher(DEFAULT_APP_CONFIG)
        organized_count = 0
        failed_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, url in enumerate(urls):
            try:
                progress = (i + 1) / len(urls)
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1}/{len(urls)}: {url[:50]}...")
                
                # Generate filename from URL hash
                filename = f"{sha256_of_text(url)}.jpg"
                
                # Determine split (80% train, 20% val)
                split = "train" if hash(url) % 5 != 0 else "val"
                
                # Target path
                target_path = dataset_dir / split / label / filename
                
                # Skip if already exists
                if target_path.exists():
                    continue
                
                # Download and save image
                image = fetcher.fetch(url)
                image.save(target_path, "JPEG", quality=95)
                organized_count += 1
                
            except Exception as e:
                LOGGER.warning("Failed to process URL %s: %s", url, e)
                failed_count += 1
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        st.success(f"✅ Organized {organized_count} images into '{label}' class")
        if failed_count > 0:
            st.warning(f"⚠️ Failed to process {failed_count} URLs")
        
        return organized_count
        
    except Exception as e:
        st.error(f"Failed to setup training data: {e}")
        LOGGER.exception("Training data setup failed")
        return 0

def ui_train(tcfg: TrainConfig):
    """Training UI"""
    st.header("🎯 Model Training")
    
    # Dataset preparation section
    st.subheader("📁 Dataset Preparation")
    
    # Bulk URL input for training data
    with st.expander("🔗 Add Training Data from URLs", expanded=False):
        st.info("Add multiple URLs to automatically download and organize training images")
        
        col1, col2 = st.columns(2)
        with col1:
            label_choice = st.selectbox(
                "Label for these images:",
                ["Genuine", "Service", "Unknown Part", "Data not correct"],
                key="bulk_label_choice"
            )
        
        urls_input = st.text_area(
            "Paste URLs (one per line):",
            height=150,
            placeholder="https://s3n.cashify.in/public/logistics-integration/images/MPMKC10684991_1702030885655.jpg\nhttps://example.com/image2.jpg\n...",
            key="bulk_urls_input"
        )
        
        if st.button("📥 Download and Organize Images", key="download_organize"):
            urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
            if urls:
                st.info(f"Processing {len(urls)} URLs for '{label_choice}' class...")
                count = setup_training_data_from_urls(urls, label_choice)
                if count > 0:
                    st.balloons()
            else:
                st.warning("Please paste some URLs first")
    
    st.divider()
    
    # Dataset summary
    if st.button("📊 Check Dataset", key="check_dataset"):
        try:
            summary = summarize_imagefolder(tcfg.data_dir)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", summary.total_images)
            col2.metric("Classes", summary.num_classes)
            col3.metric("Classes Found", len(summary.class_counts))
            
            if summary.class_counts:
                st.subheader("📈 Class Distribution")
                
                # Create bar chart
                classes = list(summary.class_counts.keys())
                counts = list(summary.class_counts.values())
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(classes, counts)
                ax.set_ylabel('Number of Images')
                ax.set_title('Dataset Class Distribution')
                
                # Color bars
                colors = {'Genuine': '#44ff44', 'Service': '#ffaa44', 'Unknown Part': '#ff4444', 'Data not correct': '#4488ff'}
                for i, (bar, class_name) in enumerate(zip(bars, classes)):
                    bar.set_color(colors.get(class_name, '#888888'))
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show detailed counts
                for class_name, count in summary.class_counts.items():
                    percentage = count / summary.total_images * 100
                    st.write(f"**{class_name}**: {count} images ({percentage:.1f}%)")
            
        except Exception as e:
            st.error(f"❌ Failed to check dataset: {e}")
            st.info("Make sure your dataset follows the structure:\n"
                   "```\n"
                   "dataset/\n"
                   "  train/\n"
                   "    Unknown Part/\n"
                   "    Genuine/\n"
                   "  val/\n"
                   "    Unknown Part/\n"
                   "    Genuine/\n"
                   "```")
    
    st.divider()
    
    # Training controls
    if st.button("🚀 Start Training", key="start_training", type="primary"):
        # Validate dataset first
        try:
            summary = summarize_imagefolder(tcfg.data_dir)
            if summary.total_images == 0:
                st.error("❌ No images found in dataset")
                return
            if summary.num_classes < 2:
                st.error("❌ Need at least 2 classes for training")
                return
        except Exception as e:
            st.error(f"❌ Dataset validation failed: {e}")
            return
        
        # Start training
        trainer = Trainer(tcfg)
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            st.subheader("🔥 Training Progress")
            epoch_progress = st.progress(0)
            batch_progress = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(epoch, batch_progress_val, phase):
                epoch_progress.progress(epoch / tcfg.epochs)
                batch_progress.progress(batch_progress_val)
                status_text.text(f"Epoch {epoch+1}/{tcfg.epochs} - {phase}: {batch_progress_val:.1%}")
            
            # Run training
            success = trainer.train(progress_callback)
            
            if success:
                st.success("🎉 Training completed successfully!")
                
                # Load and display training history
                if HISTORY_PATH.exists():
                    with open(HISTORY_PATH, "r") as f:
                        history = json.load(f)
                    
                    # Plot training curves
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Loss curves
                    ax1.plot(history["train_loss"], label="Train Loss", color='blue')
                    ax1.plot(history["val_loss"], label="Val Loss", color='red')
                    ax1.set_title("Training and Validation Loss")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss")
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Accuracy curves
                    ax2.plot(history["train_acc"], label="Train Acc", color='blue')
                    ax2.plot(history["val_acc"], label="Val Acc", color='red')
                    ax2.set_title("Training and Validation Accuracy")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy (%)")
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show final metrics
                    final_train_acc = history["train_acc"][-1]
                    final_val_acc = history["val_acc"][-1]
                    best_val_acc = max(history["val_acc"])
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Final Train Accuracy", f"{final_train_acc:.1f}%")
                    col2.metric("Final Val Accuracy", f"{final_val_acc:.1f}%")
                    col3.metric("Best Val Accuracy", f"{best_val_acc:.1f}%")
            
            else:
                st.error("❌ Training failed. Check the logs for details.")

def ui_evaluate(tcfg: TrainConfig):
    """Model evaluation UI"""
    st.header("📈 Model Evaluation")
    
    if st.button("🔍 Evaluate Model", key="evaluate_model"):
        predictor = EnsemblePredictor()
        
        if not predictor.load_model():
            st.error("❌ No trained model found. Train a model first.")
            return
        
        # Check for test set
        test_dir = Path(tcfg.data_dir) / "test"
        if not test_dir.exists():
            st.warning("⚠️ No test directory found. Using validation set for evaluation.")
            test_dir = Path(tcfg.data_dir) / "val"
        
        if not test_dir.exists():
            st.error("❌ No validation or test data found")
            return
        
        try:
            # Load test dataset
            transform = build_transforms(224, is_train=False)
            test_dataset = datasets.ImageFolder(test_dir, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Evaluation metrics
            all_predictions = []
            all_labels = []
            all_confidences = []
            
            with st.spinner("Evaluating model..."):
                # Use the CNN predictor from the ensemble for evaluation
                cnn_model = predictor.cnn_predictor
                cnn_model.model.eval()
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(cnn_model.device)
                        outputs = cnn_model.model(images)
                        
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidences, predictions = torch.max(probabilities, 1)
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.numpy())
                        all_confidences.extend(confidences.cpu().numpy())
            
            # Calculate metrics
            from collections import Counter
            
            correct = sum(p == l for p, l in zip(all_predictions, all_labels))
            total = len(all_labels)
            accuracy = correct / total
            
            # Per-class accuracy
            class_names = test_dataset.classes
            class_accuracies = {}
            
            for i, class_name in enumerate(class_names):
                class_predictions = [p for p, l in zip(all_predictions, all_labels) if l == i]
                class_correct = sum(p == i for p in class_predictions)
                class_total = len(class_predictions)
                class_accuracies[class_name] = class_correct / class_total if class_total > 0 else 0
            
            # Display results
            st.subheader("🎯 Evaluation Results")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Accuracy", f"{accuracy:.1%}")
            col2.metric("Total Samples", total)
            col3.metric("Average Confidence", f"{np.mean(all_confidences):.1%}")
            
            # Per-class results
            st.subheader("📊 Per-Class Performance")
            for class_name, acc in class_accuracies.items():
                color = "normal"
                if acc >= 0.9:
                    color = "normal"
                elif acc >= 0.7:
                    color = "normal" 
                else:
                    color = "normal"
                
                st.metric(f"{class_name} Accuracy", f"{acc:.1%}")
            
            # Confusion matrix
            from collections import defaultdict
            confusion_matrix = defaultdict(lambda: defaultdict(int))
            
            for pred, true in zip(all_predictions, all_labels):
                pred_class = class_names[pred]
                true_class = class_names[true]
                confusion_matrix[true_class][pred_class] += 1
            
            st.subheader("🔄 Confusion Matrix")
            
            # Create confusion matrix visualization
            import pandas as pd
            
            matrix_data = []
            for true_class in class_names:
                row = []
                for pred_class in class_names:
                    row.append(confusion_matrix[true_class][pred_class])
                matrix_data.append(row)
            
            df = pd.DataFrame(matrix_data, index=class_names, columns=class_names)
            st.dataframe(df, use_container_width=True)
            
            # Save evaluation metrics
            eval_metrics = {
                "overall_accuracy": accuracy,
                "class_accuracies": class_accuracies,
                "total_samples": total,
                "average_confidence": float(np.mean(all_confidences)),
                "evaluation_date": datetime.now().isoformat(),
                "confusion_matrix": {true: dict(preds) for true, preds in confusion_matrix.items()}
            }
            
            with open(METRICS_PATH, "w") as f:
                json.dump(eval_metrics, f, indent=2)
            
            st.success("✅ Evaluation completed and metrics saved!")
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            LOGGER.exception("Evaluation error")

def ui_metrics():
    """Display saved metrics"""
    st.header("📊 Model Metrics")
    
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            
            st.subheader("🎯 Latest Evaluation Results")
            st.write(f"**Evaluation Date:** {metrics['evaluation_date']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
            col2.metric("Total Samples", metrics['total_samples'])
            col3.metric("Avg Confidence", f"{metrics['average_confidence']:.1%}")
            
            # Per-class metrics
            st.subheader("📈 Per-Class Performance")
            for class_name, accuracy in metrics['class_accuracies'].items():
                st.metric(f"{class_name}", f"{accuracy:.1%}")
            
            # Training history
            if HISTORY_PATH.exists():
                st.subheader("📉 Training History")
                
                with open(HISTORY_PATH, "r") as f:
                    history = json.load(f)
                
                # Plot training curves
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Loss curves
                ax1.plot(history["train_loss"], label="Train Loss", color='blue')
                ax1.plot(history["val_loss"], label="Val Loss", color='red')
                ax1.set_title("Loss Curves")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.legend()
                ax1.grid(True)
                
                # Accuracy curves
                ax2.plot(history["train_acc"], label="Train Acc", color='blue')
                ax2.plot(history["val_acc"], label="Val Acc", color='red')
                ax2.set_title("Accuracy Curves")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Accuracy (%)")
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
        except Exception as e:
            st.error(f"❌ Failed to load metrics: {e}")
    else:
        st.info("📊 No evaluation metrics found. Run model evaluation first.")

def ui_logs():
    """Display application logs"""
    st.header("📋 Application Logs")
    
    # Log level filter
    log_level = st.selectbox("Filter by level:", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Display recent logs from StreamlitHandler
    if hasattr(STREAMLIT_HANDLER, 'buffer') and STREAMLIT_HANDLER.buffer:
        logs = STREAMLIT_HANDLER.buffer
        
        # Filter logs
        if log_level != "ALL":
            logs = [log for log in logs if log_level in log]
        
        # Display logs
        if logs:
            # Show recent logs (last 50)
            recent_logs = logs[-50:]
            
            st.subheader(f"📝 Recent Logs ({len(recent_logs)})")
            
            # Create scrollable text area
            log_text = "\n".join(reversed(recent_logs))  # Most recent first
            st.text_area("Logs", log_text, height=400, key="log_display")
            
            # Download logs
            if st.button("📥 Download All Logs"):
                all_logs = "\n".join(STREAMLIT_HANDLER.buffer)
                st.download_button(
                    "Download logs.txt",
                    all_logs,
                    file_name=f"app_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("No logs found matching the selected level")
    else:
        st.info("No logs available")
    
    # File logs info
    if LOG_FILE.exists():
        file_size = LOG_FILE.stat().st_size
        st.info(f"📁 Log file: {LOG_FILE} ({file_size/1024:.1f} KB)")

def save_runtime_config(tcfg, appcfg):
    """Save runtime configuration"""
    try:
        config = {
            "train_config": asdict(tcfg),
            "app_config": asdict(appcfg),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(RUNTIME_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
            
    except (OSError, IOError) as e:
        if e.errno == 122:  # Disk quota exceeded
            # Silently skip saving when disk is full - this is not critical
            pass
        else:
            LOGGER.warning("Failed to save runtime config: %s", e)
    except Exception as e:
        LOGGER.warning("Failed to save runtime config: %s", e)

# ========== Main Application ==========
def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(f"📱 {APP_NAME}")
    st.markdown("---")
    
    # Initialize session state
    if 'predictor_loaded' not in st.session_state:
        st.session_state['predictor_loaded'] = False
    
    # Load configurations
    tcfg, appcfg = DEFAULT_TRAIN_CONFIG, DEFAULT_APP_CONFIG
    tcfg, appcfg = ui_sidebar(tcfg, appcfg)
    
    # Save runtime config
    save_runtime_config(tcfg, appcfg)
    
    # Main tabs
    tabs = st.tabs([
        "🔍 Predict", 
        "📊 Batch", 
        "✏️ Annotate", 
        "🎯 Train", 
        "📈 Evaluate", 
        "📊 Metrics", 
        "📋 Logs"
    ])
    
    with tabs[0]:
        ui_predict_single(appcfg, tcfg)
    
    with tabs[1]:
        ui_predict_batch(appcfg, tcfg)
    
    with tabs[2]:
        ui_annotate(appcfg, tcfg)
    
    with tabs[3]:
        ui_train(tcfg)
    
    with tabs[4]:
        ui_evaluate(tcfg)
    
    with tabs[5]:
        ui_metrics()
    
    with tabs[6]:
        ui_logs()

if __name__ == "__main__":
    main()
