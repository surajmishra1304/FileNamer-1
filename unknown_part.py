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
FILE_HANDLER = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
FILE_HANDLER.setLevel(logging.DEBUG)
FILE_HANDLER.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
LOGGER.addHandler(FILE_HANDLER)

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
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.7),
            transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            num_workers=self.config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers
        )
        
        # Save class names
        class_names = train_dataset.classes
        with open(LABELS_PATH, "w") as f:
            json.dump(class_names, f)
        LOGGER.info("Saved class names: %s", class_names)
        
        # Create model
        model = ModelFactory.create_backbone(
            self.config.model_name, 
            len(class_names), 
            self.config.freeze_backbone
        )
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
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
            
            scheduler.step()
        
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
            # Load class names
            if not LABELS_PATH.exists():
                raise FileNotFoundError("Class names file not found")
            
            with open(LABELS_PATH, "r") as f:
                self.class_names = json.load(f)
            
            # Create model architecture
            self.model = ModelFactory.create_backbone("resnet50", len(self.class_names), False)
            
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
                LOGGER.error("Model file not found: %s", model_path)
                return False
                
        except Exception as e:
            LOGGER.error("Failed to load model: %s", e)
            return False
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Predict class and confidence for a single image"""
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
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
        pass
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(opencv_image)
            return extracted_text
        except Exception as e:
            LOGGER.error(f"OCR extraction failed: {e}")
            return ""
    
    def detect_service_status(self, image: Image.Image) -> Tuple[str, float, str]:
        """Detect if parts show 'Service' or 'Unknown Part' in iPhone service history"""
        
        extracted_text = self.extract_text_from_image(image)
        
        # Look for "Parts and Service History" section
        if "parts and service history" not in extracted_text.lower():
            return "No Service History Found", 0.0, extracted_text
        
        # Count occurrences of "Service" and "Unknown Part"
        service_count = extracted_text.lower().count("service")
        unknown_part_count = extracted_text.lower().count("unknown part")
        
        # Determine prediction based on counts
        if unknown_part_count > 0:
            confidence = min(0.9, (unknown_part_count * 0.3) + 0.6)
            return "Unknown Part", confidence, extracted_text
        elif service_count > 2:  # More than 2 to account for the header
            confidence = min(0.9, (service_count * 0.15) + 0.5)
            return "Service", confidence, extracted_text
        else:
            return "Unclear", 0.3, extracted_text

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
    
    predictor = Predictor()
    
    # Load model
    if st.button("Load Model", key="load_single"):
        with st.spinner("Loading model..."):
            if predictor.load_model():
                st.success("✅ Model loaded successfully")
                st.session_state['predictor_loaded'] = True
            else:
                st.error("❌ Failed to load model. Train a model first.")
                return
    
    if not st.session_state.get('predictor_loaded', False):
        st.info("👆 Click 'Load Model' to start predictions")
        return
    
    # Image input options
    input_method = st.radio("Choose input method:", ["Upload Image", "Image URL"])
    
    image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'webp'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        url = st.text_input("Enter image URL:")
        if url and st.button("Fetch Image"):
            try:
                fetcher = ImageFetcher(appcfg)
                with st.spinner("Fetching image..."):
                    image = fetcher.fetch(url)
                st.success("✅ Image fetched successfully")
            except Exception as e:
                st.error(f"❌ Failed to fetch image: {e}")
    
    # Display and predict
    if image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Input Image", use_column_width=True)
        
        with col2:
            if st.button("🔮 Predict", key="predict_single"):
                try:
                    if not predictor.load_model():  # Ensure model is loaded
                        st.error("Model not loaded")
                        return
                    
                    with st.spinner("Predicting..."):
                        predicted_class, confidence = predictor.predict(image)
                    
                    # Display results
                    st.subheader("🎯 Prediction Results")
                    
                    # Color-coded result
                    if predicted_class == "Unknown Part":
                        st.error(f"🚨 **{predicted_class}**")
                    else:
                        st.success(f"✅ **{predicted_class}**")
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Confidence bar
                    st.progress(confidence)
                    
                    if confidence < 0.5:
                        st.warning("⚠️ Low confidence prediction")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    LOGGER.exception("Prediction error")

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
        
        predictor = Predictor()
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
                
                # Predict
                predicted_class, confidence = predictor.predict(image)
                
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
    """iPhone Service History Analysis Tool - Multiple URLs"""
    st.header("📱 iPhone Service History Analysis")
    
    st.info("📋 Paste your iPhone service history image URLs below (one per line) to analyze for Service vs Unknown Part detection")
    
    # Multiple URL input
    url_text = st.text_area(
        "Paste iPhone service history image URLs (one per line):", 
        height=120, 
        placeholder="https://s3n.cashify.in/public/logistics-integration/images/MPMQD10707255_1702102878563.jpg\nhttps://example.com/iphone_service_history2.jpg\n...",
        key="annotate_url_text_area"
    )
    
    urls = [u.strip() for u in url_text.splitlines() if u.strip()]
    
    if not urls:
        st.markdown("""
        ### 📱 How to use:
        1. **Paste URLs**: Copy your iPhone service history image URLs above (one per line)
        2. **Analyze**: Each image will be analyzed for "Service" vs "Unknown Part" detection
        3. **Provide Feedback**: Confirm correct predictions or correct wrong ones
        4. **Batch Process**: Use "Analyze All Images" for processing multiple URLs at once
        5. **Track Performance**: View accuracy metrics and confusion matrix
        """)
        return
    
    st.success(f"✅ Found {len(urls)} URLs ready for analysis")
    
    # Image navigation
    if len(urls) > 1:
        idx = st.number_input(
            "Select image to analyze:", 
            min_value=0, 
            max_value=len(urls)-1, 
            value=0, 
            step=1, 
            key="annotate_idx_number_input"
        )
    else:
        idx = 0
    
    url = urls[idx]
    st.write(f"**Image {idx+1}/{len(urls)}:** {url}")
    
    # Fetch and analyze image
    try:
        fetcher = ImageFetcher(appcfg)
        with st.spinner("Loading and analyzing image..."):
            image = fetcher.fetch(url)
            
            # Analyze service history immediately
            detector = iPhoneServiceHistoryDetector()
            predicted_class, confidence, extracted_text = detector.detect_service_status(image)
            
    except Exception as e:
        st.error(f"❌ Failed to load image: {e}")
        return
    
    # Analysis interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption=f"iPhone Service History - Image {idx+1}/{len(urls)}", use_column_width=True)
        
        # Show OCR results
        with st.expander("🔍 OCR Extracted Text", expanded=False):
            st.text_area("Raw OCR Output:", extracted_text, height=200, key=f"ocr_text_{idx}")
    
    with col2:
        st.subheader("🎯 Analysis Results")
        
        # Display prediction
        if predicted_class == "Unknown Part":
            st.error(f"🚨 **{predicted_class}**")
        elif predicted_class == "Service":
            st.success(f"✅ **{predicted_class}**")
        else:
            st.warning(f"⚠️ **{predicted_class}**")
        
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)
        
        # User feedback section
        st.subheader("📝 Provide Feedback")
        st.write("Help improve accuracy:")
        
        # Feedback buttons
        col3, col4 = st.columns(2)
        with col3:
            if st.button("✅ Correct", key=f"correct_{idx}"):
                feedback_manager = FeedbackManager()
                feedback_manager.save_feedback(url, predicted_class, confidence, True, predicted_class, extracted_text)
                st.success("Feedback saved!")
        
        with col4:
            if st.button("❌ Incorrect", key=f"incorrect_{idx}"):
                st.session_state[f'show_correction_{idx}'] = True
        
        # Correction interface
        if st.session_state.get(f'show_correction_{idx}', False):
            correct_class = st.selectbox(
                "What is the correct classification?",
                ["Service", "Unknown Part", "No Service History Found", "Unclear"],
                key=f"correct_class_{idx}"
            )
            if st.button("Submit Correction", key=f"submit_{idx}"):
                feedback_manager = FeedbackManager()
                feedback_manager.save_feedback(url, predicted_class, confidence, False, correct_class, extracted_text)
                st.success(f"Correction saved: {correct_class}")
                st.session_state[f'show_correction_{idx}'] = False
        
        # Navigation for multiple images
        if len(urls) > 1:
            st.subheader("🔄 Navigation")
            col5, col6 = st.columns(2)
            
            with col5:
                if st.button("⬅️ Previous", disabled=idx==0, key=f"prev_{idx}"):
                    st.session_state['annotate_idx_number_input'] = max(0, idx-1)
                    st.rerun()
            
            with col6:
                if st.button("➡️ Next", disabled=idx==len(urls)-1, key=f"next_{idx}"):
                    st.session_state['annotate_idx_number_input'] = min(len(urls)-1, idx+1)
                    st.rerun()

    # Batch analysis summary
    if urls and len(urls) > 1:
        st.subheader("📊 Batch Analysis Summary")
        
        if st.button("🚀 Analyze All Images", key="analyze_all"):
            detector = iPhoneServiceHistoryDetector()
            feedback_manager = FeedbackManager()
            fetcher = ImageFetcher(appcfg)
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, batch_url in enumerate(urls):
                try:
                    status_text.text(f"Analyzing image {i+1}/{len(urls)}...")
                    
                    # Fetch and analyze
                    batch_image = fetcher.fetch(batch_url)
                    pred_class, conf, ocr_text = detector.detect_service_status(batch_image)
                    
                    results.append({
                        "Image": f"{i+1}/{len(urls)}",
                        "URL": batch_url[:50] + "..." if len(batch_url) > 50 else batch_url,
                        "Prediction": pred_class,
                        "Confidence": f"{conf:.1%}",
                        "Status": "✅ Success"
                    })
                    
                    # Auto-save high confidence predictions
                    if conf > 0.7:
                        feedback_manager.save_feedback(batch_url, pred_class, conf, True, pred_class, ocr_text)
                    
                except Exception as e:
                    results.append({
                        "Image": f"{i+1}/{len(urls)}",
                        "URL": batch_url[:50] + "..." if len(batch_url) > 50 else batch_url,
                        "Prediction": "Error",
                        "Confidence": "0%",
                        "Status": f"❌ {str(e)[:30]}"
                    })
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("✅ Batch analysis completed!")
            
            # Display results
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_column_width=True)
            
            # Summary stats
            successful = len([r for r in results if r["Status"] == "✅ Success"])
            unknown_parts = len([r for r in results if r["Prediction"] == "Unknown Part"])
            service_only = len([r for r in results if r["Prediction"] == "Service"])
            
            col7, col8, col9, col10 = st.columns(4)
            col7.metric("Total Analyzed", len(urls))
            col8.metric("Successful", successful)
            col9.metric("Unknown Parts", unknown_parts)
            col10.metric("Service Only", service_only)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis Results",
                data=csv,
                file_name=f"iphone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Show feedback summary
    if st.button("📈 View Feedback Summary"):
        feedback_manager = FeedbackManager()
        stats = feedback_manager.get_feedback_stats()
        
        col11, col12, col13 = st.columns(3)
        col11.metric("Total Feedback", stats['total_feedback'])
        col12.metric("Correct Predictions", stats['correct_predictions'])
        col13.metric("Accuracy", f"{stats['accuracy']:.1%}")
        
        if stats['total_feedback'] > 0:
            st.write("**Prediction Distribution:**")
            for class_name, count in stats['class_distribution'].items():
                st.write(f"- {class_name}: {count}")

def organize_annotations_to_dataset():
    """Organize annotated data into training dataset structure"""
    if not ANNOTATIONS_CSV.exists():
        return
    
    try:
        # Create dataset directories
        dataset_dir = Path(DEFAULT_TRAIN_CONFIG.data_dir)
        for split in ["train", "val"]:
            for class_name in ["Unknown Part", "Genuine"]:
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

def ui_train(tcfg: TrainConfig):
    """Training UI"""
    st.header("🎯 Model Training")
    
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
                for i, (bar, class_name) in enumerate(zip(bars, classes)):
                    if class_name == "Unknown Part":
                        bar.set_color('#ff4444')
                    else:
                        bar.set_color('#44ff44')
                
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
        predictor = Predictor()
        
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
                predictor.model.eval()
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(predictor.device)
                        outputs = predictor.model(images)
                        
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
