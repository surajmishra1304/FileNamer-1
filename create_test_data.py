#!/usr/bin/env python3
"""
Create synthetic test images for iPhone part detection testing.
This script creates simple synthetic images to test the model functionality.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_synthetic_iphone_part(image_type="genuine", size=(224, 224)):
    """Create a synthetic iPhone part image"""
    
    # Create base image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    if image_type == "genuine":
        # Create genuine part - clean, organized appearance
        # Add some geometric shapes to simulate genuine part
        draw.rectangle([20, 20, 200, 200], fill=(50, 50, 50), outline=(0, 0, 0))
        draw.rectangle([30, 30, 190, 190], fill=(100, 100, 100), outline=(255, 255, 255))
        
        # Add some "circuit" patterns
        for i in range(5):
            x = 40 + i * 30
            draw.line([x, 40, x, 180], fill=(255, 255, 0), width=2)
            draw.line([40, 40 + i * 30, 180, 40 + i * 30], fill=(255, 255, 0), width=2)
        
        # Add some small rectangles to simulate components
        for i in range(3):
            for j in range(3):
                x = 50 + i * 40
                y = 50 + j * 40
                draw.rectangle([x, y, x+20, y+15], fill=(0, 255, 0))
                
    else:  # unknown part
        # Create unknown part - irregular, different appearance
        # Add irregular shapes and different colors
        draw.rectangle([15, 25, 205, 195], fill=(80, 40, 40), outline=(255, 0, 0))
        
        # Add some irregular patterns
        for i in range(8):
            x = random.randint(20, 180)
            y = random.randint(20, 180)
            w = random.randint(10, 30)
            h = random.randint(10, 30)
            color = (random.randint(100, 255), random.randint(0, 100), random.randint(0, 100))
            draw.rectangle([x, y, x+w, y+h], fill=color)
        
        # Add some random circles
        for i in range(5):
            x = random.randint(30, 170)
            y = random.randint(30, 170)
            r = random.randint(5, 15)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(100, 255))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
    
    return img

def create_test_dataset():
    """Create a complete test dataset"""
    
    # Dataset structure
    splits = ['train', 'val', 'test']
    classes = ['Genuine', 'Unknown Part']
    
    # Number of images per split
    counts = {
        'train': 20,  # 20 images per class for training
        'val': 10,    # 10 images per class for validation  
        'test': 8     # 8 images per class for testing
    }
    
    print("Creating synthetic iPhone part dataset...")
    
    for split in splits:
        for class_name in classes:
            dir_path = f"dataset/{split}/{class_name}"
            os.makedirs(dir_path, exist_ok=True)
            
            image_type = "genuine" if class_name == "Genuine" else "unknown"
            
            for i in range(counts[split]):
                # Add some variation to each image
                random.seed(i * 100 + ord(split[0]) + ord(class_name[0]))
                
                img = create_synthetic_iphone_part(image_type)
                
                # Add some noise for variation
                img_array = np.array(img)
                noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
                img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # Save image
                filename = f"{class_name.lower().replace(' ', '_')}_{i:03d}.jpg"
                img.save(os.path.join(dir_path, filename), 'JPEG', quality=85)
            
            print(f"Created {counts[split]} images for {split}/{class_name}")
    
    print(f"\nDataset creation complete!")
    print(f"Total images: {sum(counts.values()) * len(classes)}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    for split in splits:
        for class_name in classes:
            dir_path = f"dataset/{split}/{class_name}"
            count = len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
            print(f"  {split}/{class_name}: {count} images")

if __name__ == "__main__":
    create_test_dataset()