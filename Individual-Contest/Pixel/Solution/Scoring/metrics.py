import json
from pathlib import Path 
from argparse import ArgumentParser
import os

import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_from_disk, load_dataset
from PIL import Image
import numpy as np
import math
from tqdm.auto import tqdm  # Progress bar
import random
import matplotlib.pyplot as plt

if os.environ.get('METRIC_PATH'):
    METRIC_PATH = os.environ.get("METRIC_PATH") + "/" 
else:
    METRIC_PATH = ""  # Fallback for local testing
H, W = 224, 224
MODEL_PATH = "/bohr/clip-vit-large-patch14-aft9/v1/clip-vit-large-patch14"
DATASET_PATH = METRIC_PATH + "reference_dataset"
MASK_PATH = "masks.jsonl"
SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETAIN_RATIO = 0.0625
SCORE_OUTPUT_FILE = "score.json"
MEAN_COLOR = (0, 0, 0)


parser = ArgumentParser()
parser.add_argument("--mask-file", default='masks.npy', type=str)
parser.add_argument("--debug", default=False, action='store_true')
args = parser.parse_args()


def write_error_score(error_message):
    """Write error score to JSON file"""
    error_json = {
        "status": False,
        "score": {
            "public_a": 0.0,
            "public_detail": {
                "Score": 0.0,
                "Accuracy": 0.0,
            },
            "private_b": 0.0,
            "private_detail": {
                "Score": 0.0,
                "Accuracy": 0.0,
            },
        },
        "msg": f"Error: {error_message}",
    }
    
    with open(SCORE_OUTPUT_FILE, 'w') as f:
        json.dump(error_json, f, indent=2)
    print(f"Error written to {SCORE_OUTPUT_FILE}: {error_message}")


def safe_load_masks(mask_file_path, expected_dataset_size):
    """
    Safely load and validate the masks file from contestants.
    
    Parameters:
        mask_file_path: Path to the masks file
        expected_dataset_size: Expected number of test cases
        
    Returns:
        dict: Validated masks dictionary or None if invalid
    """
    try:
        # Check if file exists
        if not os.path.exists(mask_file_path):
            write_error_score("Mask file not found.")
            return None
        
        # Check file size (prevent extremely large files)
        file_size = os.path.getsize(mask_file_path)
        max_file_size = 50 * 1024 * 1024  # 50MB limit
        if file_size > max_file_size:
            write_error_score("Mask file too large.")
            return None
        
        masks = {}
        
        # Load based on file extension
        if mask_file_path.endswith('.jsonl'):
            # Load JSONL format (one JSON object per line)
            try:
                with open(mask_file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():  # Skip empty lines
                            try:
                                data = json.loads(line.strip())
                                idx = data.get('idx')
                                coordinates = data.get('coordinates')
                                
                                if idx is None or coordinates is None:
                                    write_error_score("Invalid JSONL format.")
                                    return None
                                
                                masks[idx] = coordinates
                            except json.JSONDecodeError:
                                write_error_score("Invalid JSON in mask file.")
                                return None
            except Exception:
                write_error_score("Unable to load JSONL mask file.")
                return None
                
        # Validate it's a dictionary
        if not isinstance(masks, dict):
            write_error_score("Mask data must be a dictionary.")
            return None
        
        # Check number of entries
        if len(masks) != expected_dataset_size:
            # print(len(masks), expected_dataset_size)
            write_error_score("Incorrect number of mask entries.")
            return None
        
        # Validate each mask entry
        for idx, coordinates in masks.items():
            # Validate index
            if not isinstance(idx, (int, np.integer, str)):
                write_error_score("Invalid mask index format.")
                return None
            
            # Validate coordinates structure
            if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
                write_error_score("Invalid mask coordinate structure.")
                return None
            
            try:
                (top, left), (bottom, right) = coordinates
            except (ValueError, TypeError):
                write_error_score("Invalid mask coordinate format.")
                return None
            
            # Validate coordinate types and values
            coords = [top, left, bottom, right]
            for coord in coords:
                if not isinstance(coord, (int, np.integer)):
                    write_error_score("Mask coordinates must be integers.")
                    return None
                
                if not (0 <= coord <= 224):
                    write_error_score("Mask coordinates out of valid range.")
                    return None
            
            # Validate coordinate ordering
            if not (top < bottom and left < right):
                write_error_score("Invalid mask coordinate ordering.")
                return None
            
            # Validate area constraint
            crop_area = (bottom - top) * (right - left)
            max_area = RETAIN_RATIO * 224 * 224
            if crop_area > max_area:
                write_error_score("Mask area exceeds allowed limit.")
                return None
            
            # Additional security: prevent degenerate cases
            if crop_area <= 0:
                write_error_score("Invalid mask area.")
                return None
        
        return masks
        
    except Exception as e:
        write_error_score("Unexpected error loading mask file.")
        return None


def check_validity(coordinates):
    """
    Check if coordinates are valid according to the requirements.
    Returns True if valid, False otherwise.
    """
    try:
        # Check if coordinates is a tuple of two tuples
        if not hasattr(coordinates, '__iter__') or len(coordinates) != 2:
            return False
        
        (top, left), (bottom, right) = coordinates
        
        # Check if all coordinates are integers
        if not all(isinstance(coord, (int, np.integer)) for coord in [top, left, bottom, right]):
            return False
        
        # Check if coordinates are within image bounds
        # For slicing mask[top:bottom, left:right], valid ranges are:
        # top, left: [0, 223] (inclusive)
        # bottom, right: [1, 224] (inclusive) since we need top < bottom and left < right
        if not (0 <= top < 224 and 0 <= left < 224 and 1 <= bottom <= 224 and 1 <= right <= 224):
            return False
        
        # Check if top-left is actually top-left of bottom-right (proper ordering)
        if not (top < bottom and left < right):
            return False
        
        # Check that the crop area doesn't exceed RETAIN_RATIO
        crop_area = (bottom - top) * (right - left)
        max_area = RETAIN_RATIO * 224 * 224
        if crop_area > max_area:
            return False
        
        return True
    except Exception:
        return False

def generate_mask_from_coordinates(image, coordinates):
    """
    Generate a binary mask from crop coordinates.
    
    Parameters:
        image: PIL Image
        coordinates: tuple of ((top, left), (bottom, right))
    
    Returns:
        numpy array: Binary mask with 1s in the crop area
    """
    H, W = 224, 224  # Standard image size
    mask = np.zeros((H, W), dtype=np.int8)
    
    (top, left), (bottom, right) = coordinates
    mask[top:bottom, left:right] = 1
    
    return mask

def apply_mask_with_mean(image, mask, mean_rgb=MEAN_COLOR):
    """
    Apply arbitrary binary mask to image, replacing masked areas with mean values

    Parameters:
    - image: PIL Image (224x224)
    - mask: Binary numpy array or PIL Image (224x224) where 0 is the area to drop and 1 is the area to keep
    - mean_rgb: RGB mean values to use (default: from config)

    Returns: Modified PIL Image
    """
    # Convert images to numpy arrays
    img_array = np.array(image).copy()

    # Ensure mask is numpy array
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask.convert('L')) > 127  # Convert to binary
    else:
        mask_array = mask > 0

    # Reshape mask for broadcasting with RGB
    mask_3d = np.stack([mask_array] * 3, axis=2)

    # Convert mean values to 0-255 range
    mean_values = np.array([int(m * 255) for m in mean_rgb])
    # Apply mask - replace areas where mask is 0 (drop) with mean values, keep areas where mask is 1
    img_array = np.where(mask_3d, img_array, mean_values.reshape(1, 1, 3))

    return Image.fromarray(img_array.astype(np.uint8))


if __name__ == '__main__':
    try:
        # Load the animals dataset using the function from animal_dataset.py
        try:
            dataset = load_from_disk(DATASET_PATH)
            dataset = dataset[SPLIT]
        except Exception:
            write_error_score("Unable to load reference dataset.")
            exit(1)

        # Safely load and validate masks
        masks = safe_load_masks(MASK_PATH, len(dataset))
        if masks is None:
            exit(1)  # Error already written by safe_load_masks

        # Check validity of coordinates and report invalid ones
        invalid_coordinates = []
        valid_coordinates = 0
        for idx, coordinates in masks.items():
            if not check_validity(coordinates):
                invalid_coordinates.append(idx)
            else:
                valid_coordinates += 1
        
        if invalid_coordinates:
            print(f"Warning: Found {len(invalid_coordinates)} invalid coordinates (indices: {invalid_coordinates[:10]}{'...' if len(invalid_coordinates) > 10 else ''})")
            print(f"Invalid coordinates will be treated as incorrect predictions")
        print(f"Valid coordinates: {valid_coordinates}/{len(masks)}")

        #dataset = dataset.select(range(10)) # debug remove later

        # --- Step 1: Load Model and Processor ---
        print(f"Loading CLIP model and processor: {MODEL_PATH}...")
        try:
            model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
            processor = CLIPProcessor.from_pretrained(MODEL_PATH)
            model.eval()  # Set to evaluation mode
            print("Model and processor loaded successfully.")
        except Exception as e:
            write_error_score("Unable to load model.")
            exit(1)

        try:
            labels = sorted(list(set(dataset['name']))) + ['other']
            text_inputs = processor(text=labels, return_tensors="pt", padding=True).to(DEVICE)
        except Exception:
            write_error_score("Unable to process labels.")
            exit(1)

        # Map label names to indices for later comparison
        label_to_index = {label: i for i, label in enumerate(labels)}
        index_to_label = {i: label for label, i in label_to_index.items()}  # For mapping prediction back

        def predict_with_coordinates(image, coordinates):
            try:
                # Generate mask from coordinates
                mask = generate_mask_from_coordinates(image, coordinates)
                assert len(mask.shape) == 2

                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = apply_mask_with_mean(image, mask)
                image_processed = processor(images=image, return_tensors="pt").to(DEVICE)
                pixel_values = image_processed['pixel_values']
                outputs_full = model(pixel_values=pixel_values, **text_inputs)
                logits_full = outputs_full.logits_per_image  # Shape: (1, num_labels)
                predicted_index_full = logits_full.argmax(dim=-1).item()
                return predicted_index_full
            except Exception:
                # Return a random prediction if processing fails
                return len(labels) - 1  # Return 'other' class

        def get_accuracy(masks):
            try:
                with torch.no_grad():  # Disable gradient calculations for inference
                    correct = 0
                    for item in tqdm(dataset):
                        idx = item['idx']
                        if idx not in masks:
                            continue
                        coordinates = masks[idx]
                        
                        # Check coordinates validity - if invalid, mark as incorrect
                        if not check_validity(coordinates):
                            print(f"Invalid coordinates for item {idx}")
                            continue  # Skip this item, treating it as incorrect
                        
                        image = item['image']
                        true_label_label = item['name']  # This is now the animal class name

                        # Store true label for confusion matrix
                        true_label_idx = label_to_index[true_label_label]
                        if predict_with_coordinates(image, coordinates) == true_label_idx:
                            correct += 1
                return correct / len(masks)
            except Exception:
                return 0.0

        def get_accuracy_by_sets(masks):
            """Calculate accuracy for A set (smaller) and B set (larger) with 30:70 split"""
            try:
                # Set random seed for reproducible shuffling
                random.seed(42)
                
                with torch.no_grad():
                    correct_a = 0
                    correct_b = 0
                    total_a = 0
                    total_b = 0
                    
                    # First, collect all valid items that have masks
                    valid_items = []
                    for item in dataset:
                        idx = item['idx']
                        if idx in masks:
                            valid_items.append(item)
                    
                    # Group items by class name for stratified sampling
                    items_by_class = {}
                    for item in valid_items:
                        class_name = item['name']
                        if class_name not in items_by_class:
                            items_by_class[class_name] = []
                        items_by_class[class_name].append(item)
                    
                    # Stratified split: for each class, allocate 30% to A and 70% to B
                    set_a_items = []
                    set_b_items = []
                    
                    for class_name, class_items in items_by_class.items():
                        # Shuffle items within each class for random stratified sampling
                        random.shuffle(class_items)
                        
                        # Calculate split point for this class (30% to A, 70% to B)
                        split_point = int(len(class_items) * 0.3)
                        
                        # Ensure at least one item goes to each set if possible
                        if len(class_items) >= 2:
                            if split_point == 0:
                                split_point = 1
                            elif split_point == len(class_items):
                                split_point = len(class_items) - 1
                        
                        class_a_items = class_items[:split_point]
                        class_b_items = class_items[split_point:]
                        
                        set_a_items.extend(class_a_items)
                        set_b_items.extend(class_b_items)
                        
                        print(f"Class '{class_name}': {len(class_items)} total, {len(class_a_items)} to A, {len(class_b_items)} to B")
                    
                    print(f"Stratified split: Set A has {len(set_a_items)} items, Set B has {len(set_b_items)} items")
                    
                    # Verify class distribution
                    a_class_counts = {}
                    b_class_counts = {}
                    for item in set_a_items:
                        class_name = item['name']
                        a_class_counts[class_name] = a_class_counts.get(class_name, 0) + 1
                    for item in set_b_items:
                        class_name = item['name']
                        b_class_counts[class_name] = b_class_counts.get(class_name, 0) + 1
                    
                    print("Class distribution verification:")
                    for class_name in sorted(labels):
                        a_count = a_class_counts.get(class_name, 0)
                        b_count = b_class_counts.get(class_name, 0)
                        total_count = a_count + b_count
                        if total_count > 0:
                            a_ratio = a_count / total_count
                            b_ratio = b_count / total_count
                            print(f"  {class_name}: A={a_count} ({a_ratio:.1%}), B={b_count} ({b_ratio:.1%})")
                    
                    # Process Set A
                    for item in tqdm(set_a_items, desc="Processing Set A"):
                        idx = item['idx']
                        coordinates = masks[idx]
                        
                        # Check coordinates validity - if invalid, mark as incorrect
                        if not check_validity(coordinates):
                            total_a += 1
                            continue  # Skip prediction, treating as incorrect
                        
                        image = item['image']
                        true_label_label = item['name']  # This is now the animal class name
                        true_label_idx = label_to_index[true_label_label]
                        
                        # Get prediction on masked image
                        masked_pred_idx = predict_with_coordinates(image, coordinates)
                        
                        is_correct = masked_pred_idx == true_label_idx
                        
                        total_a += 1
                        if is_correct:
                            correct_a += 1
                    
                    # Process Set B
                    for item in tqdm(set_b_items, desc="Processing Set B"):
                        idx = item['idx']
                        coordinates = masks[idx]
                        
                        # Check coordinates validity - if invalid, mark as incorrect
                        if not check_validity(coordinates):
                            total_b += 1
                            continue  # Skip prediction, treating as incorrect
                        
                        image = item['image']
                        true_label_label = item['name']  # This is now the animal class name
                        true_label_idx = label_to_index[true_label_label]
                        
                        # Get prediction on masked image
                        masked_pred_idx = predict_with_coordinates(image, coordinates)
                        
                        is_correct = masked_pred_idx == true_label_idx
                        
                        total_b += 1
                        if is_correct:
                            correct_b += 1
                    
                    accuracy_a = correct_a / total_a if total_a > 0 else 0
                    accuracy_b = correct_b / total_b if total_b > 0 else 0
                    
                    print(f"Set A (30%): {total_a} samples, accuracy: {accuracy_a:.4f}")
                    print(f"Set B (70%): {total_b} samples, accuracy: {accuracy_b:.4f}")
                    
                    return accuracy_a, accuracy_b
            except Exception:
                return 0.0, 0.0

        def predict_without_mask(image):
            """Predict on original image without mask"""
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_processed = processor(images=image, return_tensors="pt").to(DEVICE)
                pixel_values = image_processed['pixel_values']
                outputs_full = model(pixel_values=pixel_values, **text_inputs)
                logits_full = outputs_full.logits_per_image
                predicted_index_full = logits_full.argmax(dim=-1).item()
                return predicted_index_full
            except Exception:
                return len(labels) - 1  # Return 'other' class

        # Calculate accuracies for A and B sets
        accuracy_a, accuracy_b = get_accuracy_by_sets(masks)

        
        score_a = accuracy_a
        score_b = accuracy_b
        
        # Ensure scores are within valid bounds [0.0, 1.0]
        if not (0.0 <= score_a <= 1.0) or not isinstance(score_a, (int, float)) or math.isnan(score_a) or math.isinf(score_a):
            score_a = 0.0
        if not (0.0 <= score_b <= 1.0) or not isinstance(score_b, (int, float)) or math.isnan(score_b) or math.isinf(score_b):
            score_b = 0.0
        
        print(f"Score A: {score_a}, Score B: {score_b}")
        
        #----------calculate the score on the leaderboard------------#
        score = {
            "public_a": score_a,
            "public_detail": {
                "Score": score_a,
                "Accuracy": accuracy_a,
            },
            "private_b": score_b,
            "private_detail": {
                "Score": score_b,
                "Accuracy": accuracy_b,
            },
        }
        
        ret_json = {
            "status": True,
            "score": score,
            "msg": "Success!",
        }
        
        # Save the score to JSON file
        with open(SCORE_OUTPUT_FILE, 'w') as f:
            json.dump(ret_json, f, indent=2)
        
        print(f"Score saved to {SCORE_OUTPUT_FILE}")

    except Exception as e:
        # Catch any unexpected errors during execution
        write_error_score("Unexpected error during evaluation.")
        exit(1)
