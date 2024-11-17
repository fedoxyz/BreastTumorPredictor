import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_config(config_path='configs/config.yaml'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_config_path = os.path.join(base_dir, config_path)
    
    if not os.path.exists(absolute_config_path):
        raise FileNotFoundError(f"File not found: {absolute_config_path}")
    
    with open(absolute_config_path, "r") as file:
        config = yaml.safe_load(file)
        
        # Resolve relative paths in the config
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if "dir" in sub_key or "path" in sub_key:
                        config[key][sub_key] = os.path.join(base_dir, sub_value)
            elif "dir" in key or "path" in key:
                config[key] = os.path.join(base_dir, value)
                
    return config

def dump_config(config, config_path="../configs/config.yaml"):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=4)

def split_dataset(dataset, val_ratio=0.2):
    """Split the dataset into training and validation sets."""
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
    return train_idx, val_idx

def calculate_iou(pred_masks, gt_masks):
    # Ensure both masks are of the same shape
    assert pred_masks.shape == gt_masks.shape, "Predicted and ground truth masks must have the same shape."

    # Calculate intersection and union
    intersection = np.logical_and(pred_masks, gt_masks)
    union = np.logical_or(pred_masks, gt_masks)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou
