import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def dump_config(config, config_path="configs/config.yaml"):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=4)

def split_dataset(dataset, val_ratio=0.2):
    """Split the dataset into training and validation sets."""
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=42)
    return train_idx, val_idx
