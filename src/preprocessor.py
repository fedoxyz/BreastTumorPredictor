import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, config):
        self.img_size = config["data"]["img_size"]
        self.data_dir = config["data"]["dir"]
        self.config = config
        self.labels = []
        self.image_files = []
        
        # Iterate over all class directories inside data_dir
        class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        for class_label in class_dirs:
            class_dir = os.path.join(self.data_dir, class_label)
            image_dir = os.path.join(class_dir, 'images')
            mask_dir = os.path.join(class_dir, 'masks')
  
            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                continue  # Skip if any directory is missing

            class_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
     
            for file_name in class_files:
                # Append full path relative to class directory
                self.image_files.append(os.path.join(class_label, 'image', file_name))
                self.labels.append(int(class_label))  # Store label as integer
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image and mask paths based on class label
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = os.path.join(self.data_dir, self.image_files[idx].replace('image/', 'masks/').replace('.jpg', '_mask.jpg'))

        # Ensure correct path construction
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = image_path.replace('image/', 'masks/').replace('.jpg', '_mask.jpg')

        image = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Error loading image or mask: {image_path}, {mask_path}")

        # Apply augmentations if configured
        if self.config["data"]["augmentation"]["flip"]:
            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        if self.config["data"]["augmentation"]["rotation"]:
            angle = np.random.uniform(-self.config["data"]["augmentation"]["rotation"], self.config["data"]["augmentation"]["rotation"])
            image = self.rotate_image(image, angle)
            mask = self.rotate_image(mask, angle, is_mask=True)

        # Resize image and mask
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0

        # Convert mask to binary
        mask = (mask > 0).astype(np.float32)

        # Transpose dimensions to match PyTorch expectations (C x H x W)
        image = image.transpose(2, 0, 1)
        mask = np.expand_dims(mask, axis=0)

        label = self.labels[idx]
        return image, mask, label

    def rotate_image(self, image, angle, is_mask=False):
        """Rotate image by a specified angle."""
        # Rotate around the center
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        if not is_mask:
            # If it's not a mask, apply smoothing (optional)
            rotated_image = cv2.GaussianBlur(rotated_image, (5, 5), 0)
        return rotated_image
