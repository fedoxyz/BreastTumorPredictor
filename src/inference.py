import torch
import cv2
import numpy as np
import yaml
import os
from torchvision import transforms
from PIL import Image
from model import BreastTumorModel

class TumorPredictor:
    def __init__(self, config_path="configs/config.yaml", model_path=None):
        """
        Initialize the tumor predictor
        
        Args:
            config_path (str): Path to the configuration file
            model_path (str): Path to the trained model weights
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(self.config["inference"]["device"])
        
        # Initialize model
        self.model = BreastTumorModel(self.config).to(self.device)
        
        # Load model weights
        if model_path is None:
            model_path = os.path.join(self.config["train"]["save_dir"], "best_model.pth")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.config["data"]["img_size"], self.config["data"]["img_size"])),
            transforms.ToTensor(),
        ])
        
        # Class labels
        self.class_labels = ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']

    def preprocess_image(self, image):
        """
        Preprocess a single image for inference
        
        Args:
            image: PIL Image, numpy array, or path to image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image, numpy array, or path to image")
            
        return self.transform(image).unsqueeze(0)

    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: PIL Image, numpy array, or path to image
            
        Returns:
            dict: Prediction results including class, confidence, and segmentation mask
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            class_output, seg_output = self.model(image_tensor)
            
            # Get class prediction
            class_probs = torch.softmax(class_output, dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            class_confidence = class_probs[0][predicted_class].item()
            class_probabilities = class_probs[0].cpu().numpy()
            
            # Get segmentation mask
            seg_mask = torch.sigmoid(seg_output) > 0.5
            seg_mask = seg_mask.cpu().numpy()[0, 0]
        
        return {
            'class_label': self.class_labels[predicted_class],
            'class_idx': predicted_class,
            'class_confidence': class_confidence,
            'class_probabilities': class_probabilities,
            'segmentation_mask': seg_mask,
            'raw_segmentation': torch.sigmoid(seg_output).cpu().numpy()[0, 0]
        }

    def predict_batch(self, images):
        """
        Process a batch of images
        
        Args:
            images: List of images (PIL Images, numpy arrays, or paths)
            
        Returns:
            list: List of prediction results for all images
        """
        batch_tensors = []
        for image in images:
            batch_tensors.append(self.preprocess_image(image))
        
        batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
        
        results = []
        with torch.no_grad():
            class_output, seg_output = self.model(batch_tensor)
            
            # Get predictions for each image in batch
            class_probs = torch.softmax(class_output, dim=1)
            predicted_classes = torch.argmax(class_probs, dim=1)
            seg_masks = (torch.sigmoid(seg_output) > 0.5).cpu().numpy()
            
            for i in range(len(images)):
                results.append({
                    'class_label': self.class_labels[predicted_classes[i].item()],
                    'class_idx': predicted_classes[i].item(),
                    'class_confidence': class_probs[i][predicted_classes[i]].item(),
                    'class_probabilities': class_probs[i].cpu().numpy(),
                    'segmentation_mask': seg_masks[i, 0],
                    'raw_segmentation': torch.sigmoid(seg_output[i]).cpu().numpy()[0]
                })
        
        return results

def main():
    # Example usage
    predictor = TumorPredictor(
        config_path="configs/config.yaml",
        model_path="checkpoints/best_model.pth"
    )
    
    # Single image prediction
    result = predictor.predict("path/to/test/image.jpg")
    print(f"Predicted class: {result['class_label']}")
    print(f"Confidence: {result['class_confidence']:.2f}")

if __name__ == "__main__":
    main()
