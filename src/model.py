import torch
import torch.nn as nn
from torchvision import models
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import subprocess
import os

class BreastTumorModel(nn.Module):
    def __init__(self, config):
        super(BreastTumorModel, self).__init__()
        
        # Define number of classes
        num_classes = config["data"]["num_classes"] 
        architecture = config["features_model"]["architecture"]

        if architecture == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=True)
            feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()  # Remove the final classification layer
        elif architecture == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=True)
            feature_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        super(BreastTumorModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # Assuming 5 classes
            nn.Softmax(dim=1)
        )

        checkpoint = config["seg_model"]["path"]
        download_script = config["seg_model"]["download_file"]
        model_cfg = config["seg_model"]["config"]
        
        if not os.path.exists(checkpoint):
            print(f"Checkpoint file not found. Running {download_script} to download it...")
            try:
                subprocess.run(["bash", download_script], check=True)
                print("Checkpoint file downloaded successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download checkpoint file. Error: {e}")
                raise

        self.sam2_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        
        # unfreeze SAM 2 model parameters for tuning
        for param in self.sam2_predictor.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Classification output
        class_output = self.classifier(features)
        
        # Segmentation output using SAM 2
        segmentation_output = self.sam2_predictor.predict(x)
        
        return class_output, segmentation_output
    
