import torch
import torch.nn as nn
from torchvision import models
from sam2.build_sam import build_sam2
import subprocess
import os
import torch.nn.functional as F
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck



class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SAM2CustomDecoder(nn.Module):
    def __init__(self, sam2_encoder: ImageEncoder, device):
        super().__init__()
        self.device = device
        self.sam2_encoder = sam2_encoder.to(self.device)

        # Decoder blocks to upsample and refine the segmentation mask
        self.decoder = nn.ModuleList([
            # Block 1: Handles the highest resolution feature map
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0).to(device),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0).to(device),
                LayerNorm2d(128).to(device),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0).to(device),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0).to(device),
                LayerNorm2d(64).to(device),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0).to(device),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0).to(device),
                LayerNorm2d(32).to(device),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ),
            # Block 4
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0).to(device),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0).to(device),
                LayerNorm2d(16).to(device),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            ),
            # Final 1x1 conv: 16x512x512 -> 1x512x512
            nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0).to(device),
                nn.Sigmoid()
            )
        ])   

    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Process through SAM encoder and neck
        encoder_output = self.sam2_encoder(x)
        
        # Extract the feature pyramid network (FPN) features
        features = encoder_output["backbone_fpn"]
        
        # Move features to the correct device
        features = [f.to(self.device) for f in features]
        
        # Use the highest resolution feature map as the starting point
        x = features[-1]  # Assuming the last feature map is the highest resolution
        
        # Reduce initial feature map to 256 channels if needed
        if x.shape[1] != 256:
            channel_adapter = nn.Conv2d(
                x.shape[1], 256, 
                kernel_size=1, 
                stride=1, 
                padding=0
            ).to(self.device)
            x = channel_adapter(x)
        
        # Decode back to segmentation mask
        for i, decoder_block in enumerate(self.decoder):
            # Progressively reduce channels through decoder blocks
            x = decoder_block(x)
            
            if i < len(features) - 1:
                # Merge or concatenate with the next feature map
                next_feature_map = features[-i - 2]
                
                # Ensure next_feature_map is interpolated to match x's spatial dimensions
                next_feature_map = F.interpolate(next_feature_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                # Reduce next feature map channels to match current x
                if next_feature_map.shape[1] != x.shape[1]:
                    channel_adapter = nn.Conv2d(
                        next_feature_map.shape[1], 
                        x.shape[1], 
                        kernel_size=1, 
                        stride=1, 
                        padding=0
                    ).to(self.device)
                    next_feature_map = channel_adapter(next_feature_map)
                
                x = x + next_feature_map
        
        # Ensure output is 512x512
        if x.shape[-1] != 512:
            x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        return x   


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, config, trial=False):
        super(Classifier, self).__init__()
        self.num_layers = trial.suggest_int("n_layers", 1, 3) if trial else config["model"]["num_layers"]
        self.classifier_layers = nn.ModuleList()

        for i in range(self.num_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 16, 256) if trial else config["model"][f"out_features_{i}"]
            if i == 0:
                self.classifier_layers.append(nn.Linear(input_dim, out_features))
            else:
                # Find the last linear layer to get its output features
                last_linear_layer = [layer for layer in self.classifier_layers if isinstance(layer, nn.Linear)][-1]
                self.classifier_layers.append(nn.Linear(last_linear_layer.out_features, out_features))
            self.classifier_layers.append(nn.ReLU())
            dropout_rate = trial.suggest_float("dropout_l{}".format(i), 0.15, 0.4) if trial else config["model"][f"dropout_rate_{i}"]
            self.classifier_layers.append(nn.Dropout(dropout_rate))

        self.classifier_layers.append(nn.Linear(out_features, num_classes))

    def forward(self, x):
        for layer in self.classifier_layers:
            x = layer(x)
        return x

class BreastTumorModel(nn.Module):
    def __init__(self, config, trial=False):
        super(BreastTumorModel, self).__init__()

        device = config["train"]["device"]
        
        # Define number of classes
        num_classes = config["data"]["num_classes"] 
        
        # Set up SAM-2 model
        checkpoint = config["seg_model"]["path"]
        download_script = config["seg_model"]["download_file_path"]
        model_cfg = config["seg_model"]["config"]
        
        # Download SAM-2 checkpoint if not exists
        if not os.path.exists(checkpoint):
            print(f"Checkpoint file not found. Running {download_script} to download it...")
            try:
                subprocess.run(["bash", download_script], check=True)
                print("Checkpoint file downloaded successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download checkpoint file. Error: {e}")
                raise
        # Initialize SAM-2 model
        print(f"sam2 model cfg - {model_cfg}")
        sam2_model = build_sam2(model_cfg, checkpoint)
        
        # Use the SAM 2 encoder for feature extraction
        self.sam2_encoder = sam2_model.image_encoder.to(device)
        
        # Create custom decoder using SAM-2 encoder
        self.sam2_decoder = SAM2CustomDecoder(sam2_encoder=self.sam2_encoder, device=device)
        
        # Classifier for tumor classification
        self.classifier = Classifier(
            input_dim=self.calculate_input_dim(device),
            num_classes=num_classes,
            config=config,
            trial=trial
        )

        self.classifier = self.classifier.to(device)
    
    def calculate_input_dim(self, device):
        # Process a dummy input to get the shape of the feature maps
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        encoder_output = self.sam2_encoder(dummy_input)
        features = encoder_output["backbone_fpn"]
        classification_features = features[-1]
        
        classification_features = classification_features.reshape(classification_features.size(0), -1)
        
        # Return the flattened dimension
        return classification_features.shape[1]


    def forward(self, x):
        # Process through SAM encoder and neck
        encoder_output = self.sam2_encoder(x)
        
        # Extract the feature pyramid network (FPN) features
        features = encoder_output["backbone_fpn"]
        
        # Extract the highest resolution feature map for classification
        classification_features = features[-1]
        
        classification_features = classification_features.reshape(classification_features.size(0), -1)

        # Classification output
        class_output = self.classifier(classification_features)
        
        # Segmentation output using custom decoder
        segmentation_output = self.sam2_decoder(x)
        
        return class_output, segmentation_output

    def count_parameters(self):
        """Count total and trainable parameters in the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

