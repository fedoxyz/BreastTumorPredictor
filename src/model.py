import torch
import torch.nn as nn
from torchvision import models
from sam2.build_sam import build_sam2
import subprocess
import os
from torch import F

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self._setup_feature_extractor_training(config["feature_model"])

    def _setup_feature_extractor_training(self, config):
        # Helper function to freeze or unfreeze layers
        def set_requires_grad(module, requires_grad):
            for param in module.parameters():
                param.requires_grad = requires_grad
        
        # First, freeze everything
        set_requires_grad(self.feature_extractor, False)
        
        # Unfreeze specific parts based on config
        if config["train_layer4"]:
            set_requires_grad(self.feature_extractor.layer4, True)
        
        if config["train_bn"]:
            for m in self.feature_extractor.modules():
                if isinstance(m, (nn.BatchNorm2d,)):
                    set_requires_grad(m, True)
        
        # Unfreeze last n layers
        if config["train_last_n_layers"] > 0:
            layers_to_train = []
            for m in self.feature_extractor.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                    layers_to_train.append(m)
            
            # Get the last n layers
            for layer in layers_to_train[-config["train_last_n_layers"]:]:
                set_requires_grad(layer, True)

    def forward(self, x):
        return self.model(x)


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
    def __init__(self, sam2_encoder, sam2_preprocess):
        super().__init__()
        self.sam2_encoder = sam2_encoder
        self.sam2_preprocess = sam2_preprocess
        
        # Resolution adapter layer (512 -> 1024)
        self.resolution_adapter = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            LayerNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        # Freeze the encoder except for the last 6 layers (neck)
        total_params = len(list(self.sam2_encoder.parameters()))
        for layer_no, param in enumerate(self.sam2_encoder.parameters()):
            if layer_no > (total_params - 6):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.decoder = nn.ModuleList([
            # Block 1: 256x32x32 -> 128x64x64
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(128),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ),
            
            # Block 2: 128x64x64 -> 64x128x128
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(64),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ),
            
            # Block 3: 64x128x128 -> 32x256x256
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(32),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ),
            
            # Block 4: 32x256x256 -> 16x512x512
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
                LayerNorm2d(16),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            ),
            
            # Final 1x1 conv: 16x512x512 -> 1x512x512
            nn.Sequential(
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Adapt resolution from 512x512 to 1024x1024
        x = self.resolution_adapter(x)
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # Process through SAM encoder
        x = self.sam2_preprocess(x)
        features = self.sam2_encoder(x)
        
        # Decode back to segmentation mask
        for decoder_block in self.decoder:
            features = decoder_block(features)
        
        # Ensure output is 512x512
        if features.shape[-1] != 512:
            features = F.interpolate(features, size=(512, 512), mode='bilinear', align_corners=False)
        
        return features

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
                self.classifier_layers.append(nn.Linear(self.classifier_layers[-1].out_features, out_features))
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
        
        # Define number of classes
        num_classes = config["data"]["num_classes"] 
        
        self.feature_extractor = FeatureExtractor(config)
        
        feature_dim = self.feature_extractor.feature_dim
        self.classifier = Classifier(feature_dim, num_classes, config, trial)

        # Set up SAM-2 model and custom decoder
        checkpoint = config["seg_model"]["path"]
        download_script = config["seg_model"]["download_file"]
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
        sam2_model = build_sam2(model_cfg, checkpoint)
        
        # Create custom decoder using SAM-2 encoder
        self.sam2_decoder = SAM2CustomDecoder(
            sam2_encoder=sam2_model.image_encoder,
            sam2_preprocess=sam2_model.preprocess
        )
        
    def forward(self, x):
        # Extract features for classification
        features = self.feature_extractor(x)
        
        # Classification output
        class_output = self.classifier(features)
        
        # Segmentation output using custom decoder
        segmentation_output = self.sam2_decoder(x)
        
        return class_output, segmentation_output

    def count_parameters(self):
        """Count total and trainable parameters in the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
