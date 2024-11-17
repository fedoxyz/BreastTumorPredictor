import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from preprocessor import ImageDataset
from model import BreastTumorModel
from loss_funcs import CombinedLoss
from utils import load_config, split_dataset, calculate_iou

def train_model(config=load_config()):
    # Set device
    device = torch.device(config["train"]["device"])
    
    # Create dataset
    dataset = ImageDataset(
        config=config
    )
    
    # Split dataset
    train_idx, val_idx = split_dataset(dataset)
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["train"]["batch_size"], 
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True
    )
    

    # Initialize model
    model = BreastTumorModel(config).to(device)
   
    weight_bce = config["train"]["weight_bce"]
    weight_dice = config["train"]["weight_dice"]
    weight_iou = config["train"]["weight_iou"]
    weight_focal = config["train"]["weight_focal"]

    criterion_seg = CombinedLoss(weight_bce=weight_bce, weight_dice=weight_dice, weight_iou=weight_iou, weight_focal=weight_focal)
    criterion_class = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    # Training loop
    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        
        for batch_idx, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                class_output, seg_output = model(images)
                
                # Calculate losses
                loss_seg = criterion_seg(seg_output, masks)
                loss_class = criterion_class(class_output, labels)
                loss = loss_seg + config["train"]["class_weight"] * loss_class
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate IoU
            with torch.no_grad():
                pred_masks = (torch.sigmoid(seg_output) > 0.5).float()
                batch_iou = calculate_iou(pred_masks, masks)
                epoch_iou += batch_iou.item()
            
            epoch_loss += loss.item()
            
            if batch_idx % config["train"]["log_frequency"] == 0:
                print(f'Epoch {epoch+1}/{config["train"]["epochs"]} '
                      f'[{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\t'
                      f'IoU: {batch_iou.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                
                class_output, seg_output = model(images)
                
                # Calculate validation losses
                loss_seg = criterion_seg(seg_output, masks)
                loss_class = criterion_class(class_output, labels)
                loss = loss_seg + config["train"]["class_weight"] * loss_class
                
                val_loss += loss.item()
                
                # Calculate validation IoU
                pred_masks = (torch.sigmoid(seg_output) > 0.5).float()
                batch_iou = calculate_iou(pred_masks, masks)
                val_iou += batch_iou.item()
        
        # Calculate epoch metrics
        epoch_loss /= len(train_loader)
        epoch_iou /= len(train_loader)
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_ious.append(epoch_iou)
        val_ious.append(val_iou)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config["train"]["save_dir"], 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config["train"]["checkpoint_frequency"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config["train"]["save_dir"], f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f'Epoch {epoch+1}/{config["train"]["epochs"]} completed')
        print(f'Train Loss: {epoch_loss:.6f}, Train IoU: {epoch_iou:.6f}')
        print(f'Val Loss: {val_loss:.6f}, Val IoU: {val_iou:.6f}')
        
        # Save metrics to JSON after each epoch
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_ious': train_ious,
            'val_ious': val_ious
        }
        
        # Load previous metrics to append new ones
        json_path = os.path.join(config["train"]["save_dir"], 'training_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                existing_metrics = json.load(f)
                existing_metrics['train_losses'].extend(train_losses)
                existing_metrics['val_losses'].extend(val_losses)
                existing_metrics['train_ious'].extend(train_ious)
                existing_metrics['val_ious'].extend(val_ious)
        else:
            existing_metrics = metrics
        
        with open(json_path, 'w') as f:
            json.dump(existing_metrics, f)
