import torch
import torch.nn as nn

# Dice Loss Implementation
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Apply sigmoid if needed
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# IoU (Jaccard) Loss Implementation
def iou_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Apply sigmoid if needed
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss

# Combined Loss Function
class CombinedLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, weight_iou=1.0, weight_focal=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou
        self.focal_loss = FocalLoss()
        self.weight_focal = weight_focal

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = dice_loss(inputs, targets)
        iou = iou_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        # Weighted sum of losses
        combined_loss = (
            self.weight_bce * bce +
            self.weight_dice * dice +
            self.weight_iou * iou +
            self.weight_focal * focal
        )
        return combined_loss


