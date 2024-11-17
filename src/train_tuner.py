import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessor import ImageDataset
from model import BreastTumorModel
from loss_funcs import CombinedLoss
from utils import load_config, dump_config, split_dataset
import optuna
from torchinfo import summary



def objective(trial):
    config = load_config()
    device = config["train"]["device"]

    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    weight_bce = trial.suggest_float('weight_bce', 0.1, 2.0)
    weight_dice = trial.suggest_float('weight_dice', 0.1, 2.0)
    weight_iou = trial.suggest_float('weight_iou', 0.1, 2.0)
    weight_focal = trial.suggest_float('weight_focal', 0.1, 2.0)
    class_weight = trial.suggest_float('class_weight', 0.3, 2.0)

    # Load dataset and create data loaders
    dataset = ImageDataset(config=config)
    train_idx, val_idx = split_dataset(dataset)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss functions
    model = BreastTumorModel(config, trial).to(device)
    model = BreastTumorModel(config)

    # Print the model summary
    input_size = (1, 3, 512, 512)  # Adjust the input size according to your model's requirements
    summary(model, input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_seg = CombinedLoss(weight_bce=weight_bce, weight_dice=weight_dice, weight_iou=weight_iou, weight_focal=weight_focal)
    criterion_class = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            class_output, seg_output = model(images)
            loss_seg = criterion_seg(seg_output, masks)
            loss_class = criterion_class(class_output, labels)
            loss = loss_seg + class_weight * loss_class
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                class_output, seg_output = model(images)
                loss_seg = criterion_seg(seg_output, masks)
                loss_class = criterion_class(class_output, labels)
                loss = loss_seg + config["train"]["class_weight"] * loss_class
                val_loss += loss.item()

        # Report back to Optuna
        trial.report(val_loss / len(val_loader))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss / len(val_loader)

def save_params(config, trial):
    # Save the configuration including suggested hyperparameters
    config_to_save = config.copy()
    if trial:
        # Include suggested hyperparameters from the trial
        config_to_save["train"]["learning_rate"] = trial.params["learning_rate"]
        config_to_save["train"]["weight_decay"] = trial.params["weight_decay"]
        config_to_save["train"]["batch_size"] = trial.params["batch_size"]
        config_to_save["train"]["weight_bce"] = trial.params["weight_bce"]
        config_to_save["train"]["weight_dice"] = trial.params["weight_dice"]
        config_to_save["train"]["weight_iou"] = trial.params["weight_iou"]
        config_to_save["train"]["weight_focal"] = trial.params["weight_focal"]
        config_to_save["train"]["class_weight"] = trial.params["class_weight"]

    # Save updated config
    dump_config(config_to_save)

if __name__ == "__main__":
    config = load_config()

    # Create a study and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Get the best trial and save parameters
    best_trial = study.best_trial
    save_params(config, best_trial)
