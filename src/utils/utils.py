from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_class_weights(dataloader: DataLoader) -> torch.Tensor:
    """
    Calculate class weights for balanced loss.
    
    Args:
        dataloader (DataLoader): Training dataloader
    
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = torch.zeros(6)
    for _, labels in tqdm(dataloader, desc="Calculating class weights"):
        for label in labels:
            class_counts[label] += 1
    
    weights = 1. / class_counts.float()
    weights = weights / weights.sum()
    return weights

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): Model to train
        loader (DataLoader): Training data loader
        criterion (nn.Module): Loss criterion
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use for training
        epoch (int): Current epoch number
    
    Returns:
        Tuple[float, float]: Epoch loss and accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_acc': acc,
                'train/learning_rate': scheduler.get_last_lr()[0]
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """
    Validate the model.
    
    Args:
        model (nn.Module): Model to validate
        loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss criterion
        device (torch.device): Device to use for validation
    
    Returns:
        Tuple[float, float, List[int], List[int]]: 
            Validation loss, accuracy, predictions and targets
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validating'):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

def plot_confusion_matrix(
    targets: List[int],
    predictions: List[int],
    class_names: List[str]
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        targets (List[int]): True labels
        predictions (List[int]): Predicted labels
        class_names (List[str]): Names of classes
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict,
    filename: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler state
        epoch (int): Current epoch
        metrics (Dict): Training metrics
        filename (str): Where to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f'model-checkpoint-{epoch}',
            type='model',
            description=f'Model checkpoint from epoch {epoch}'
        )
        artifact.add_file(filename)
        wandb.log_artifact(artifact)