import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from huggingface_hub import HfApi
import json


from src.utils.utils import train_epoch, validate, plot_confusion_matrix, save_checkpoint

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            config (Dict): Training configuration
            device (torch.device): Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Initialize wandb if needed
        if config.get('use_wandb', False):
            wandb.init(
                project=config['wandb_project'],
                name=config['wandb_run_name'],
                config=config
            )
            wandb.watch(model)
        
        # Setup training components
        self.setup_training()
        
    def setup_training(self) -> None:
        """Setup training components based on config."""
        # Calculate class weights for balanced loss
        weights = torch.tensor(self.config['class_weights']).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=self.config.get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['initial_lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['max_lr'],
            epochs=self.config['num_epochs'],
            steps_per_epoch=len(self.train_loader),
            div_factor=self.config['div_factor'],
            final_div_factor=self.config['final_div_factor']
        )
    
    def train(self) -> None:
        """Execute the training loop."""
        best_val_acc = 0
        early_stopping_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss, train_acc = train_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.scheduler, self.device, epoch
            )
            
            # Validation phase
            val_loss, val_acc, preds, targets = validate(
                self.model, self.val_loader, self.criterion, self.device
            )
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'train/epoch_acc': train_acc,
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                    'epoch': epoch
                })
            
            # Plot confusion matrix every few epochs
            if (epoch + 1) % self.config['cm_interval'] == 0:
                plot_confusion_matrix(
                    targets, preds, self.config['class_names']
                )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                metrics = {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc
                }
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, metrics, self.config['checkpoint_path']
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping check
            if early_stopping_counter >= self.config['patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Final evaluation
        self.evaluate()
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Tuple[float, float]: Test loss and accuracy
        """
        test_loss, test_acc, preds, targets = validate(
            self.model, self.test_loader, self.criterion, self.device
        )
        
        # Plot final confusion matrix
        plot_confusion_matrix(targets, preds, self.config['class_names'])
        
        if wandb.run is not None:
            wandb.log({
                'test/loss': test_loss,
                'test/acc': test_acc
            })
        
        return test_loss, test_acc
    
    def push_to_hub(self, repo_id: str, token: Optional[str] = None) -> None:
        """
        Push the model to Hugging Face Hub using HTTP API.
        
        Args:
            repo_id (str): Repository ID on Hugging Face Hub
            token (Optional[str]): HuggingFace token
        """
        # Get the base model if using DataParallel
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Create temporary directory
        os.makedirs("./temp_model", exist_ok=True)
        
        # Save model state dict
        torch.save(model_to_save.state_dict(), "./temp_model/model.pth")
        
        # Create config file
        config = {
            "num_classes": 6,
            "architecture": "ImprovedTrashNet",
            "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        }
        
        with open("./temp_model/config.json", "w") as f:
            json.dump(config, f)
        
        # Initialize Hugging Face API
        api = HfApi()
        
        print(f"Pushing files to {repo_id}...")
        
        try:
            # Upload files using HTTP
            api.upload_file(
                path_or_fileobj="./temp_model/model.pth",
                path_in_repo="model.pth",
                repo_id=repo_id,
                token=token or os.environ.get('HF_TOKEN')
            )
            
            api.upload_file(
                path_or_fileobj="./temp_model/config.json",
                path_in_repo="config.json",
                repo_id=repo_id,
                token=token or os.environ.get('HF_TOKEN')
            )
            
            print("Successfully pushed model to hub!")
            
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            raise e
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree("./temp_model", ignore_errors=True)