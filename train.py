import os
import argparse
from pathlib import Path

import torch
import yaml

from src.data.dataset import create_dataloaders
from src.models.model import ImprovedTrashNet
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train trash classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to config file')
    # I downloaded the dataset in my local
    parser.add_argument('--data-dir', type=str, 
                      default='/home/aftermath00/trashnet/notebooks/trash_data/train',
                      help='Directory containing the dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config['gpu_id']}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Using data from: {data_dir}")
    
    # Create dataloaders using existing data
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Created dataloaders with batch size: {config['batch_size']}")
    
    # Initialize model
    model = ImprovedTrashNet(num_classes=config['num_classes'])
    if config.get('multi_gpu', False) and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(config['checkpoint_path']).parent
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    print("Starting training...")
    # Train model
    trainer.train()
    
    # Push to hub if configured
    if config.get('push_to_hub', False):
        trainer.push_to_hub(
            repo_id=config['hub_repo_id'],
            token=os.environ.get('HF_TOKEN')
        )

if __name__ == '__main__':
    main()