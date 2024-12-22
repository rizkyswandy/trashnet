from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TrashDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize TrashDataset.
        
        Args:
            root_dir (str): Directory containing the image files
            transform (Optional[transforms.Compose]): Transformations to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_files = list(self.root_dir.glob("*.jpg"))
        self.class_to_idx = {
            'cardboard': 0, 'glass': 1, 'metal': 2,
            'paper': 3, 'plastic': 4, 'trash': 5
        }
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(img_path.stem.split('class')[-1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(phase: str) -> transforms.Compose:
    """
    Get transforms for a specific phase (train/val/test).
    
    Args:
        phase (str): Either 'train' or 'val'/'test'
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir (str): Directory containing the images
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloaders
    """
    # Create full dataset
    full_dataset = TrashDataset(data_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms('train')
    val_dataset.dataset.transform = get_transforms('val')
    test_dataset.dataset.transform = get_transforms('val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


"""
1. Random Resized Cropping (224px with scale 0.8-1.0):
- Helps the model learn to recognize objects at different scales and positions
- The scale range (0.8-1.0) ensures we don't crop too aggressively and lose important features
- Real-world trash photos might capture items from different distances or angles


2. Random Horizontal/Vertical Flips:
- Horizontal flips: Most trash items look the same flipped horizontally (like a plastic bottle)
- Vertical flips (20% probability): Less common but still valid - trash can be oriented any way
- Teaches the model orientation invariance - it should recognize items regardless of their orientation


3. Random Rotation (30 degrees):
- Trash items in real photos won't always be perfectly aligned
- 30 degrees is a moderate range that preserves recognizability while adding variation
- Helps model learn rotational invariance without extreme angles that might confuse it


4. Random Affine Transformations:
- Translation: Objects won't always be centered in real photos
- Scaling: Accounts for varying distances and sizes of trash items
- Makes model more robust to different perspectives and positions


5. Color Jittering:
- Brightness: Accounts for different lighting conditions (indoor/outdoor, sunny/cloudy)
- Contrast: Helps with varying photo quality and lighting conditions
- Saturation: Different cameras might capture colors differently
- Hue: Handles slight color variations in similar materials
"""