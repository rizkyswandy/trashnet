import torch
import torch.nn as nn
from typing import Dict, Any

class ImprovedTrashNet(nn.Module):
    def __init__(self, num_classes: int = 6):
        """
        ImprovedTrashNet: A CNN architecture for classifying different types of trash.
        
        Architecture Details:
        -------------------
        Input: RGB image (3 channels) of size 224x224
        
        Feature Extraction Blocks:
        ------------------------
        First Block (Basic Pattern Detection):
            - Conv2d(3->64): Initial feature detection for basic patterns
                * kernel_size=3: Look at small local regions
                * padding=1: Preserve spatial dimensions
            - BatchNorm2d: Normalize activations for stable training
            - ReLU: Non-linear activation to capture complex patterns
            - MaxPool2d(2): Reduce spatial dimensions by half (224->112)
            Output: 112x112x64
        
        Second Block (Pattern Composition):
            - Conv2d(64->128): Combine basic features into more complex patterns
                * kernel_size=3: Maintain local context
                * padding=1: Preserve information flow
            - BatchNorm2d: Maintain stable gradients
            - ReLU: Introduce non-linearity
            - MaxPool2d(2): Further dimension reduction (112->56)
            Output: 56x56x128
        
        Third Block (Complex Feature Detection):
            - Conv2d(128->256): Detect higher-level features and combinations
                * Same kernel and padding setup for consistency
            - BatchNorm2d: Control internal covariate shift
            - ReLU: Non-linear feature transformation
            - MaxPool2d(2): Spatial reduction (56->28)
            Output: 28x28x256
        
        Fourth Block (Abstract Feature Learning):
            - Conv2d(256->512): Highest level feature detection
                * Maximum channel depth for rich feature representation
            - BatchNorm2d: Normalize deep features
            - ReLU: Final non-linear transformation
            - MaxPool2d(2): Final spatial reduction (28->14)
            Output: 14x14x512
        
        Global Feature Processing:
            - AdaptiveAvgPool2d((1,1)): Convert spatial features to vector
                * Reduces 14x14x512 to 1x1x512
            - Dropout(0.4): Prevent overfitting through regularization
            Output: 512 features
        
        Classifier:
            - Linear(512->256): Dimensionality reduction
            - ReLU: Non-linear transformation
            - Dropout(0.4): Additional regularization
            - Linear(256->num_classes): Final classification
            Output: num_classes predictions (default: 6)
        
        Design Rationale:
        ----------------
        1. Progressive Feature Learning:
        - Channel depth increases gradually (3->64->128->256->512)
        - Spatial dimensions reduce systematically (224->112->56->28->14->1)
        
        2. Regularization Strategy:
        - BatchNorm after each conv layer for training stability
        - Dual Dropout layers (0.4) for robust generalization
        
        3. Architectural Choices:
        - No skip connections for simplicity
        - Consistent conv kernel size (3x3) throughout
        - Regular spatial reduction pattern
        
        Args:
            num_classes (int): Number of output classes (default: 6 for trash types)
        """
        super(ImprovedTrashNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block of layers
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second block of layers
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third block of layers
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth block of layers
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global Feature Processing
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.4)
        )
        
        # The classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Add model weights",
        token: str = None
    ) -> None:
        """
        Save model to Hugging Face Hub.
        
        Args:
            repo_id (str): Repository ID on Hugging Face Hub
            commit_message (str): Commit message
            token (str): HuggingFace token
        """
        from huggingface_hub import Repository, HfApi
        import os
        import json
        
        config = {
            "num_classes": self.classifier[-1].out_features,
            "architecture": "ImprovedTrashNet",
            "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        }
        
        repo = Repository(
            local_dir="./hf_model",
            clone_from=repo_id,
            use_auth_token=token
        )
        
        torch.save(self.state_dict(), "./hf_model/model.pth")
        
        with open("./hf_model/config.json", "w") as f:
            json.dump(config, f)
        
        repo.push_to_hub(commit_message=commit_message)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str = None
    ) -> 'ImprovedTrashNet':
        """
        Load model from Hugging Face Hub.
        
        Args:
            repo_id (str): Repository ID on Hugging Face Hub
            token (str): HuggingFace token
        
        Returns:
            ImprovedTrashNet: Loaded model
        """
        from huggingface_hub import hf_hub_download
        import json
        
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            use_auth_token=token
        )
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        model = cls(num_classes=config['num_classes'])
        
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.pth",
            use_auth_token=token
        )
        
        model.load_state_dict(torch.load(weights_file))
        
        return model