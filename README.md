# TrashNet Classification

A deep learning model for classifying different types of trash using the TrashNet dataset. This project implements a custom CNN architecture with state-of-the-art training techniques and MLOps practices.

## Features

- Custom CNN architecture optimized for trash classification
- Multi-GPU training support
- Data augmentation pipeline
- Weights & Biases(WANDB) integration for experiment tracking
- Hugging Face Hub integration for model sharing

## Project Structure

```
trashnet-classification/
├── src/
│   ├── data/         # Dataset and data loading utilities
│   ├── models/       # Model architecture definitions
│   └── training/     # Training loops and utilities
├── scripts/          # Training and evaluation scripts
├── configs/          # Configuration files
├── notebooks/        # Development notebooks
└── tests/           # Unit tests (Not done yet!)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aftermath00/trashnet-classification.git
cd trashnet-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

### Training

1. Configure your training parameters in `configs/config.yaml`

2. Start training:
```bash
python scripts/train.py --config configs/config.yaml --data-dir /path/to/your/data
```

### Model Weights

The trained model weights will be saved in the `checkpoints/` directory. You can use these weights for inference or further training.

## Model Architecture

The ImprovedTrashNet architecture consists of:
- 4 convolutional blocks with batch normalization and ReLU activation
- Global average pooling
- Dropout for regularization
- Fully connected layers for classification

Key features:
- BatchNorm for stable training
- Dropout for preventing overfitting
- Residual connections for better gradient flow

## Training Strategy

- AdamW optimizer with OneCycleLR scheduler
- Label smoothing and class weights for handling imbalanced data
- Data augmentation including:
  - Random resized crops
  - Random flips and rotations
  - Color jittering
  - Affine transformations

## Experiment Tracking

This project uses Weights & Biases for experiment tracking. To enable tracking:

1. Sign up for a W&B account
2. Set your W&B API key:
```bash
export WANDB_API_KEY='your-key-here'
```
3. Enable W&B in the config file

## Model Deployment

Models can be automatically deployed to Hugging Face Hub. To enable this:

1. Create a Hugging Face account
2. Set your HF token:
```bash
export HF_TOKEN='your-token-here'
```
3. Enable model pushing in the config file

## Testing

(Not done yet!)

## Huggingface model
```bash

```

## Acknowledgments

- Dataset: [TrashNet](https://huggingface.co/datasets/garythung/trashnet)