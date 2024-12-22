# GitHub Actions Workflows

This repository uses GitHub Actions to automate the model training and deployment process. There are two main workflows:

## Training Workflow (`train.yml`)

The training workflow handles the model training process.

### Triggers
- Push to `train` branch
- Manual trigger via GitHub Actions UI (workflow_dispatch)

### What it does
1. Sets up Python 3.10 environment
2. Installs required dependencies
3. Configures Weights & Biases for experiment tracking
4. Downloads the training dataset (trashnet)
5. Runs the training script
6. Uploads the trained model as workflow artifacts

### Prerequisites
- Python requirements specified in `requirements.txt`
- Weights & Biases API key stored in repository secrets as `WANDB_API_KEY`

## Deployment Workflow (`deploy.yml`)

The deployment workflow handles pushing the trained model to HuggingFace Hub.

### Triggers
- Automatic trigger when Training workflow completes successfully
- Manual trigger via GitHub Actions UI (workflow_dispatch)

### What it does
1. Sets up Python 3.10 environment
2. Installs required dependencies
3. Downloads the model artifacts from the training workflow
4. Logs in to HuggingFace Hub
5. Pushes the model to HuggingFace Hub

### Prerequisites
- HuggingFace API token stored in repository secrets as `HF_TOKEN`

## Workflow Execution Flow

1. When code is pushed to the `train` branch or manually triggered:
   - Training workflow starts
   - Model is trained
   - Model artifacts are saved

2. After successful training:
   - Deployment workflow automatically triggers
   - Model is pushed to HuggingFace Hub

## Manual Execution

Both workflows can be manually triggered from the GitHub Actions UI:
1. Go to the "Actions" tab in your repository
2. Select the workflow you want to run
3. Click "Run workflow"