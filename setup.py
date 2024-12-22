from setuptools import setup, find_packages

setup(
    name="trashnet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "Pillow>=8.0.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "datasets>=2.0.0",
        "huggingface-hub>=0.4.0",
        "PyYAML>=6.0",
        "jupyter>=1.0.0",
    ],
)