# Self-Pruning Neural Network

This repository contains the solution for the "Self-Pruning Neural Network" case study. 

## Project Overview

The objective of this project is to build a standard feed-forward neural network for CIFAR-10 classification with a built-in self-pruning mechanism. Instead of pruning connections *after* training, this network learns to dynamically prune its own connections *during* training.

This is achieved by introducing a learnable "gate" parameter for every weight in the network. A sparsity regularization term (L1 penalty on the gates) is added to the total loss, forcing the network to balance classification accuracy with sparsity.

## Repository Structure

```text
self-pruning-neural-network/
├── results/            # (Generated) CSV logs, Matplotlib plots, and best model checkpoints
├── data/               # (Generated) CIFAR-10 dataset downloads
├── self_pruning_nn.py  # Main script: Contains PrunableLinear, SelfPruningMLP, and the training loop
├── requirements.txt    # Python dependencies
├── report.md           # Final report containing mathematical intuition and experiment analysis
└── README.md           # Execution instructions (this file)
```

## Setup and Execution

1. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages via:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Experiment Sweep**:
   Execute the `self_pruning_nn.py` script. This script will download CIFAR-10 (if not present), train the model under different sparsity trade-off values ($\lambda$), evaluate hard and soft pruning accuracy, and generate plots.
   ```bash
   python self_pruning_nn.py
   ```

3. **View Results**:
   Once training finishes, check the `results/` folder for:
   - `lambda_results.csv`: Table of test accuracy and sparsity levels.
   - `accuracy_vs_sparsity.png`: Trade-off curve between accuracy and sparsity.
   - `gate_distribution.png`: Histogram of the final gate values showing the bimodal distribution.
   - `training_curves.png`: Loss and sparsity evolution across epochs.

## Key Features
- **PyTorch Fundamentals**: Custom `PrunableLinear` layer built from scratch with properly registered parameters.
- **Optimization**: L1 sparsity loss seamlessly integrated into the training loop alongside CrossEntropy.
- **Engineering Quality**: Modular code, grouped optimizer parameters (weight decay applied properly), and fully reproducible experiments with automatic best-model selection.
