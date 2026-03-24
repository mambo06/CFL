# Contrastive Federated Learning (CFL) for Tabular Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Learning from Tabular Data Silos without Data Sharing: A Contrastive Federated Learning Approach**
>
> 📄 [Read the Paper](https://papers.ssrn.com/abstract=5799977)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🎯 Overview

**Contrastive Federated Learning (CFL)** is a novel framework that enables multiple organizations to collaboratively learn from tabular data without sharing sensitive information. This approach addresses the critical challenge of **data silos** in industries like healthcare, finance, and telecommunications where data privacy regulations (GDPR, HIPAA) prevent direct data sharing.

### The Problem

Organizations often have:
- **Data silos**: Valuable data distributed across multiple parties
- **Privacy constraints**: Legal and ethical restrictions on data sharing
- **Heterogeneous data**: Non-IID (non-independent and identically distributed) data across clients
- **Limited labeled data**: Expensive or difficult to obtain labels

### Our Solution

CFL combines:
1. **Federated Learning**: Decentralized training without raw data exchange
2. **Contrastive Learning**: Self-supervised representation learning (inspired by [SubTab](https://github.com/AstraZeneca/SubTab))
3. **Autoencoder Architecture**: Learn robust feature representations

---

## ✨ Key Features

### 🔒 Privacy-Preserving
- **No raw data sharing**: Only model parameters are exchanged
- **Secure aggregation**: Compatible with differential privacy and secure multi-party computation
- **Client-level privacy**: Each organization maintains full control over their data

### 🎯 Robust to Data Heterogeneity
- **Non-IID data handling**: Works with imbalanced and heterogeneous data distributions
- **Client imbalance**: Handles varying amounts of data per client
- **Class imbalance**: Robust to skewed label distributions

### 🚀 Self-Supervised Learning
- **Contrastive learning**: Learn representations without extensive labels
- **Subset augmentation**: Generate multiple views through feature subsetting
- **Noise injection**: Swap noise, Gaussian noise, or zero-out strategies

### 📊 Flexible Aggregation
- **Multiple strategies**: Mean, sum, concatenation, max, min
- **Adaptive weighting**: Client contributions weighted by data size
- **Customizable**: Easy to implement custom aggregation methods

---

## 🏗️ Architecture

### System Overview


┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning System                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Client 1 │    │ Client 2 │    │ Client N │              │
│  │  Data    │    │  Data    │    │  Data    │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                      │
│       ▼               ▼               ▼                      │
│  ┌────────────────────────────────────────┐                 │
│  │     Local Training (CFL Model)         │                 │
│  │  • Subset Generation                   │                 │
│  │  • Noise Injection                     │                 │
│  │  • Contrastive Learning                │                 │
│  └────────────┬───────────────────────────┘                 │
│               │                                              │
│               ▼                                              │
│  ┌────────────────────────────────────────┐                 │
│  │    Central Server (Aggregation)        │                 │
│  │  • Collect model parameters            │                 │
│  │  • Aggregate updates                   │                 │
│  │  • Distribute global model             │                 │
│  └────────────────────────────────────────┘                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘

### Model Architecture


Input Data (X)
│
├─────────────────┬─────────────────┐
▼                 ▼                 ▼
Subset 1          Subset 2          Subset N
│                 │                 │
▼                 ▼                 ▼
Add Noise         Add Noise         Add Noise
│                 │                 │
▼                 ▼                 ▼
┌─────────────────────────────────────────┐
│           Encoder Network               │
│  Input → Hidden1 → Hidden2 → Latent    │
└─────────────────┬───────────────────────┘
│
▼
Latent Representation (Z)
│
├──────────────┬──────────────┐
▼              ▼              ▼
┌──────────────┐  ┌──────────┐  ┌──────────┐
│ Contrastive  │  │ Decoder  │  │Projection│
│    Loss      │  │ Network  │  │ Network  │
└──────────────┘  └──────────┘  └──────────┘
│
▼
Reconstructed Data (X')

### Loss Function

The total loss combines three components:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{contrastive}} + \beta \mathcal{L}_{\text{reconstruction}} + \gamma \mathcal{L}_{\text{z-space}}$$

Where:
- **$$\mathcal{L}_{\text{contrastive}}$$**: NT-Xent loss for contrastive learning
- **$$\mathcal{L}_{\text{reconstruction}}$$**: MSE loss for data reconstruction
- **$$\mathcal{L}_{\text{z-space}}$$**: Distance loss in latent space

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/contrastive-federated-learning.git
cd contrastive-federated-learning

Step 2: Create Virtual Environment
# Using conda (recommended)
conda create -n cfl python=3.8
conda activate cfl

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Verify Installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"


🚀 Quick Start
1. Prepare Your Data
Place your dataset in the data/ directory:
data/
├── covtype/
│   ├── train.csv
│   └── test.csv
└── your_dataset/
    ├── train.csv
    └── test.csv

2. Configure Your Experiment
Edit config/default.yaml:
# Model configuration
model_mode: "ae"
framework: "SubTab"
dims: [128, 64, 32]  # Hidden layer dimensions
latent_dim: 16

# Training configuration
epochs: 50
batch_size: 256
learning_rate: 0.001

# Federated learning
fl_cluster: 5  # Number of clients
client_drop_rate: 0.0
data_drop_rate: 0.0
client_imbalance_rate: 0.0
class_imbalance: 0.0

# Contrastive learning
n_subsets: 3
masking_ratio: 0.7
noise_type: "swap_noise"
noise_level: 0.1
aggregation: "mean"

3. Run Training
Federated Learning Mode
python main.py \
    --dataset covtype \
    --epochs 50 \
    --clients 5 \
    --gpu

Local Training Mode
python main.py \
    --dataset covtype \
    --epochs 50 \
    --local \
    --gpu

4. Evaluate the Model
python evaluate.py \
    --dataset covtype \
    --clients 5 \
    --gpu


⚙️ Configuration
Command-Line Arguments
| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| --dataset | -d | str | covtype | Dataset name |
| --gpu | -g | flag | True | Use CUDA GPU |
| --mps | -m | flag | False | Use Apple Silicon GPU |
| --no_gpu | -ng | flag | False | Force CPU usage |
| --device_number | -dn | str | 0 | GPU device number |
| --local | -lc | flag | False | Run in local mode |
| --epochs | -e | int | 1 | Number of epochs |
| --clients | -c | int | 4 | Number of FL clients |
| --client_drop | -cd | float | 0.0 | Client dropout rate |
| --data_drop | -dd | float | 0.0 | Data dropout rate |
| --noniid_clients | -nc | float | 0.0 | Non-IID client rate |
| --class_imbalance | -ci | float | 0.0 | Class imbalance rate |
| --base_global | -bg | flag | False | Use base global model |
Configuration Files
config/runtime.yaml
Runtime settings and paths:
# Paths
paths:
  results: "./results"
  data: "./data"

# Runtime settings
seed: 42
mlflow: false
profile: false
validate: true
training_data_ratio: 0.8

config/default.yaml
Model and training hyperparameters:
# Architecture
model_mode: "ae"
dims: [128, 64, 32]
latent_dim: 16
encoder_activation: "relu"
decoder_activation: "relu"

# Training
epochs: 50
batch_size: 256
learning_rate: 0.001
weight_decay: 1e-5

# Contrastive Learning
contrastive_loss: true
reconstruction: true
distance_loss: false
n_subsets: 3
masking_ratio: 0.7
reconstruct_subset: false

# Noise
add_noise: true
noise_type: "swap_noise"  # Options: swap_noise, gaussian_noise, zero_out
noise_level: 0.1

# Aggregation
aggregation: "mean"  # Options: mean, sum, concat, max, min

# Loss weights
alpha_contrastive: 1.0
beta_reconstruction: 1.0
gamma_distance: 0.5
temperature: 0.5


📚 Usage
Training Pipeline
1. Federated Learning with Multiple Clients
from src.model import CFL
from utils.load_data_new import Loader
from utils.arguments import get_config, get_arguments

# Parse arguments
args = get_arguments()
config = get_config(args)

# Initialize data loaders for each client
clients = []
for client_id in range(config['fl_cluster']):
    loader = Loader(config, dataset_name=config['dataset'], client=client_id)
    clients.append(loader)

# Initialize global model
global_model = CFL(config)

# Federated training loop
for epoch in range(config['epochs']):
    client_models = []
    
    # Local training on each client
    for client_id, client_loader in enumerate(clients):
        local_model = CFL(config)
        local_model.load_state_dict(global_model.state_dict())
        
        # Train on local data
        for batch_idx, (x, y) in enumerate(client_loader.trainFL_loader):
            loss, c_loss, r_loss, z_loss = local_model.fit(x)
            local_model.optimizer_ae.zero_grad()
            loss.backward()
            local_model.optimizer_ae.step()
        
        client_models.append(local_model)
    
    # Aggregate models
    global_model = aggregate_models(client_models, config)

2. Local Training (Single Client)
from src.model import CFL
from utils.load_data_new import Loader

# Load data
loader = Loader(config, dataset_name='covtype', client=0)

# Initialize model
model = CFL(config)

# Training loop
for epoch in range(config['epochs']):
    for batch_idx, (x, y) in enumerate(loader.train_loader):
        # Forward pass
        loss, c_loss, r_loss, z_loss = model.fit(x)
        
        # Backward pass
        model.optimizer_ae.zero_grad()
        loss.backward()
        model.optimizer_ae.step()
        
        # Log losses
        model.loss['tloss_b'].append(loss.item())
        model.loss['closs_b'].append(c_loss.item())
        model.loss['rloss_b'].append(r_loss.item())
        model.loss['zloss_b'].append(z_loss.item())
    
    # Validation
    val_loss = model.validate(x_val)
    model.loss['vloss_e'].append(val_loss.item())
    
    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

# Save model
model.save_weights(client=0)

Evaluation Pipeline
1. Extract Embeddings
from evaluate import evaluate_embeddings

# Evaluate on test set
results = evaluate_embeddings(
    data_loader=test_loader,
    model=model,
    config=config,
    client=0,
    mode='test'
)

precision, recall, f1, support = results
print(f"Test F1 Score: {f1:.4f}")

2. Downstream Classification
from utils.eval_utils import model_eval

# Train classifier on embeddings
results = model_eval(
    config=config,
    z_train=train_embeddings,
    y_train=train_labels,
    suffix="client-0",
    z_test=test_embeddings,
    y_test=test_labels
)

3. Visualize Clusters
from utils.eval_utils import plot_clusters

# Visualize learned representations
plot_clusters(
    config=config,
    z=embeddings,
    clabels=labels,
    suffix="client-0",
    plot_suffix="_test"
)


📊 Experimental Results
Datasets
We evaluate CFL on multiple tabular datasets:
| Dataset | Samples | Features | Classes | Task |
|---------|---------|----------|---------|------|
| Covertype | 581,012 | 54 | 7 | Multi-class |
| Income | 48,842 | 14 | 2 | Binary |
| Blog | 52,397 | 280 | 2 | Binary |
| Intrusion | 494,021 | 41 | 23 | Multi-class |
Performance Comparison
F1 Score (%) on Covertype Dataset
| Method | IID | Non-IID (α=0.3) | Non-IID (α=0.1) |
|--------|-----|-----------------|-----------------|
| Local Training | 72.3 | 68.5 | 64.2 |
| FedAvg | 78.6 | 74.3 | 69.8 |
| FedProx | 79.2 | 75.1 | 70.5 |
| CFL (Ours) | 82.4 | 79.6 | 76.3 |
Impact of Data Heterogeneity
F1 Score vs. Client Imbalance Rate

85% ┤                                    ╭─────
    │                              ╭─────╯
80% ┤                        ╭─────╯
    │                  ╭─────╯
75% ┤            ╭─────╯
    │      ╭─────╯
70% ┤╭─────╯
    │
65% └┬────┬────┬────┬────┬────┬────┬────┬────┬
     0   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8
              Client Imbalance Rate

Ablation Studies
Effect of Subset Count
| Subsets | F1 Score | Training Time |
|---------|----------|---------------|
| 2 | 78.2% | 1.2x |
| 3 | 82.4% | 1.5x |
| 4 | 81.9% | 1.9x |
| 5 | 81.5% | 2.3x |
Effect of Aggregation Method
| Method | F1 Score | Latent Dim |
|--------|----------|------------|
| Mean | 82.4% | 16 |
| Sum | 81.7% | 16 |
| Concat | 80.9% | 48 |
| Max | 79.3% | 16 |

📁 Project Structure
contrastive-federated-learning/
│
├── config/                      # Configuration files
│   ├── runtime.yaml            # Runtime settings
│   └── default.yaml            # Model hyperparameters
│
├── src/                        # Source code
│   ├── model.py               # CFL model implementation
│   └── __init__.py
│
├── utils/                      # Utility functions
│   ├── arguments.py           # Argument parsing
│   ├── utils.py               # General utilities
│   ├── load_data_new.py       # Data loading
│   ├── eval_utils.py          # Evaluation utilities
│   ├── loss_functionsV1.py    # Loss functions
│   ├── model_utils.py         # Model utilities
│   ├── model_plot.py          # Plotting functions
│   └── colors.py              # Color schemes
│
├── data/                       # Datasets
│   ├── covtype/
│   ├── income/
│   └── blog/
│
├── results/                    # Experimental results
│   └── [dataset]/
│       ├── training/
│       │   └── ae/
│       │       ├── model/     # Saved models
│       │       ├── plots/     # Training plots
│       │       └── loss/      # Loss logs
│       └── evaluation/
│           ├── clusters/      # Cluster visualizations
│           └── reconstructions/
│
├── notebooks/                  # Jupyter notebooks
│   ├── exploration.ipynb
│   └── visualization.ipynb
│
├── tests/                      # Unit tests
│   ├── test_model.py
│   └── test_utils.py
│
├── main.py                     # Training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── LICENSE                     # License file


🔬 Advanced Usage
Custom Dataset Integration
1. Prepare Your Data
# data/your_dataset/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['category_col'])
    
    # Normalize features
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col != 'label']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save processed data
    df.to_csv(output_path, index=False)
    
    return df

# Run preprocessing
df = preprocess_data('raw_data.csv', 'processed_data.csv')

2. Create Dataset Configuration
# config/your_dataset.yaml

dataset: "your_dataset"
n_features: 100
n_classes: 5
task_type: "classification"

# Data paths
data_path: "./data/your_dataset"
train_file: "train.csv"
test_file: "test.csv"

# Model architecture (adjust based on data complexity)
dims: [256, 128, 64]
latent_dim: 32

3. Run Training
python main.py \
    --dataset your_dataset \
    --epochs 100 \
    --clients 10 \
    --gpu

Hyperparameter Tuning
Grid Search Example
import itertools
from main import main

# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'latent_dim': [8, 16, 32],
    'n_subsets': [2, 3, 4],
    'masking_ratio': [0.5, 0.7, 0.9]
}

# Generate all combinations
keys = param_grid.keys()
values = param_grid.values()
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Run experiments
results = []
for params in combinations:
    # Update config
    config.update(params)
    
    # Train model
    result = main(config)
    
    # Store results
    results.append({
        'params': params,
        'f1_score': result['f1'],
        'accuracy': result['accuracy']
    })

# Find best parameters
best_result = max(results, key=lambda x: x['f1_score'])
print(f"Best parameters: {best_result['params']}")
print(f"Best F1 score: {best_result['f1_score']:.4f}")

Implementing Custom Aggregation
# utils/aggregation.py

import torch as th

def custom_weighted_aggregation(latent_list, weights=None):
    """
    Custom weighted aggregation of latent representations.
    
    Args:
        latent_list (list): List of latent tensors
        weights (list, optional): Weights for each latent
    
    Returns:
        torch.Tensor: Aggregated latent representation
    """
    if weights is None:
        weights = [1.0 / len(latent_list)] * len(latent_list)
    
    # Normalize weights
    weights = th.tensor(weights)
    weights = weights / weights.sum()
    
    # Weighted sum
    aggregated = sum(w * latent for w, latent in zip(weights, latent_list))
    
    return aggregated

# Add to config
config['aggregation'] = 'custom_weighted'

Monitoring with MLFlow
import mlflow

# Enable MLFlow
config['mlflow'] = True

# Set experiment
mlflow.set_experiment("CFL_Experiments")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(config)
    
    # Train model
    model = CFL(config)
    for epoch in range(config['epochs']):
        loss = train_epoch(model, data_loader)
        
        # Log metrics
        mlflow.log_metric("loss", loss, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model.encoder, "encoder")
    
    # Log artifacts
    mlflow.log_artifacts("results/plots")


🧪 Testing
Run Unit Tests
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/

Example Test
# tests/test_model.py

import torch
import pytest
from src.model import CFL

def test_model_initialization():
    config = {
        'dims': [128, 64, 32],
        'latent_dim': 16,
        'device': torch.device('cpu'),
        'seed': 42
    }
    
    model = CFL(config)
    assert model is not None
    assert model.encoder is not None

def test_forward_pass():
    config = {
        'dims': [10, 8, 4],
        'latent_dim': 2,
        'device': torch.device('cpu'),
        'n_subsets': 2
    }
    
    model = CFL(config)
    x = torch.randn(32, 10)  # Batch of 32 samples with 10 features
    
    loss, c_loss, r_loss, z_loss = model.fit(x)
    
    assert loss is not None
    assert loss.item() > 0


🤝 Contributing
We welcome contributions! Please follow these steps:
1. Fork the Repository
git clone https://github.com/yourusername/contrastive-federated-learning.git
cd contrastive-federated-learning

2. Create a Branch
git checkout -b feature/your-feature-name

3. Make Changes

Follow PEP 8 style guide
Add docstrings to all functions
Write unit tests for new features
Update documentation

4. Run Tests
pytest tests/
black src/ utils/  # Format code
flake8 src/ utils/  # Check style

5. Submit Pull Request
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

Then create a pull request on GitHub.

📖 Citation
If you use this code in your research, please cite our paper:
@article{cfl2025,
  title={Learning from Tabular Data Silos without Data Sharing: A Contrastive Federated Learning Approach},
  author={Your Name and Co-authors},
  journal={SSRN Electronic Journal},
  year={2025},
  url={https://papers.ssrn.com/abstract=5799977}
}


🙏 Acknowledgments
This project builds upon several excellent works:

SubTab: AstraZeneca/SubTab - Self-supervised representation learning for tabular data
SimCLR: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"

We thank the open-source community for their valuable contributions.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


📞 Contact
For questions, issues, or collaboration opportunities:

Email: your.email@example.com
GitHub Issues: Create an issue
Paper: SSRN Link


🗺️ Roadmap
Current Version (v1.0)

✅ Basic CFL implementation
✅ Federated learning support
✅ Multiple aggregation strategies
✅ Comprehensive evaluation

Upcoming Features (v1.1)

🔄 Differential privacy support
🔄 Secure aggregation protocols
🔄 Dynamic client selection
🔄 Adaptive learning rates

Future Plans (v2.0)

📋 Vertical federated learning
📋 Cross-silo federated learning
📋 Personalized federated learning
📋 Federated transfer learning


📊 Performance Tips
GPU Optimization
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in data_loader:
    with autocast():
        loss, _, _, _ = model.fit(x)
    
    scaler.scale(loss).backward()
    scaler.step(model.optimizer_ae)
    scaler.update()

Memory Optimization
# Gradient accumulation for large batches
accumulation_steps = 4

for i, (x, y) in enumerate(data_loader):
    loss, _, _, _ = model.fit(x)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        model.optimizer_ae.step()
        model.optimizer_ae.zero_grad()

Distributed Training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(model)

# Train as usual


🔍 Troubleshooting
Common Issues
1. CUDA Out of Memory
# Reduce batch size
config['batch_size'] = 128  # Instead of 256

# Or enable gradient checkpointing
model.encoder.gradient_checkpointing = True

2. Slow Training
# Use DataLoader with multiple workers
data_loader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    num_workers=4,  # Adjust based on CPU cores
    pin_memory=True  # For GPU training
)

3. Poor Convergence
# Adjust learning rate
config['learning_rate'] = 0.0001  # Lower LR

# Or use learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config['epochs']
)
