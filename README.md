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

```
┌─────────────────────────────────────────────────┐
│                    Federated Learning System    │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ Client 1 │    │ Client 2 │    │ Client N │   │
│  │  Data    │    │  Data    │    │  Data    │   │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘   │
│       │               │               │         │
│       ▼               ▼               ▼         │
│  ┌────────────────────────────────────────┐     │
│  │     Local Training (CFL Model)         │     │
│  │  • Subset Generation                   │     │
│  │  • Noise Injection                     │     │
│  │  • Contrastive Learning                │     │
│  └────────────┬───────────────────────────┘     │
│               │                                 │
│               ▼                                 │
│  ┌────────────────────────────────────────┐     │
│  │    Central Server (Aggregation)        │     │
│  │  • Collect model parameters            │     │
│  │  • Aggregate updates                   │     │
│  │  • Distribute global model             │     │
│  └────────────────────────────────────────┘     │
│                                                 │
└─────────────────────────────────────────────────┘
```


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
git clone https://github.com/mambo06/CFL.git
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
  author={Achmad Ginanjar, Xue Li, Priyanka Singh, Wen Hua},
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


📞 Contact
For questions, issues, or collaboration opportunities:

Email: mambo06@gmail.com
Paper: SSRN Link


🗺️ Roadmap
Current Version (v1.0)

✅ Basic CFL implementation
✅ Attack and defense


