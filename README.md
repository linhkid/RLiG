# RLiG: Reinforcement Learning for Bayesian Network Structure in Tabular GANs

This repository contains the implementation of RLiG (Reinforcement Learning Inspired Layered Tabular Data Generation), an innovative generative model for creating synthetic tabular data. RLiG extends the GANBLR framework by incorporating reinforcement learning to dynamically learn the structure of the Bayesian network generator during the GAN training process.

## Overview

Traditional deep learning-based GANs face challenges when applied to tabular data due to difficulties in modeling feature interactions implicitly. GANBLR introduced a novel approach using Bayesian Networks (BNs) for both generator and discriminator components, offering more interpretability and explicit feature interaction modeling. 

However, GANBLR uses a fixed Bayesian network structure that is learned prior to GAN training, limiting its data generation quality. RLiG addresses this limitation through several key innovations:

### Key Technical Features

- **Dynamic Structure Learning**: Uses reinforcement learning to learn the Bayesian network structure during the GAN training process
- **Layered Approach**: Employs a layered framework that trains multiple generators to handle the variable-sized parameter space
- **Hybrid Policy**: Combines greedy Hill Climbing search with reinforcement learning to efficiently explore the structure space
- **Generative and Non-Generative States**: Alternates between computing empirical probabilities (non-generative states) and training GANBLR models to sample synthetic data (generative states)
- **Advanced Reward Function**: Incorporates both log-likelihood metrics and discriminator loss to guide structure learning

### Advantages

- Explicitly models complex feature interactions through learned Bayesian network structures
- Offers interpretability by using probabilistic models rather than black-box neural networks
- Adapts network structures to better represent the underlying data distribution
- Achieves higher quality synthetic data generation through integration of structure learning with generative training
- Demonstrates potential for causal structure discovery

RLiG has been extensively evaluated for both machine learning utility (using TSTR - Train on Synthetic, Test on Real) and statistical similarity metrics (Jensen-Shannon Divergence, Wasserstein Distance), showing competitive or superior performance compared to state-of-the-art tabular data generation methods including CTGAN, TableGAN, MedGAN, and GANBLR.

## Technical Architecture

RLiG implements a sophisticated reinforcement learning framework for Bayesian network structure learning:

```
+--------------------------------------------+
|                  RLiG                      |
+--------------------------------------------+
|                                            |
|  +----------+                              |
|  |          |                              |
|  |  Layer 1 | Tile 1.1  Tile 1.2  Tile 1.3 |
|  |          |    |         |        |      |
|  +----------+    V         V        V      |
|                States    States    States   |
|                                            |
|  +----------+                              |
|  |          |                              |
|  |  Layer 2 | Tile 2.1  Tile 2.2  Tile 2.3 |
|  |          |    |         |        |      |
|  +----------+    V         V        V      |
|                States    States    States   |
|                                            |
|          ...                               |
|                                            |
|  +----------+                              |
|  |          |                              |
|  |  Layer N | Final Bayesian Network       |
|  |          | Structure for GANBLR         |
|  +----------+                              |
|                                            |
+--------------------------------------------+
```

### States in Each Tile

Each tile contains a sequence of states, which can be either:

1. **Non-Generative States**: Compute empirical probabilities based on the current structure and real data
   - Reward: Log-likelihood of original data given structure, penalized for complexity

2. **Generative States**: Train a GANBLR model with the current structure to sample synthetic data
   - Reward: Incorporates discriminator loss from GANBLR training and log-likelihood/complexity term

### RL Algorithm Implementation

The core reinforcement learning algorithm uses:

- **State Space**: Current Bayesian network structure (nodes and edges)
- **Action Space**: Adding/removing/flipping edges between nodes
- **Hybrid Policy**: 
  - Hill Climbing (HC) for local structure optimization
  - Q-learning for exploring promising structure directions
  - β-decay strategy to balance between greedy and RL-based actions
- **Reward Function**: Combination of BIC score, discriminator loss, and complexity penalty
- **Learning Process**: Experience replay with a stack buffer to improve sample efficiency

### GANBLR Integration

The final learned structure is used to initialize a GANBLR model where:
- Generator: BNe (Bayesian Network with Equality constraints)
- Discriminator: BNd (Discriminatively trained Bayesian Network)
- Training: Adversarial process with the discriminator guiding generator updates

## Getting Started

### Quick Start (Recommended)

For team members who want to get started quickly with the fewest compatibility issues:

```bash
# 1. Create a conda environment with Python 3.8
conda create -n rlig python=3.8 -y
conda activate rlig

# 2. Install core dependencies
pip install numpy pandas matplotlib tqdm

# 3. Install exact versions for compatibility
pip install scikit-learn==0.24.0
pip install tensorflow==2.6.2
pip install pgmpy==0.1.25
pip install causalnex==0.11.0 
pip install ucimlrepo  # For baseline comparisons

# 4. Install RLiG package
cd ganblr-0.1.1
pip install -e .
cd ..

# 5. Test installation
python -c "from ganblr.models import RLiG; print('RLiG successfully imported!')"
```

### Prerequisites (Alternative)

RLiG can also work with newer dependency versions, but might require some adjustments:

```bash
# Create a conda environment with Python 3.8+
conda create -n rlig python=3.11 -y
conda activate rlig

# Install dependencies
pip install numpy pandas scikit-learn tqdm matplotlib pgmpy
pip install tensorflow  # Latest version
pip install causalnex ucimlrepo  # For baseline comparisons
```

Note: If you use newer versions and encounter import errors, see the troubleshooting section in INSTALL.md.

### Installation

Clone this repository and install the RLiG package:

```bash
# Clone the repository
git clone https://github.com/your-username/RLiG.git
cd RLiG

# Install the package
cd ganblr-0.1.1
pip install -e .
cd ..
```

## Usage

### Model Comparison

Two main evaluation scripts are provided:

1. **Real Data Evaluation**:
   ```bash
   python compare_models_real_data.py
   ```
   This script evaluates models by training and testing on real data.

2. **TSTR (Train on Synthetic, Test on Real) Evaluation**:
   ```bash
   python eval_tstr_final.py
   ```
   This script implements the TSTR methodology for evaluating generative models:
   - Trains generative models on real data
   - Generates synthetic data from trained models
   - Trains classification models on synthetic data
   - Tests on real data
   - Compares results across different models (RLiG, GANBLR, GANBLR++, CTGAN, NaiveBayes)

### Using the Interactive Notebook

For a step-by-step comparison process with more visualization and diagnostics:

```bash
jupyter notebook model_comparison_guide.ipynb
```

The notebook provides:
- Environment setup
- Data preprocessing
- Individual testing of baseline models
- RLiG model evaluation
- Result visualization and comparison
- Troubleshooting guidance

### Running with Custom Datasets

To run with your own datasets, modify the `datasets` dictionary in either evaluation script:

```python
datasets = {
    'YourDataset': '/path/to/your/dataset.csv',
    # Add more datasets as needed
}
```

For UCI datasets, you can use their ID:

```python
datasets = {
    'Rice': 545,  # UCI ID for Rice dataset
    'YourDataset': 123  # Replace with actual UCI ID
}
```

## Repository Structure

- `ganblr-0.1.1/`: Source code for the RLiG implementation
  - `ganblr/models/rlig.py`: Main RLiG model implementation
  - `ganblr/structure_learning/`: Structure learning algorithms
- `Baselines/`: Baseline methods for comparison
- `compare_models_real_data.py`: Script for comparing models using real data
- `eval_tstr_final.py`: Script for evaluating models using TSTR methodology
- `model_comparison_guide.ipynb`: Interactive notebook for guided comparison
- `img/`: Directory for storing network visualizations
- `results/`: Directory for storing evaluation results
- `train_data/`: Directory for storing training data CSV files

## Docker Setup

A Dockerfile is provided for easier setup, especially useful for GPU acceleration:

```bash
# Build the Docker image
docker build -t rlig-env .

# Run a container with the image
docker run -it --gpus all -v $(pwd):/workspace -p 8888:8888 rlig-env

# Inside the container, install the package
cd /workspace/ganblr-0.1.1
pip install -e .
cd ..

# Run Jupyter notebook (if needed)
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root --no-browser
```

This Docker setup provides:
- TensorFlow 2.7.0 with GPU support
- All required dependencies pre-installed
- SSH access for remote development
- Port 8888 exposed for Jupyter notebook

## Troubleshooting

If you encounter CPD normalization issues:
- The code includes fixes to ensure CPDs sum to 1
- Check the notebook for diagnostic outputs
- Ensure your input data is properly preprocessed

For other issues:
- Check the error logs and ensure all dependencies are correctly installed
- For TensorFlow compatibility issues, try using the Docker environment
- If using CPUs only, reduce batch sizes and model complexity

---

<!-- Original README content:
# Unrestricted-GANBLR

This is the repository for an GANBLR variation using the unrestricted Bayesian Network

https://ganblr-docs.readthedocs.io/en/latest/

Here's the description of each folder
* ganblr-0.1.1: The origin ganblr source code with a newly added dockerfile
-->
