# RLiG (Reduced Incomplete Local Graphs)

This repository contains the implementation of RLiG, a novel approach for structure learning in Bayesian networks using reinforcement learning. RLiG combines reinforcement learning with generative modeling techniques to learn network structures effectively.

## Getting Started

### Prerequisites

RLiG requires Python 3.8+ and several dependencies. We recommend using a conda environment for installation:

```bash
# Create a conda environment with Python 3.8
conda create -n rlig python=3.8 -y
conda activate rlig

# Install dependencies
pip install numpy pandas scikit-learn tqdm matplotlib pgmpy
pip install tensorflow==2.6.2  # Specific version required
pip install causalnex ucimlrepo  # For baseline comparisons
```

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

To compare RLiG with baseline models (HillClimbSearch, TreeSearch, GaussianNB, NOTEARS):

```bash
python compare_models.py
```

This will run experiments on sample datasets and output performance metrics including accuracy, BIC score, and training time.

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

To run with your own datasets, modify the `datasets` dictionary in `compare_models.py`:

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
- `compare_models.py`: Script for comparing RLiG with baselines
- `model_comparison_guide.ipynb`: Interactive notebook for guided comparison

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
