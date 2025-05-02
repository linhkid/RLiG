# Installation Guide for RLiG

This document provides detailed instructions for installing RLiG and all required dependencies in different environments.

## Method 1: Installation using Conda (Recommended)

### Step 1: Create a conda environment
```bash
# Create a new conda environment with Python 3.10 that is the required version of Python to run
conda create -n rlig python=3.10 -y

# Activate the environment
conda activate rlig

# Install by using the requirements.txt file
pip install -r requirements.txt
# then go to directly step 5 to install graphicviz
```

### Step 2: Install core dependencies
```bash
# Install core packages
pip install numpy pandas scikit-learn matplotlib pyitlib Pympler -y

# Install specialized packages
pip install git+https://github.com/pgmpy/pgmpy.git@dev
pip install networkx tqdm -y
```

### Step 3: Install TensorFlow (specific version)
```bash
# TensorFlow 2.19.0 is recommended for compatibility
pip install tensorflow
```

### Step 4: Install additional packages for comparisons
```bash
# Install packages for baseline comparisons
pip install ucimlrepo
```

### Step 5: Install graphviz packages for specific machine: https://pygraphviz.github.io/documentation/stable/install.html
```bash
brew install graphviz
pip install pygraphviz
```
or if not working, use:
```bash
pip install --config-settings="--global-option=build_ext" \
            --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
            --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
            pygraphviz
```

### Step 5: Check installation with a test run
```bash
# Run a test of the installation
python ganblr-0.1.1/main.py

# Test evaluation scripts
python eval_tstr_final.py --datasets TicTacToe
```

### Step 6: Install RLiG package
```bash
# Navigate to the ganblr directory
cd ganblr-0.1.1

# Install in development mode
pip install -e .

# Return to main directory
cd ..
```

## Method 2: Installation using Docker

Docker provides an isolated environment with all dependencies pre-configured, which is especially useful for GPU support.

### Step 1: Build the Docker image
```bash
# Build the Docker image using the provided Dockerfile
docker build -t rlig-env .
```

### Step 2: Run a container
```bash
# Run a container with the current directory mounted
docker run -it --gpus all -v $(pwd):/workspace -p 8888:8888 rlig-env
```

### Step 3: Install RLiG inside the container
```bash
# Navigate to the ganblr directory in the mounted volume
cd /workspace/ganblr-0.1.1

# Install in development mode
pip install -e .

# Return to workspace
cd ..
```

### Step 4: Run Jupyter (if needed)
```bash
# Start Jupyter notebook server
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root --no-browser
```

## Method 3: Manual Installation (Python venv)

If you prefer not to use Conda or Docker, you can use Python's built-in venv.

### Step 1: Create a virtual environment
```bash
# Create a new virtual environment
python -m venv rlig_env

# Activate the environment (Linux/Mac)
source rlig_env/bin/activate

# Activate the environment (Windows)
# rlig_env\Scripts\activate
```

### Step 2: Install dependencies
```bash
# Install all dependencies from requirements.txt
pip install -r ganblr-0.1.1/requirements.txt

# Install additional packages
pip install causalnex ucimlrepo matplotlib jupyter
```

### Step 3: Install RLiG package
```bash
# Navigate to the ganblr directory
cd ganblr-0.1.1

# Install in development mode
pip install -e .

# Return to main directory
cd ..
```

## Troubleshooting Installation Issues

### Package version conflicts
If you encounter package version conflicts, try installing dependencies one by one in this order:
```bash
pip install numpy pandas
pip install scikit-learn==0.24.0
pip install tensorflow==2.6.2
pip install pgmpy==0.1.25
```

### pgmpy API Compatibility Issues
The codebase was originally designed for pgmpy 0.1.25, but newer versions have API changes that cause import errors. If you encounter errors like:

```
ImportError: cannot import name 'config' from 'pgmpy'
ImportError: cannot import name 'AICScore' from 'pgmpy.estimators'
```

You have two options:

1. Install the specific version of pgmpy:
```bash
pip install pgmpy==0.1.25
```

2. Use our compatibility fixes which handle the API differences between versions.
   - The code has been updated to import only available modules
   - Fallback implementations are provided for missing components
   - Warnings will be displayed when using fallbacks

### TensorFlow GPU issues
If you're having trouble with TensorFlow GPU support:
1. Ensure you have the correct CUDA and cuDNN versions installed
2. Use the Docker setup which includes the correct versions
3. Fall back to CPU-only version with: `pip install tensorflow==2.6.2`

### PyItLib installation issues
If you encounter errors installing PyItLib:
```bash
# Install from source
git clone https://github.com/pafoster/pyitlib.git
cd pyitlib
pip install -e .
cd ..
```

### Memory issues during installation
If you're experiencing memory issues during installation:
```bash
# Install with reduced parallelism
pip install --no-cache-dir --jobs 1 tensorflow==2.6.2
```

## Verification

To verify your installation:

```bash
# Start Python interpreter
python

# Try importing key packages
>>> import numpy as np
>>> import pandas as pd
>>> import tensorflow as tf
>>> from ganblr.models import RLiG
>>> print(f"TensorFlow version: {tf.__version__}")
>>> print("Installation successful!")
```

## Running Evaluation Scripts

After installation, you can run the evaluation scripts:

```bash
# For TSTR (Train on Synthetic, Test on Real) evaluation
python eval_tstr_final.py

# For evaluation on real data
python compare_models_real_data.py
```

The scripts will:
1. Load datasets (from UCI repository or local files)
2. Preprocess data for model compatibility
3. Train and evaluate models
4. Save results to CSV files in the 'results' directory
5. Save network visualizations to the 'img' directory
```