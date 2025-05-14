# TabSyn Integration for RLiG

This document explains how to use the TabSyn integration scripts to generate synthetic tabular data using the TabSyn VAE+Diffusion approach.

## Overview

TabSyn is a deep generative model for tabular data synthesis that combines Variational Autoencoders (VAE) and diffusion models. The integration consists of:

1. `tabsyn_proper_wrapper.py` - A wrapper class that manages TabSyn's file structure, data preparation, and execution
2. `test_tabsyn.py` - A testing script for running TabSyn on a single dataset
3. `tabsyn_evaluation.py` - A pipeline script for generating synthetic data for multiple datasets and running evaluations

## Requirements

TabSyn requires:
- Python 3.7+
- PyTorch
- CUDA-compatible GPU (recommended)
- Various data science packages (numpy, pandas, scikit-learn)

All requirements should be included in the project's requirements.txt.

## Usage Instructions

### Testing with a Single Dataset

To test TabSyn on a single dataset:

```bash
python test_tabsyn.py --dataset car --epochs 500 --gpu 0 --verbose
```

Arguments:
- `--dataset`: Dataset name (adult, car, chess, magic, nursery, letter-recog, poker-hand)
- `--epochs`: Number of training epochs for the VAE model (default: 500)
- `--gpu`: GPU index to use (default: 0)
- `--nfe`: Number of function evaluations for diffusion sampling (default: 50)
- `--n_samples`: Number of synthetic samples to generate (default: same as original data)
- `--discretize`: Flag to discretize continuous target variables
- `--verbose`: Flag for detailed output

### Running the Full Evaluation Pipeline

To generate synthetic data for multiple datasets and run evaluations:

```bash
python tabsyn_evaluation.py --datasets adult car chess --epochs 1000 --gpu 0 --verbose
```

Arguments:
- `--datasets`: List of datasets to process (if omitted, all datasets are processed)
- `--exclude`: List of datasets to exclude from processing
- `--epochs`: Number of training epochs for the VAE model (default: 500)
- `--gpu`: GPU index to use (default: 0)
- `--nfe`: Number of function evaluations for diffusion sampling (default: 50)
- `--results_dir`: Directory to store evaluation results (default: 'results')
- `--skip_eval`: Flag to skip evaluation and only generate synthetic data
- `--discretize`: Flag to discretize continuous target variables
- `--verbose`: Flag for detailed output

### Using the Wrapper Directly

You can also use the TabSynWrapper class directly in your code:

```python
from tabsyn_proper_wrapper import TabSynWrapper

# Initialize wrapper
tabsyn = TabSynWrapper(dataset_name='custom', epochs=1000, gpu=0, nfe=50)

# Prepare data and train
tabsyn.fit(X, y)

# Generate synthetic data
synthetic_data = tabsyn.sample(n_samples=1000)
```

## How TabSyn Works

The TabSyn process involves several steps:

1. **Data Preparation**:
   - Processes numerical and categorical features separately
   - Creates proper metadata as expected by TabSyn
   - Saves data in specific directory structure

2. **VAE Training**:
   - Trains a VAE model to learn a latent representation of the data
   - Captures relationships between features
   - Creates embeddings in latent space

3. **Diffusion Model Training**:
   - Trains a diffusion model in the latent space
   - Learns the denoising score function for generating new samples

4. **Synthetic Data Generation**:
   - Samples from the diffusion model in latent space
   - Uses the VAE decoder to transform latent samples back to feature space
   - Post-processes to ensure data validity

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or model size if GPU memory is insufficient
- **File Not Found Errors**: Check that the TabSyn directory structure is correctly set up
- **IndexError**: Ensure the dataset has at least one numerical and one categorical feature
- **KeyError**: Check that the column names in the dataset match those expected in info.json

If TabSyn fails to run properly, you can examine the error output to identify the issue or set `--verbose` flag for more detailed logging.

## Notes on Performance

- Training the VAE and diffusion models can take significant time, especially for large datasets
- GPU acceleration is highly recommended for reasonable performance
- Models are saved after training, so generation can be done separately later
- Higher epochs and NFE values generally produce better quality synthetic data but take longer to train/generate