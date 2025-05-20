# Synthetic Data Similarity Measurement Tool

This script provides a standalone utility to measure the similarity between synthetic and real datasets using two key metrics:

1. **Jensen-Shannon Divergence (JSD)** - For categorical features
2. **Wasserstein Distance (WD)** - For numerical features

It is based on the methodology described in the paper ["GANBLR: GAN-Based Bayesian learning for classification using realistically-created synthetic data"](https://www.nayyarzaidi.com/papers/ICDM_Ganblr.pdf).

## Features

- Measure similarity between a single synthetic dataset and a real dataset
- Compare multiple synthetic datasets (from different models) against a real dataset
- Automatic detection of categorical columns
- Detailed reporting of JSD, WD, and overall similarity scores
- Direct fetching of datasets from the UCI Machine Learning Repository

## Installation

This script requires several Python packages. Basic functionality requires:
```bash
pip install numpy pandas scipy scikit-learn
```

For UCI dataset fetching functionality, install:
```bash
pip install ucimlrepo
```

## Usage

### Basic Usage

```bash
python measure_similarity.py --real path/to/real_data.csv --synth_file path/to/synthetic_data.csv
```

### Compare Multiple Models

```bash
python measure_similarity.py --real path/to/real_data.csv --synth_dir path/to/synthetic_datasets/
```

### Save Results to CSV

```bash
python measure_similarity.py --real path/to/real_data.csv --synth_dir path/to/synthetic_datasets/ --output results.csv
```

### Specify Categorical Columns Manually

```bash
python measure_similarity.py --real path/to/real_data.csv --synth_file path/to/synthetic_data.csv --cat_cols 0,1,3,5
```

### Adjust Categorical Column Detection Threshold

```bash
python measure_similarity.py --real path/to/real_data.csv --synth_file path/to/synthetic_data.csv --cat_threshold 15
```

## Arguments

### Specifying Real Data (required one of these)
- `--real`: Path to the real dataset CSV
- `--uci_id`: UCI ML Repository dataset ID to use as real data
- `--uci_name`: UCI ML Repository dataset name to use as real data (e.g., "adult", "iris", "car")

### Specifying Synthetic Data (required one of these)
- `--synth_file`: Path to a specific synthetic dataset CSV 
- `--synth_dir`: Directory containing multiple synthetic dataset CSVs

### Other Options
- `--output`: Path to save the comparison results CSV
- `--cat_cols`: Comma-separated list of categorical column indices (0-based)
- `--cat_threshold`: Threshold for automatic categorical column detection (default: 10)
- `--target_col`: Name of the target column to exclude from comparison
- `--drop_first_col`: Flag to drop the first column (often an index column)
- `--tmp_dir`: Directory to save temporary files (for UCI downloads)

You must provide either one real data source and one synthetic data source for comparison.

## Examples

### Basic Comparison
```bash
python measure_similarity.py --real data/adult/train.csv --synth_file train_data/ganblr_Adult_synthetic.csv
```

### Compare Multiple Models
```bash
python measure_similarity.py --real data/adult/train.csv --synth_dir train_data/ --output similarity_results.csv
```

### Handle Mismatched Columns
```bash
python measure_similarity.py --real data/adult/train.csv --synth_dir train_data/ --target_col target --drop_first_col
```

### Specify Categorical Columns
```bash
python measure_similarity.py --real data/adult/train.csv --synth_dir train_data/ --cat_cols 1,3,5,6,7,8,9,13
```

### Fetch Dataset from UCI by ID
```bash
python measure_similarity.py --uci_id 2 --synth_file train_data/ganblr_Adult_synthetic.csv
```

### Fetch Dataset from UCI by Name
```bash
python measure_similarity.py --uci_name adult --synth_dir train_data/ --output adult_similarity_results.csv
```

## Output

The script will output the average JSD for categorical columns, average WD for numerical columns, and an overall similarity score (lower is better). When comparing multiple models, the results will be sorted by the overall score.

Example output:

```
Results summary (sorted by Overall Score):
        Model  Avg JSD (Categorical)  Avg WD (Numerical)  Overall Score
      ganblr                   0.1135              0.1523        0.1329
    ganblrpp                   0.1243              0.1647        0.1445
        rlig                   0.1298              0.1702        0.1500
      ctabgan                  0.1365              0.1845        0.1605
       ctgan                   0.1487              0.1923        0.1705
```