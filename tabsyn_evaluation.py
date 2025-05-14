"""
Integration script to run TabSyn through the RLiG evaluation pipeline.

This script uses our TabSyn wrapper to generate synthetic data
for all datasets and then evaluates them using the existing
evaluation pipeline.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tabsyn_proper_wrapper import TabSynWrapper
from sklearn.model_selection import train_test_split

# List of datasets to evaluate
DATASETS = [
    'adult', 
    'car', 
    'chess', 
    'magic', 
    'nursery', 
    'letter-recog', 
    'poker-hand'
]

def load_dataset(dataset_name, uci_id=None, verbose=True):
    """Load a dataset by name"""
    if verbose:
        print(f"Loading dataset: {dataset_name}")
    
    # Handle all file paths
    if dataset_name == 'adult':
        file_path = 'data/adult.arff'
    elif dataset_name == 'car':
        file_path = 'data/car.arff'
    elif dataset_name == 'chess':
        file_path = 'data/chess.arff'
    elif dataset_name == 'magic':
        file_path = 'data/magic.arff'
    elif dataset_name == 'nursery':
        file_path = 'data/nursery.arff'
    elif dataset_name == 'letter-recog':
        file_path = 'data/letter-recog.arff'
    elif dataset_name == 'poker-hand':
        file_path = 'data/poker-hand.arff'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Simple handling for ARFF files
    if file_path.endswith('.arff'):
        with open(file_path, 'r') as f:
            data_section = False
            header_lines = []
            data_lines = []
            
            for line in f:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                
                if line.upper() == '@DATA':
                    data_section = True
                    continue
                
                if data_section:
                    data_lines.append(line)
                else:
                    header_lines.append(line)
            
            # Parse attributes
            attributes = []
            for line in header_lines:
                if line.upper().startswith('@ATTRIBUTE'):
                    parts = line.split(maxsplit=2)
                    if len(parts) >= 2:
                        attr_name = parts[1].strip("'\" ")
                        attributes.append(attr_name)
            
            # Parse data
            X_data = []
            for line in data_lines:
                if ',' in line:
                    values = line.split(',')
                    X_data.append(values)
            
            data = pd.DataFrame(X_data, columns=attributes)
            
            # Convert numeric columns
            for col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    pass  # Keep as string if not numeric
            
            # Separate features and target (last column)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            return X, y
    
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_data(X, y, discretize=False, verbose=True):
    """Preprocess data for TabSyn"""
    if verbose:
        print("Preprocessing data...")
    
    # Convert categorical columns to strings
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str)
    
    # Handle target variable
    if isinstance(y, pd.Series):
        y = y.to_frame()
    elif isinstance(y, np.ndarray) and y.ndim == 1:
        y = pd.DataFrame(y, columns=['target'])
    
    # If target is numerical and discretize=True, use quantiles
    if discretize and pd.api.types.is_numeric_dtype(y.iloc[:, 0]):
        if verbose:
            print("Discretizing target variable...")
        # Convert to quantile-based categories (e.g., 5 bins)
        y_arr = y.iloc[:, 0].to_numpy()
        bins = np.percentile(y_arr, [0, 20, 40, 60, 80, 100])
        y_binned = np.digitize(y_arr, bins)
        y = pd.DataFrame(y_binned, columns=y.columns)
    
    # Combine X and y
    data = pd.concat([X, y], axis=1)
    
    if verbose:
        print(f"Data shape: {data.shape}")
        print(f"Number of categorical features: {len(X.select_dtypes(exclude=['number']).columns)}")
        print(f"Number of numerical features: {len(X.select_dtypes(include=['number']).columns)}")
    
    return data

def generate_synthetic_data(dataset_name, epochs=500, gpu=0, nfe=50, discretize=False, verbose=True):
    """Generate synthetic data for a dataset using TabSyn"""
    if verbose:
        print(f"=== Generating synthetic data for {dataset_name} ===")
    
    # Load dataset
    X, y = load_dataset(dataset_name, verbose=verbose)
    
    # Preprocess data
    data = preprocess_data(X, y, discretize=discretize, verbose=verbose)
    
    # Initialize TabSyn wrapper
    tabsyn = TabSynWrapper(
        dataset_name=dataset_name,
        epochs=epochs,
        gpu=gpu,
        nfe=nfe
    )
    
    # Fit TabSyn
    if verbose:
        print(f"Fitting TabSyn on {dataset_name}...")
    
    success = tabsyn.fit(X, y)
    
    if not success:
        print(f"TabSyn training failed for {dataset_name}.")
        return None
    
    # Generate samples (same number as original dataset)
    if verbose:
        print(f"Generating {len(data)} samples...")
    
    synthetic_data = tabsyn.sample(len(data))
    
    if synthetic_data is None:
        print(f"Failed to generate synthetic data for {dataset_name}.")
        return None
    
    # Save synthetic data
    output_dir = 'train_data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'tabsyn_{dataset_name}_synthetic.csv')
    synthetic_data.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Saved synthetic data to {output_path}")
        print(f"Synthetic data shape: {synthetic_data.shape}")
    
    return synthetic_data

def run_evaluation(results_dir='results', verbose=True):
    """Run the evaluation pipeline on generated datasets"""
    if verbose:
        print("=== Running evaluation on synthetic datasets ===")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the evaluation script
    cmd = f"python eval_tstr_final.py --model tabsyn --output_dir {results_dir}"
    
    if verbose:
        print(f"Running: {cmd}")
    
    os.system(cmd)
    
    if verbose:
        print("Evaluation complete!")

def main(args):
    """Main function to generate and evaluate synthetic data"""
    # Filter datasets based on command line arguments
    if args.datasets:
        datasets = [d for d in args.datasets if d in DATASETS]
    else:
        datasets = DATASETS
    
    if args.exclude:
        datasets = [d for d in datasets if d not in args.exclude]
    
    if not datasets:
        print("No valid datasets selected.")
        return
    
    if args.verbose:
        print(f"Selected datasets: {datasets}")
    
    # Generate synthetic data for each dataset
    for dataset in datasets:
        synthetic_data = generate_synthetic_data(
            dataset_name=dataset,
            epochs=args.epochs,
            gpu=args.gpu,
            nfe=args.nfe,
            discretize=args.discretize,
            verbose=args.verbose
        )
        
        if synthetic_data is None:
            print(f"Skipping evaluation for {dataset} due to generation failure.")
            continue
    
    # Run evaluation
    if not args.skip_eval:
        run_evaluation(results_dir=args.results_dir, verbose=args.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TabSyn Evaluation Pipeline')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', help='Datasets to process (leave empty for all)')
    parser.add_argument('--exclude', nargs='+', help='Datasets to exclude')
    
    # TabSyn parameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--nfe', type=int, default=50, help='Number of function evaluations')
    
    # Evaluation options
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation, only generate data')
    
    # Other options
    parser.add_argument('--discretize', action='store_true', help='Discretize continuous target variables')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    main(args)