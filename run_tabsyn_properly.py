"""
A focused script for running TabSyn properly on a dataset

This script sets up the data in TabSyn's expected format and runs the TabSyn pipeline
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Run TabSyn properly on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (csv file in train_data directory)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def prepare_tabsyn_data(dataset_path, dataset_name, verbose=False):
    """Prepare data in TabSyn's expected format"""
    if verbose:
        print(f"Preparing dataset {dataset_name} for TabSyn...")
    
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Create TabSyn directories
    data_dir = os.path.join('data', dataset_name)
    info_dir = os.path.join('data', 'Info')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    
    # Copy original data to TabSyn directory
    data_csv_path = os.path.join(data_dir, f"{dataset_name}.csv")
    data.to_csv(data_csv_path, index=False)
    
    # Identify categorical and numerical columns
    cat_col_idx = []
    num_col_idx = []
    
    for i, col in enumerate(data.columns):
        if data[col].dtype == 'object' or len(data[col].unique()) < 10:
            cat_col_idx.append(i)
        else:
            num_col_idx.append(i)
    
    # Target column (last column)
    target_col_idx = [len(data.columns) - 1]
    
    # Create metadata JSON
    task_type = "binclass" if len(data[data.columns[-1]].unique()) <= 2 else "multiclass"
    if target_col_idx[0] in num_col_idx:
        task_type = "regression"
    
    # Create a name-to-index mapping for columns
    idx_name_mapping = {i: col for i, col in enumerate(data.columns)}
    
    metadata = {
        "name": dataset_name,
        "task_type": task_type,
        "header": "infer",
        "column_names": list(data.columns),
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": data_csv_path,
        "test_path": None,
        "idx_name_mapping": idx_name_mapping
    }
    
    # Write metadata to JSON file
    json_path = os.path.join(info_dir, f"{dataset_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Also write metadata to the dataset directory (TabSyn vae/main.py looks for it here)
    data_json_path = os.path.join(data_dir, "info.json")
    with open(data_json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    if verbose:
        print(f"Dataset prepared for TabSyn at {data_dir}")
        print(f"Metadata saved to {json_path} and {data_json_path}")
        print(f"Numerical columns: {len(num_col_idx)}")
        print(f"Categorical columns: {len(cat_col_idx)}")
        print(f"Task type: {task_type}")
    
    return metadata

def process_dataset(dataset_name, verbose=False):
    """Process dataset to TabSyn's required numpy format"""
    if verbose:
        print(f"Processing dataset {dataset_name} to TabSyn's expected format...")
    
    try:
        # Run TabSyn's process_dataset.py script
        process_cmd = [
            sys.executable, 
            'process_dataset.py',
            '--dataname', dataset_name
        ]
        
        if verbose:
            print(f"Running command: {' '.join(process_cmd)}")
            
        process_proc = subprocess.run(process_cmd, capture_output=True, text=True)
        if process_proc.returncode != 0:
            if verbose:
                print(f"Dataset processing failed with error:\n{process_proc.stderr}")
            raise Exception(f"Dataset processing failed")
        
        if verbose:
            print("Dataset processed successfully")
            print(process_proc.stdout)
        
        return True
    except Exception as e:
        if verbose:
            print(f"Error processing dataset: {e}")
        return False

def train_vae(dataset_name, epochs, verbose=False):
    """Train VAE model"""
    if verbose:
        print(f"Training TabSyn VAE model for {epochs} epochs...")
    
    try:
        # Run TabSyn's VAE training
        vae_cmd = [
            sys.executable, 
            'main.py',
            '--dataname', dataset_name,
            '--method', 'vae',
            '--mode', 'train',
            '--epochs', str(epochs),
            '--gpu', '0'
        ]
        
        if verbose:
            print(f"Running command: {' '.join(vae_cmd)}")
            
        vae_proc = subprocess.run(vae_cmd, capture_output=True, text=True)
        if vae_proc.returncode != 0:
            if verbose:
                print(f"VAE training failed with error:\n{vae_proc.stderr}")
                print(f"Output: {vae_proc.stdout}")
            raise Exception(f"VAE training failed")
        
        if verbose:
            print("VAE training completed successfully")
            print(vae_proc.stdout)
        
        return True
    except Exception as e:
        if verbose:
            print(f"Error training VAE: {e}")
        return False

def train_diffusion(dataset_name, epochs, verbose=False):
    """Train diffusion model"""
    if verbose:
        print(f"Training TabSyn diffusion model for {epochs} epochs...")
    
    try:
        # Run TabSyn's diffusion training
        diffusion_cmd = [
            sys.executable, 
            'main.py',
            '--dataname', dataset_name,
            '--method', 'tabsyn',
            '--mode', 'train',
            '--epochs', str(epochs),
            '--gpu', '0'
        ]
        
        if verbose:
            print(f"Running command: {' '.join(diffusion_cmd)}")
            
        diffusion_proc = subprocess.run(diffusion_cmd, capture_output=True, text=True)
        if diffusion_proc.returncode != 0:
            if verbose:
                print(f"Diffusion training failed with error:\n{diffusion_proc.stderr}")
                print(f"Output: {diffusion_proc.stdout}")
            raise Exception(f"Diffusion training failed")
        
        if verbose:
            print("Diffusion training completed successfully")
            print(diffusion_proc.stdout)
        
        return True
    except Exception as e:
        if verbose:
            print(f"Error training diffusion model: {e}")
        return False

def sample_tabsyn(dataset_name, n_samples, output_path, verbose=False):
    """Sample from trained TabSyn model"""
    if verbose:
        print(f"Generating {n_samples} samples using TabSyn...")
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create synthetic directory
        synthetic_dir = os.path.join('synthetic', dataset_name)
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Run TabSyn's sampling
        sample_cmd = [
            sys.executable, 
            'main.py',
            '--dataname', dataset_name,
            '--method', 'tabsyn',
            '--mode', 'sample',
            '--num-samples', str(n_samples),
            '--gpu', '0',
            '--save_path', os.path.join(synthetic_dir, 'tabsyn.csv')
        ]
        
        if verbose:
            print(f"Running command: {' '.join(sample_cmd)}")
            
        sample_proc = subprocess.run(sample_cmd, capture_output=True, text=True)
        if sample_proc.returncode != 0:
            if verbose:
                print(f"Sampling failed with error:\n{sample_proc.stderr}")
                print(f"Output: {sample_proc.stdout}")
            raise Exception(f"Sampling failed")
        
        # Check if file was created
        synthetic_path = os.path.join(synthetic_dir, 'tabsyn.csv')
        if not os.path.exists(synthetic_path):
            if verbose:
                print(f"Synthetic data file not found at {synthetic_path}")
            raise Exception(f"Synthetic data file not found")
        
        # Copy to output path
        shutil.copy(synthetic_path, output_path)
        
        if verbose:
            print(f"Generated {n_samples} samples successfully")
            print(f"Synthetic data saved to {output_path}")
        
        # Read and return the synthetic data
        synthetic_data = pd.read_csv(output_path)
        return synthetic_data
    
    except Exception as e:
        if verbose:
            print(f"Error sampling from TabSyn: {e}")
        return None

def main():
    args = parse_args()
    
    # Ensure dataset name has proper suffix
    dataset_name = args.dataset.replace("_train_data.csv", "")
    dataset_path = f"train_data/{dataset_name}_train_data.csv"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    print(f"Loading dataset: {dataset_path}")
    data = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Set number of samples if not specified
    n_samples = args.samples if args.samples is not None else len(data)
    
    # Set output path if not specified
    output_path = args.output if args.output else f"train_data/tabsyn_{dataset_name}_synthetic.csv"
    
    # Change to TabSyn directory
    tabsyn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabsyn')
    original_dir = os.getcwd()
    os.chdir(tabsyn_dir)
    
    try:
        # Prepare data for TabSyn
        prepare_tabsyn_data(dataset_path, dataset_name, verbose=args.verbose)
        
        # Process dataset for TabSyn
        processed = process_dataset(dataset_name, verbose=args.verbose)
        if not processed:
            raise Exception("Failed to process dataset")
        
        # Train VAE
        vae_trained = train_vae(dataset_name, args.epochs, verbose=args.verbose)
        if not vae_trained:
            raise Exception("Failed to train VAE")
        
        # Train diffusion model
        diffusion_trained = train_diffusion(dataset_name, args.epochs, verbose=args.verbose)
        if not diffusion_trained:
            raise Exception("Failed to train diffusion model")
        
        # Sample from TabSyn
        synthetic_data = sample_tabsyn(dataset_name, n_samples, output_path, verbose=args.verbose)
        if synthetic_data is None:
            raise Exception("Failed to generate synthetic data")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nOriginal data sample:")
        print(data.head())
        print("\nSynthetic data sample:")
        print(synthetic_data.head())
        
        # Compare distributions for target variable
        print("\nDistribution of target variable:")
        target_col = data.columns[-1]
        print("Original: ", data[target_col].value_counts(normalize=True))
        print("Synthetic:", synthetic_data[target_col].value_counts(normalize=True))
        
        print("\nDone!")
        
    except Exception as e:
        print(f"Error running TabSyn: {e}")
    finally:
        # Change back to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()