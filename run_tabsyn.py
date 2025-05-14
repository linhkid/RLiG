"""
Run TabSyn on a specified dataset

This script provides a simple way to run TabSyn on a specified dataset
following the official TabSyn workflow.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tabsyn_wrapper import TabSynWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Run TabSyn on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (csv file in train_data directory)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: train_data/tabsyn_{dataset}_synthetic.csv)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

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
    
    # Identify categorical columns
    categorical_columns = []
    for col in data.columns:
        if data[col].dtype == 'object' or len(data[col].unique()) < 10:
            categorical_columns.append(col)
    
    print(f"Identified {len(categorical_columns)} categorical columns: {categorical_columns}")
    
    # Set number of samples if not specified
    n_samples = args.samples if args.samples is not None else len(data)
    
    # Initialize and train TabSyn model
    print(f"Initializing TabSyn with random seed {args.seed}")
    wrapper = TabSynWrapper(
        train_data=data,
        categorical_columns=categorical_columns,
        epochs=args.epochs,
        verbose=args.verbose,
        random_seed=args.seed
    )
    
    print("Training TabSyn model...")
    wrapper.fit()
    
    # Generate synthetic data
    print(f"Generating {n_samples} synthetic samples...")
    synthetic_data = wrapper.sample(n_samples=n_samples)
    
    # Save synthetic data
    output_path = args.output if args.output else f"train_data/tabsyn_{dataset_name}_synthetic.csv"
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nOriginal data sample:")
    print(data.head())
    print("\nSynthetic data sample:")
    print(synthetic_data.head())
    
    # Compare distributions for a few columns
    print("\nDistribution of target variable:")
    target_col = data.columns[-1]
    print("Original: ", data[target_col].value_counts(normalize=True))
    print("Synthetic:", synthetic_data[target_col].value_counts(normalize=True))
    
    print("\nDone!")

if __name__ == "__main__":
    main()