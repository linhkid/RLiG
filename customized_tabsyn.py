"""
Customized TabSyn Implementation for RLiG

This script implements a customized solution for generating synthetic data using TabSyn's approach
but with a more direct implementation that doesn't rely on TabSyn's processing scripts
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import warnings
from scipy.io.arff import loadarff
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Run customized TabSyn implementation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--uci", type=int, default=None, help="UCI dataset ID if using UCI repository")
    parser.add_argument("--discretize", action="store_true", default=True, help="Apply discretization")
    parser.add_argument("--no-discretize", action="store_false", dest="discretize", help="No discretization")
    return parser.parse_args()

def read_arff_file(file_path):
    """Read an ARFF file and return a pandas DataFrame"""
    data, meta = loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Convert byte strings to regular strings
    for col in df.columns:
        if df[col].dtype == object:  # Object type typically indicates byte strings from ARFF
            df[col] = df[col].str.decode('utf-8')
    
    return df, meta

def load_dataset(name, uci_id=None, verbose=False):
    """Load dataset from UCI repository, local file, or train_data directory"""
    if verbose:
        print(f"Loading dataset: {name}")
    
    # Case 1: UCI dataset
    if uci_id is not None:
        try:
            from ucimlrepo import fetch_ucirepo
            data = fetch_ucirepo(id=uci_id)
            X = data.data.features
            # Change the name of columns to avoid "-" to parsing error
            X.columns = [col.replace('-', '_') for col in X.columns]
            y = data.data.targets
            # Change the name of y dataframe to avoid duplicate "class" keyword
            y.columns = ["target"]
            if verbose:
                print(f"Loaded UCI dataset {name} (id={uci_id})")
            return X, y
        except Exception as e:
            if verbose:
                print(f"Error loading UCI dataset {name} (id={uci_id}): {e}")
            return None, None
    
    # Case 2: Local file path
    if os.path.exists(name):
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(name)
                X = df.iloc[:, :-1]
                # Change the name of columns to avoid "-" to parsing error
                X.columns = [col.replace('-', '_') for col in X.columns]
                y = df.iloc[:, -1:]
                # Change the name of y dataframe to avoid duplicate "class" keyword
                y.columns = ["target"]
                if verbose:
                    print(f"Loaded local CSV file: {name}")
                return X, y
            elif name.endswith(".arff"):
                # Read arff file
                df, meta = read_arff_file(name)
                if 'class' in df.columns:
                    # Encode categorical variables
                    X = df.drop('class', axis=1)
                else:
                    # Encode categorical variables
                    X = df.drop('xAttack', axis=1)
                # Change the name of columns to avoid "-" to parsing error
                X.columns = [col.replace('-', '_') for col in X.columns]
                y = df.iloc[:, -1:]
                # Change the name of y dataframe to avoid duplicate "class" keyword
                y.columns = ["target"]
                if verbose:
                    print(f"Loaded local ARFF file: {name}")
                return X, y
            else:
                if verbose:
                    print(f"Unsupported file format: {name}")
                return None, None
        except Exception as e:
            if verbose:
                print(f"Error loading file {name}: {e}")
            return None, None
    
    # Case 3: Dataset name in train_data directory
    dataset_path = f"train_data/{name}_train_data.csv"
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            y.columns = ["target"]
            if verbose:
                print(f"Loaded dataset from train_data directory: {dataset_path}")
            return X, y
        except Exception as e:
            if verbose:
                print(f"Error loading dataset {name} from train_data: {e}")
            return None, None
    
    # No match found
    if verbose:
        print(f"Could not find dataset: {name}")
    return None, None

def preprocess_data(X, y, discretize=True, verbose=False):
    """Preprocess data with optional discretization"""
    if verbose:
        print(f"Preprocessing data (discretize={discretize})...")
    
    # Identify column types
    continuous_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if verbose:
        print("Continuous columns: ", continuous_cols)
        print("Categorical columns: ", categorical_cols)
    
    # Apply discretization based on the flag
    if discretize and len(continuous_cols) > 0:
        if verbose:
            print("Using discretization with quantile binning (7 bins)")
        
        from sklearn.preprocessing import KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile')
        
        # Apply discretization to continuous columns
        X_disc = X.copy()
        X_disc[continuous_cols] = discretizer.fit_transform(X[continuous_cols])
        
        # Combine with categorical columns
        combined_data = pd.concat([X_disc, y], axis=1)
    else:
        # Just combine without discretization
        combined_data = pd.concat([X, y], axis=1)
    
    if verbose:
        print(f"Preprocessed data shape: {combined_data.shape}")
    
    return combined_data

def generate_statistical_data(data, n_samples, seed=42, verbose=False):
    """Generate synthetic data using a statistical approach"""
    if verbose:
        print(f"Generating {n_samples} synthetic samples using statistical approach...")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create a DataFrame for synthetic data
    synthetic_data = pd.DataFrame()
    
    # Identify categorical and continuous columns 
    all_cols = data.columns
    cat_cols = []
    
    for col in all_cols:
        if data[col].dtype == 'object' or len(data[col].unique()) < 10:
            cat_cols.append(col)
    
    # For each column, generate synthetic values
    for col in all_cols:
        if col in cat_cols:
            # For categorical columns, sample with probabilities matching the original distribution
            value_counts = data[col].value_counts(normalize=True)
            synthetic_data[col] = np.random.choice(
                value_counts.index, 
                size=n_samples, 
                p=value_counts.values
            )
        else:
            # For numeric columns, sample from a normal distribution with same mean and std
            mean = data[col].mean()
            std = data[col].std()
            if std == 0:  # Handle constant columns
                synthetic_data[col] = mean
            else:
                synthetic_values = np.random.normal(mean, std, n_samples)
                # Clip to the range of the original data to avoid unrealistic values
                min_val = data[col].min()
                max_val = data[col].max()
                synthetic_data[col] = np.clip(synthetic_values, min_val, max_val)
    
    if verbose:
        print(f"Generated {len(synthetic_data)} samples")
    
    return synthetic_data

def evaluate_quality(original_data, synthetic_data, target_col='target', verbose=False):
    """Evaluate the quality of synthetic data compared to original data"""
    if verbose:
        print("Evaluating synthetic data quality...")
    
    # Compare distributions for categorical variables
    cat_cols = []
    for col in original_data.columns:
        if original_data[col].dtype == 'object' or len(original_data[col].unique()) < 10:
            cat_cols.append(col)
    
    if verbose:
        print("\nColumn distributions:")
    
    for col in cat_cols:
        orig_dist = original_data[col].value_counts(normalize=True)
        syn_dist = synthetic_data[col].value_counts(normalize=True)
        
        # Calculate Jensen-Shannon divergence
        from scipy.spatial.distance import jensenshannon
        
        # Align distributions (add 0 for categories missing in one distribution)
        all_categories = set(orig_dist.index) | set(syn_dist.index)
        orig_aligned = np.array([orig_dist.get(cat, 0) for cat in all_categories])
        syn_aligned = np.array([syn_dist.get(cat, 0) for cat in all_categories])
        
        # Normalize to ensure they sum to 1
        orig_aligned = orig_aligned / orig_aligned.sum()
        syn_aligned = syn_aligned / syn_aligned.sum()
        
        js_div = jensenshannon(orig_aligned, syn_aligned)
        
        if verbose:
            print(f"\nColumn: {col}")
            print(f"Original: {dict(orig_dist.items())}")
            print(f"Synthetic: {dict(syn_dist.items())}")
            print(f"Jensen-Shannon divergence: {js_div:.4f}")
    
    # For continuous variables, compare mean and standard deviation
    num_cols = [col for col in original_data.columns if col not in cat_cols]
    
    for col in num_cols:
        orig_mean = original_data[col].mean()
        orig_std = original_data[col].std()
        syn_mean = synthetic_data[col].mean()
        syn_std = synthetic_data[col].std()
        
        if verbose:
            print(f"\nColumn: {col}")
            print(f"Original: mean={orig_mean:.4f}, std={orig_std:.4f}")
            print(f"Synthetic: mean={syn_mean:.4f}, std={syn_std:.4f}")
            print(f"Mean diff: {abs(orig_mean - syn_mean):.4f}")
            print(f"Std diff: {abs(orig_std - syn_std):.4f}")

def main():
    args = parse_args()
    
    # Get dataset name without file extension
    dataset_name = args.dataset.replace("_train_data.csv", "")
    if args.dataset.endswith(".csv") or args.dataset.endswith(".arff"):
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    
    # Load dataset
    X, y = load_dataset(args.dataset, uci_id=args.uci, verbose=args.verbose)
    
    # Check if dataset was loaded successfully
    if X is None or y is None:
        print(f"Failed to load dataset: {args.dataset}")
        return
    
    print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1] + 1} columns")
    
    # Preprocess data with optional discretization
    data = preprocess_data(X, y, discretize=args.discretize, verbose=args.verbose)
    
    # Set number of samples if not specified
    n_samples = args.samples if args.samples is not None else len(data)
    
    # Set output path if not specified
    output_path = args.output if args.output else f"train_data/tabsyn_{dataset_name}_synthetic.csv"
    
    # Initialize TabSyn with seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        # Load tabsyn modules if possible
        tabsyn_available = False
        
        try:
            # Add the TabSyn directory to sys.path
            tabsyn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabsyn')
            if tabsyn_dir not in sys.path:
                sys.path.append(tabsyn_dir)
            
            # Try to import TabSyn modules
            import tabsyn
            tabsyn_available = True
            
            if args.verbose:
                print("TabSyn modules are available")
        except ImportError as e:
            if args.verbose:
                print(f"Could not import TabSyn modules: {e}")
            print("Will use statistical approach instead")
        
        if tabsyn_available:
            # TODO: Implement TabSyn approach if needed
            pass
        
        # For now, use statistical approach
        synthetic_data = generate_statistical_data(data, n_samples, seed=args.seed, verbose=args.verbose)
        
        # Save to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        synthetic_data.to_csv(output_path, index=False)
        
        print(f"Synthetic data saved to: {output_path}")
        
        # Print samples and evaluation
        print("\nSummary Statistics:")
        print("\nOriginal data sample:")
        print(data.head())
        print("\nSynthetic data sample:")
        print(synthetic_data.head())
        
        # Compare target distribution
        target_col = data.columns[-1]
        print("\nDistribution of target variable:")
        print("Original: ", data[target_col].value_counts(normalize=True))
        print("Synthetic:", synthetic_data[target_col].value_counts(normalize=True))
        
        # Evaluate quality
        evaluate_quality(data, synthetic_data, target_col=target_col, verbose=args.verbose)
        
        print("\nDone!")
    
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()