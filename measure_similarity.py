"""
Standalone script to measure similarity between synthetic and real datasets using
Jensen-Shannon Divergence (JSD) and Wasserstein Distance (WD).

Based on the paper "GANBLR: GAN-Based Bayesian learning for classification using realistically-created synthetic data"
"""

import os
import numpy as np
import pandas as pd
import argparse
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import glob
import tempfile
import requests
from io import BytesIO
import zipfile
import sys

# Optional imports for UCI data fetching
try:
    from ucimlrepo import fetch_ucirepo
    UCI_SUPPORT = True
except ImportError:
    UCI_SUPPORT = False
    print("Warning: ucimlrepo not installed. UCI dataset fetching will not be available.")
    print("To enable this feature, run: pip install ucimlrepo")

def compute_wasserstein_distance(real_data, synthetic_data, column_idx, normalize=True):
    """
    Compute Wasserstein distance between real and synthetic data for a numerical column.
    
    Args:
        real_data (numpy.ndarray): Real data
        synthetic_data (numpy.ndarray): Synthetic data
        column_idx (int): Index of the column to compute distance for
        normalize (bool): Whether to normalize the data before computing distance
    
    Returns:
        float: Wasserstein distance
    """
    real_col = real_data[:, column_idx]
    synth_col = synthetic_data[:, column_idx]
    
    if normalize:
        # Scale the data between 0 and 1 to get normalized distance
        scaler = MinMaxScaler()
        scaler.fit(real_col.reshape(-1, 1))
        real_col_scaled = scaler.transform(real_col.reshape(-1, 1)).flatten()
        synth_col_scaled = scaler.transform(synth_col.reshape(-1, 1)).flatten()
        return wasserstein_distance(real_col_scaled, synth_col_scaled)
    else:
        return wasserstein_distance(real_col, synth_col)

def compute_jensen_shannon_divergence(real_data, synthetic_data, column_idx):
    """
    Compute Jensen-Shannon Divergence between real and synthetic data for a categorical column.
    
    Args:
        real_data (numpy.ndarray): Real data
        synthetic_data (numpy.ndarray): Synthetic data
        column_idx (int): Index of the column to compute divergence for
    
    Returns:
        float: Jensen-Shannon Divergence
    """
    # Convert values to strings to ensure consistent handling
    real_values = [str(x) for x in real_data[:, column_idx]]
    synth_values = [str(x) for x in synthetic_data[:, column_idx]]
    
    # Compute probability mass functions for both distributions
    real_pmf = Counter(real_values)
    real_pmf = {k: v/len(real_values) for k, v in real_pmf.items()}
    
    synth_pmf = Counter(synth_values)
    synth_pmf = {k: v/len(synth_values) for k, v in synth_pmf.items()}
    
    # Get all unique categories
    all_categories = sorted(set(list(real_pmf.keys()) + list(synth_pmf.keys())))
    
    # Create ordered PMFs ensuring all categories are represented
    real_pmf_ordered = [real_pmf.get(cat, 0) for cat in all_categories]
    synth_pmf_ordered = [synth_pmf.get(cat, 0) for cat in all_categories]
    
    # Compute Jensen-Shannon Divergence
    return distance.jensenshannon(real_pmf_ordered, synth_pmf_ordered, base=2.0)

def measure_dataset_similarity(real_data_path, synthetic_data_path, categorical_cols=None, 
                           target_col=None, drop_first_col=False):
    """
    Measure similarity between real and synthetic datasets using JSD and WD metrics.
    
    Args:
        real_data_path (str): Path to the real dataset CSV
        synthetic_data_path (str): Path to the synthetic dataset CSV
        categorical_cols (list): Indices of categorical columns (0-based)
        target_col (str): Name of the target column, if different between datasets
        drop_first_col (bool): Whether to drop the first column (index column)
    
    Returns:
        dict: Dictionary containing average JSD for categorical columns, 
              average WD for numerical columns, and overall score
    """
    # Load the datasets
    real_df = pd.read_csv(real_data_path)
    synth_df = pd.read_csv(synthetic_data_path)
    
    # Handle index column if needed
    if drop_first_col:
        real_df = real_df.iloc[:, 1:]
        synth_df = synth_df.iloc[:, 1:]
    
    # Ensure target column is handled correctly
    if target_col:
        # Find target column in real dataset
        real_target_col = [col for col in real_df.columns if target_col.lower() in col.lower()]
        if real_target_col:
            real_df = real_df.drop(columns=real_target_col)
        
        # Find target column in synthetic dataset
        synth_target_col = [col for col in synth_df.columns if target_col.lower() in col.lower()]
        if synth_target_col:
            synth_df = synth_df.drop(columns=synth_target_col)
    
    # Check column counts and adjust if necessary
    if real_df.shape[1] != synth_df.shape[1]:
        print(f"Warning: Real data has {real_df.shape[1]} columns, synthetic data has {synth_df.shape[1]} columns.")
        # If synthetic data has more columns, trim to match real data column count
        if synth_df.shape[1] > real_df.shape[1]:
            synth_df = synth_df.iloc[:, :real_df.shape[1]]
            print(f"Trimmed synthetic data to {synth_df.shape[1]} columns.")
        # If real data has more columns, trim to match synthetic data column count
        else:
            real_df = real_df.iloc[:, :synth_df.shape[1]]
            print(f"Trimmed real data to {real_df.shape[1]} columns.")
    
    # Convert to numpy arrays for processing
    real_data = real_df.values
    synthetic_data = synth_df.values
    
    n_features = real_data.shape[1]
    
    # If categorical_cols is None, assume all columns are numerical
    if categorical_cols is None:
        categorical_cols = []
    
    # Calculate JSD for categorical columns
    jsd_scores = []
    for col_idx in categorical_cols:
        if col_idx < real_data.shape[1]:  # Make sure column index is valid
            try:
                jsd = compute_jensen_shannon_divergence(real_data, synthetic_data, col_idx)
                jsd_scores.append(jsd)
            except Exception as e:
                print(f"Warning: Error computing JSD for column {col_idx}: {e}")
    
    # Calculate WD for numerical columns
    numerical_cols = [i for i in range(n_features) if i not in categorical_cols]
    wd_scores = []
    for col_idx in numerical_cols:
        try:
            wd = compute_wasserstein_distance(real_data, synthetic_data, col_idx)
            wd_scores.append(wd)
        except Exception as e:
            print(f"Warning: Error computing WD for column {col_idx}: {e}")
    
    # Calculate averages
    avg_jsd = np.mean(jsd_scores) if jsd_scores else 0
    avg_wd = np.mean(wd_scores) if wd_scores else 0
    
    # Overall score (lower is better)
    if jsd_scores and wd_scores:
        # If we have both categorical and numerical columns
        overall_score = 0.5 * avg_jsd + 0.5 * avg_wd
    elif jsd_scores:
        # If we only have categorical columns
        overall_score = avg_jsd
    elif wd_scores:
        # If we only have numerical columns
        overall_score = avg_wd
    else:
        # If we couldn't compute any scores
        overall_score = float('nan')
    
    return {
        'avg_jsd': avg_jsd,
        'avg_wd': avg_wd,
        'overall_score': overall_score,
        'jsd_scores': jsd_scores,
        'wd_scores': wd_scores
    }

def identify_categorical_columns(data_path, threshold=10, drop_first_col=False):
    """
    Identify categorical columns in a dataset based on the number of unique values.
    
    Args:
        data_path (str): Path to the dataset CSV
        threshold (int): Maximum number of unique values for a column to be considered categorical
        drop_first_col (bool): Whether to drop the first column before analysis
        
    Returns:
        list: Indices of categorical columns
    """
    df = pd.read_csv(data_path)
    
    if drop_first_col and df.shape[1] > 1:
        df = df.iloc[:, 1:]
    
    categorical_cols = []
    
    for i, col in enumerate(df.columns):
        # Skip columns that are likely targets
        if col.lower() in ['target', 'label', 'class', 'y', 'income']:
            continue
            
        # Check if column type is object or string or has few unique values
        is_categorical = False
        try:
            n_unique = df[col].nunique()
            if pd.api.types.is_string_dtype(df[col]):
                is_categorical = True
            elif pd.api.types.is_categorical_dtype(df[col]):
                is_categorical = True
            elif n_unique <= threshold:
                is_categorical = True
        except:
            # If there's an error computing nunique, skip this column
            continue
            
        if is_categorical:
            # Adjust index if first column was dropped
            col_idx = i if not drop_first_col else i
            categorical_cols.append(col_idx)
    
    return categorical_cols

def compare_models(real_data_path, synthetic_data_dir, output_file=None, 
                categorical_cols=None, categorical_threshold=10, 
                target_col=None, drop_first_col=False):
    """
    Compare multiple synthetic datasets generated by different models against a real dataset.
    
    Args:
        real_data_path (str): Path to the real dataset CSV
        synthetic_data_dir (str): Directory containing synthetic dataset CSVs
        output_file (str): Path to save the comparison results
        categorical_cols (list): Indices of categorical columns (0-based)
        categorical_threshold (int): Threshold for automatic categorical column detection
        target_col (str): Name of the target column to exclude from comparison
        drop_first_col (bool): Whether to drop the first column (index column)
        
    Returns:
        pandas.DataFrame: Comparison results
    """
    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = identify_categorical_columns(real_data_path, categorical_threshold, drop_first_col)
    
    print(f"Identified {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # Find all synthetic datasets
    synthetic_files = glob.glob(os.path.join(synthetic_data_dir, "*.csv"))
    
    if not synthetic_files:
        print(f"No synthetic data files found in {synthetic_data_dir}")
        return None
    
    # Prepare results container
    results = []
    
    for synth_file in synthetic_files:
        model_name = os.path.basename(synth_file).replace("_synthetic.csv", "")
        
        try:
            # Measure similarity
            similarity = measure_dataset_similarity(
                real_data_path, 
                synth_file, 
                categorical_cols,
                target_col,
                drop_first_col
            )
            
            # Add to results
            results.append({
                'Model': model_name,
                'Avg JSD (Categorical)': similarity['avg_jsd'],
                'Avg WD (Numerical)': similarity['avg_wd'],
                'Overall Score': similarity['overall_score']
            })
            
            print(f"Processed {model_name}: JSD={similarity['avg_jsd']:.4f}, WD={similarity['avg_wd']:.4f}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Convert to DataFrame and sort by overall score
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Overall Score')
        
        # Save results if output file is provided
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
    
    return results_df

def fetch_uci_dataset_by_id(uci_id, target_col=None, tmp_dir=None):
    """
    Fetch a dataset from the UCI ML Repository using its numeric ID.
    
    Args:
        uci_id (int): The UCI dataset ID
        target_col (str): Name of the target column
        tmp_dir (str): Directory to save the downloaded dataset
        
    Returns:
        str: Path to the downloaded dataset CSV
    """
    if not UCI_SUPPORT:
        print("Error: UCI dataset fetching is not available. Please install ucimlrepo using:")
        print("pip install ucimlrepo")
        return None
        
    try:
        # Fetch dataset using ucimlrepo library
        dataset = fetch_ucirepo(id=uci_id)
        
        # Get features and target
        X = dataset.data.features
        y = dataset.data.targets
        
        # Combine features and target
        if target_col is None:
            target_col = 'target'
        
        # If target has a column name, use it
        if hasattr(y, 'columns') and len(y.columns) > 0:
            target_col = y.columns[0]
        
        df = pd.concat([X, y], axis=1)
        
        # Create temporary directory if not provided
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        
        # Save to CSV
        output_path = os.path.join(tmp_dir, f"uci_dataset_{uci_id}.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Downloaded UCI dataset (ID: {uci_id}) to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error fetching UCI dataset with ID {uci_id}: {e}")
        return None

def fetch_uci_dataset_by_name(dataset_name, target_col=None, tmp_dir=None):
    """
    Fetch a dataset from the UCI ML Repository using its name.
    
    Args:
        dataset_name (str): The UCI dataset name (e.g., 'adult', 'iris')
        target_col (str): Name of the target column
        tmp_dir (str): Directory to save the downloaded dataset
        
    Returns:
        str: Path to the downloaded dataset CSV
    """
    if not UCI_SUPPORT:
        print("Error: UCI dataset fetching is not available. Please install ucimlrepo using:")
        print("pip install ucimlrepo")
        return None
        
    # Dictionary mapping common dataset names to UCI IDs
    uci_dataset_map = {
        'adult': 2,
        'iris': 53,
        'wine': 109,
        'car': 19,
        'breast-cancer': 13,
        'diabetes': 37,
        'glass': 42,
        'heart-disease': 45,
        'ionosphere': 52,
        'letter': 59,
        'liver-disorders': 60,
        'mushroom': 73,
        'sonar': 151,
        'vehicle': 89,
        'wine-quality': 186,
        'zoo': 111,
        'abalone': 1,
        'bank': 222,
        'nursery': 76,
        'chess': 24,
        'connect-4': 30,
        'credit-approval': 144,
        'ecoli': 39,
        'magic': 159,
        'poker': 158,
        'rice': 545,
    }
    
    # Normalize dataset name for lookup
    normalized_name = dataset_name.lower().replace('_', '-')
    
    # Check if dataset name exists in our mapping
    if normalized_name in uci_dataset_map:
        uci_id = uci_dataset_map[normalized_name]
        return fetch_uci_dataset_by_id(uci_id, target_col, tmp_dir)
    else:
        # Try direct API search or other methods
        print(f"Dataset '{dataset_name}' not found in the predefined mapping.")
        print(f"Attempting to search for '{dataset_name}' in UCI repository...")
        
        try:
            # This is a simplified search as the ucimlrepo library doesn't support search by name
            # So we try to fetch by approximating the ID based on the name
            datasets = [name for name in uci_dataset_map.keys() if normalized_name in name]
            if datasets:
                print(f"Found possible matches: {datasets}")
                return fetch_uci_dataset_by_id(uci_dataset_map[datasets[0]], target_col, tmp_dir)
            else:
                print(f"No matching datasets found. Please provide a valid dataset name or ID.")
                return None
        except Exception as e:
            print(f"Error searching for dataset '{dataset_name}': {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Measure similarity between real and synthetic datasets')
    parser.add_argument('--real', type=str, help='Path to the real dataset CSV')
    parser.add_argument('--uci_id', type=int, help='UCI ML Repository dataset ID to use as real data')
    parser.add_argument('--uci_name', type=str, help='UCI ML Repository dataset name to use as real data')
    parser.add_argument('--synth_dir', type=str, help='Directory containing synthetic dataset CSVs')
    parser.add_argument('--synth_file', type=str, help='Path to a specific synthetic dataset CSV')
    parser.add_argument('--output', type=str, help='Path to save the comparison results')
    parser.add_argument('--cat_cols', type=str, help='Comma-separated list of categorical column indices (0-based)')
    parser.add_argument('--cat_threshold', type=int, default=10, help='Threshold for automatic categorical column detection')
    parser.add_argument('--target_col', type=str, help='Name of the target column to exclude from comparison')
    parser.add_argument('--drop_first_col', action='store_true', help='Whether to drop the first column (often an index)')
    parser.add_argument('--tmp_dir', type=str, help='Directory to save temporary files (for UCI downloads)')
    
    args = parser.parse_args()
    
    # Handle real dataset - either from file path or UCI repository
    real_data_path = args.real
    
    if real_data_path is None:
        if args.uci_id is not None:
            real_data_path = fetch_uci_dataset_by_id(args.uci_id, args.target_col, args.tmp_dir)
        elif args.uci_name is not None:
            real_data_path = fetch_uci_dataset_by_name(args.uci_name, args.target_col, args.tmp_dir)
        else:
            parser.error("Either --real, --uci_id, or --uci_name must be provided to specify the real dataset")
            return
        
    if real_data_path is None:
        print("Failed to obtain real dataset. Exiting.")
        return
    
    # Parse categorical columns if provided
    categorical_cols = None
    if args.cat_cols:
        categorical_cols = [int(i) for i in args.cat_cols.split(',')]
    
    # Compare a single synthetic dataset with the real one
    if args.synth_file:
        if categorical_cols is None:
            categorical_cols = identify_categorical_columns(args.real, args.cat_threshold, args.drop_first_col)
        
        similarity = measure_dataset_similarity(
            args.real, 
            args.synth_file, 
            categorical_cols,
            args.target_col,
            args.drop_first_col
        )
        
        print(f"Similarity between real data and {os.path.basename(args.synth_file)}:")
        print(f"Average JSD (Categorical): {similarity['avg_jsd']:.4f}")
        print(f"Average WD (Numerical): {similarity['avg_wd']:.4f}")
        print(f"Overall Score: {similarity['overall_score']:.4f}")
        
        # Save detailed results if output file is provided
        if args.output:
            pd.DataFrame([{
                'Model': os.path.basename(args.synth_file).replace("_synthetic.csv", ""),
                'Avg JSD (Categorical)': similarity['avg_jsd'],
                'Avg WD (Numerical)': similarity['avg_wd'], 
                'Overall Score': similarity['overall_score']
            }]).to_csv(args.output, index=False)
    
    # Compare multiple synthetic datasets with the real one
    elif args.synth_dir:
        results = compare_models(
            args.real, 
            args.synth_dir, 
            args.output, 
            categorical_cols,
            args.cat_threshold,
            args.target_col,
            args.drop_first_col
        )
        
        if results is not None and not results.empty:
            print("\nResults summary (sorted by Overall Score):")
            print(results.to_string(index=False))
    
    else:
        print("Error: Either --synth_file or --synth_dir must be provided")
        parser.print_help()

if __name__ == "__main__":
    main()