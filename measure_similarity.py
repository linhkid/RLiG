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

import pandas as pd
import numpy as np
# Assuming compute_jensen_shannon_divergence and compute_wasserstein_distance
# are defined as in your original script.
def identify_categorical_columns_by_name(data_path, threshold=10, exclude_cols=None):
    """
    Identify categorical column names in a dataset.

    Args:
        data_path (str): Path to the dataset CSV.
        threshold (int): Max unique values for a column to be numerically categorical.
        exclude_cols (list): List of column names to exclude from consideration.

    Returns:
        list: Names of identified categorical columns.
    """
    df = pd.read_csv(data_path)
    categorical_col_names = []

    if exclude_cols is None:
        exclude_cols = []

    # Normalize exclude_cols to lower case for case-insensitive comparison
    exclude_cols_lower = [col.lower() for col in exclude_cols]

    for col_name in df.columns:
        if col_name.lower() in exclude_cols_lower:
            continue

        # Original script's heuristic for common target column names
        # For a truly strict version, this might also be configurable or removed
        if col_name.lower() in ['target', 'label', 'class', 'y', 'income', 'id', 'index']:
            print(
                f"Note: Column '{col_name}' matches common target/identifier names and will be skipped for automatic categorical detection.")
            continue

        is_categorical = False
        try:
            # Check if column type is object (string) or explicitly categorical
            if pd.api.types.is_string_dtype(df[col_name]) or \
                    pd.api.types.is_categorical_dtype(df[col_name]):
                is_categorical = True
            # Or if it's numerical but has few unique values (potential discrete numerical)
            elif pd.api.types.is_numeric_dtype(df[col_name]) and df[col_name].nunique() <= threshold:
                is_categorical = True
            # If not distinctly string/categorical or low-nunique numeric, it's likely numerical

        except Exception as e:
            print(f"Warning: Could not determine type or nunique for column '{col_name}': {e}. Skipping.")
            continue

        if is_categorical:
            categorical_col_names.append(col_name)

    return categorical_col_names


def measure_dataset_similarity_strict(real_data_path, synthetic_data_path, categorical_col_names=None):
    """
    Measure similarity between real and synthetic datasets, strictly matching columns by name.
    No columns are dropped implicitly. Datasets must have identical column sets.

    Args:
        real_data_path (str): Path to the real dataset CSV.
        synthetic_data_path (str): Path to the synthetic dataset CSV.
        categorical_col_names (list): Names of categorical columns.

    Returns:
        dict: Dictionary containing average JSD, average WD, overall score,
              and per-column scores by name.
    """
    real_df = pd.read_csv(real_data_path)
    synth_df = pd.read_csv(synthetic_data_path)

    # 1. Verify that column sets are identical by name
    real_cols_set = set(real_df.columns)
    synth_cols_set = set(synth_df.columns)

    if real_cols_set != synth_cols_set:
        error_message = (
            f"Error: Column sets are not identical for strict comparison.\n"
            f"Real columns count: {len(real_cols_set)}, Synthetic columns count: {len(synth_cols_set)}\n"
            f"Columns only in real: {sorted(list(real_cols_set - synth_cols_set))}\n"
            f"Columns only in synthetic: {sorted(list(synth_cols_set - real_cols_set))}\n"
            "Strict comparison requires datasets to have the exact same column names."
        )
        print(error_message)
        return {
            'avg_jsd': float('nan'),
            'avg_wd': float('nan'),
            'overall_score': float('nan'),
            'jsd_scores_by_name': {},
            'wd_scores_by_name': {},
            'error': error_message
        }

    # 2. Align synthetic dataset's column order to match the real dataset's column order
    # This ensures that when we convert to NumPy, columns at the same index correspond to the same feature.
    # We use the real dataset's column order as the canonical order.
    aligned_synth_df = synth_df[real_df.columns]

    # Convert to numpy arrays for processing
    real_data = real_df.values
    synthetic_data = aligned_synth_df.values

    n_features = real_data.shape[1]

    # 3. Determine categorical and numerical column indices based on provided names
    categorical_col_indices = []
    if categorical_col_names:
        # Create a mapping from column name to index for quick lookup
        col_name_to_idx = {name: i for i, name in enumerate(real_df.columns)}
        for col_name in categorical_col_names:
            if col_name in col_name_to_idx:
                categorical_col_indices.append(col_name_to_idx[col_name])
            else:
                print(f"Warning: Provided categorical column name '{col_name}' not found in dataset columns. Skipping.")

    numerical_col_indices = [i for i in range(n_features) if i not in categorical_col_indices]

    # --- The rest of the JSD/WD calculation logic would be similar ---
    # (Using compute_jensen_shannon_divergence and compute_wasserstein_distance)

    jsd_scores = []
    jsd_scores_map = {}
    for col_idx in categorical_col_indices:
        try:
            jsd = compute_jensen_shannon_divergence(real_data, synthetic_data, col_idx) # From original script
            jsd_scores.append(jsd)
            jsd_scores_map[real_df.columns[col_idx]] = jsd
        except Exception as e:
            print(f"Warning: Error computing JSD for column '{real_df.columns[col_idx]}' (index {col_idx}): {e}")

    wd_scores = []
    wd_scores_map = {}
    for col_idx in numerical_col_indices:
        try:
            # Pass normalize=True by default, as in original, or make it a parameter
            wd = compute_wasserstein_distance(real_data, synthetic_data, col_idx, normalize=True) # From original script
            wd_scores.append(wd)
            wd_scores_map[real_df.columns[col_idx]] = wd
        except Exception as e:
            print(f"Warning: Error computing WD for column '{real_df.columns[col_idx]}' (index {col_idx}): {e}")

    avg_jsd = np.mean(jsd_scores) if jsd_scores else 0
    avg_wd = np.mean(wd_scores) if wd_scores else 0

    overall_score = float('nan') # Default to NaN
    if jsd_scores and wd_scores:
        overall_score = 0.5 * avg_jsd + 0.5 * avg_wd
    elif jsd_scores: # Only categorical
        overall_score = avg_jsd
    elif wd_scores: # Only numerical
        overall_score = avg_wd
    # If neither, overall_score remains NaN

    return {
        'avg_jsd': avg_jsd,
        'avg_wd': avg_wd,
        'overall_score': overall_score,
        'jsd_scores_by_name': jsd_scores_map,
        'wd_scores_by_name': wd_scores_map,
        'error': None
    }

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
    
    # Convert to similarity percentages (0-100%, higher is better)
    # For JSD, 0 is identical, 1 is completely different, so similarity = (1 - JSD) * 100
    jsd_similarity = (1 - avg_jsd) * 100 if jsd_scores else 0
    
    # For WD, we use an exponential transformation to map [0, inf) to (0, 100]
    # This gives a score that decreases as WD increases
    wd_similarity = 100 * np.exp(-avg_wd) if wd_scores else 0
    
    # Overall similarity score (higher is better)
    if jsd_scores and wd_scores:
        # If we have both categorical and numerical columns
        overall_similarity = 0.5 * jsd_similarity + 0.5 * wd_similarity
    elif jsd_scores:
        # If we only have categorical columns
        overall_similarity = jsd_similarity
    elif wd_scores:
        # If we only have numerical columns
        overall_similarity = wd_similarity
    else:
        # If we couldn't compute any scores
        overall_similarity = float('nan')
    
    # For backwards compatibility, also keep the original distance metrics
    overall_score = 0.5 * avg_jsd + 0.5 * avg_wd if (jsd_scores and wd_scores) else (avg_jsd if jsd_scores else avg_wd)
    
    return {
        'avg_jsd': avg_jsd,
        'avg_wd': avg_wd,
        'overall_score': overall_score,
        'jsd_scores': jsd_scores,
        'wd_scores': wd_scores,
        'jsd_similarity': jsd_similarity,
        'wd_similarity': wd_similarity,
        'overall_similarity': overall_similarity
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
                'JSD (Distance)': similarity['avg_jsd'],
                'WD (Distance)': similarity['avg_wd'],
                'Categorical Similarity (%)': similarity['jsd_similarity'],
                'Numerical Similarity (%)': similarity['wd_similarity'],
                'Overall Similarity (%)': similarity['overall_similarity']
            })
            
            print(f"Processed {model_name}: Similarity={similarity['overall_similarity']:.2f}%")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Convert to DataFrame and sort by overall similarity (higher is better)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Overall Similarity (%)', ascending=False)
        
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
        'tictactoe': 101,
        'connect4': 26,
        'chess': 22,
        'abalone': 1,
        'bank': 222,
        'nursery': 76,
        'maternal-health': 863,
        'room-occupancy': 864,
        'room_occupancy': 864,
        'occupancy': 864,
        'connect-4': 30,
        'credit-approval': 144,
        'credit': 144,
        'default': 350,
        'ecoli': 39,
        'magic': 159,
        'poker': 158,
        'pokerhand': 158,
        'poker-hand': 158,
        'rice': 545,
        'letter-recognition': 59,
        'letter_recognition': 59,
        'letter_recog': 59,
        'nsl-kdd': 235,
        'nslkdd': 235,
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


# def main():
#     parser = argparse.ArgumentParser(description='Measure similarity between real and synthetic datasets')
#     parser.add_argument('--real', type=str, help='Path to the real dataset CSV')
#     parser.add_argument('--uci_id', type=int, help='UCI ML Repository dataset ID to use as real data')
#     parser.add_argument('--uci_name', type=str, help='UCI ML Repository dataset name to use as real data')
#     parser.add_argument('--synth_dir', type=str, help='Directory containing synthetic dataset CSVs')
#     parser.add_argument('--synth_file', type=str, help='Path to a specific synthetic dataset CSV')
#     parser.add_argument('--output', type=str, help='Path to save the comparison results')
#     parser.add_argument('--cat_cols', type=str, help='Comma-separated list of categorical column indices (0-based)')
#     parser.add_argument('--cat_threshold', type=int, default=10, help='Threshold for automatic categorical column detection')
#     parser.add_argument('--target_col', type=str, help='Name of the target column to exclude from comparison')
#     parser.add_argument('--drop_first_col', action='store_true', help='Whether to drop the first column (often an index)')
#     parser.add_argument('--tmp_dir', type=str, help='Directory to save temporary files (for UCI downloads)')
#
#     args = parser.parse_args()
#
#     # Handle real dataset - either from file path or UCI repository
#     real_data_path = args.real
#
#     if real_data_path is None:
#         if args.uci_id is not None:
#             real_data_path = fetch_uci_dataset_by_id(args.uci_id, args.target_col, args.tmp_dir)
#         elif args.uci_name is not None:
#             real_data_path = fetch_uci_dataset_by_name(args.uci_name, args.target_col, args.tmp_dir)
#         else:
#             parser.error("Either --real, --uci_id, or --uci_name must be provided to specify the real dataset")
#             return
#
#     if real_data_path is None:
#         print("Failed to obtain real dataset. Exiting.")
#         return
#
#     # Parse categorical columns if provided
#     categorical_cols = None
#     if args.cat_cols:
#         categorical_cols = [int(i) for i in args.cat_cols.split(',')]
#
#     # Compare a single synthetic dataset with the real one
#     if args.synth_file:
#         if categorical_cols is None:
#             categorical_cols = identify_categorical_columns(real_data_path, args.cat_threshold, args.drop_first_col)
#
#         similarity = measure_dataset_similarity(
#             real_data_path,
#             args.synth_file,
#             categorical_cols,
#             args.target_col,
#             args.drop_first_col
#         )
#
#         print(f"Similarity between real data and {os.path.basename(args.synth_file)}:")
#         print(f"Distance metrics (lower is better):")
#         print(f"  JSD for categorical features: {similarity['avg_jsd']:.4f}")
#         print(f"  WD for numerical features: {similarity['avg_wd']:.4f}")
#
#         print(f"\nSimilarity percentages (higher is better):")
#         print(f"  Categorical similarity: {similarity['jsd_similarity']:.2f}%")
#         print(f"  Numerical similarity: {similarity['wd_similarity']:.2f}%")
#         print(f"  Overall similarity: {similarity['overall_similarity']:.2f}%")
#
#         # Save detailed results if output file is provided
#         if args.output:
#             pd.DataFrame([{
#                 'Model': os.path.basename(args.synth_file).replace("_synthetic.csv", ""),
#                 'JSD (Distance)': similarity['avg_jsd'],
#                 'WD (Distance)': similarity['avg_wd'],
#                 'Categorical Similarity (%)': similarity['jsd_similarity'],
#                 'Numerical Similarity (%)': similarity['wd_similarity'],
#                 'Overall Similarity (%)': similarity['overall_similarity']
#             }]).to_csv(args.output, index=False)
#
#     # Compare multiple synthetic datasets with the real one
#     elif args.synth_dir:
#         results = compare_models(
#             real_data_path,
#             args.synth_dir,
#             args.output,
#             categorical_cols,
#             args.cat_threshold,
#             args.target_col,
#             args.drop_first_col
#         )
#
#         if results is not None and not results.empty:
#             print("\nResults summary (sorted by Overall Score):")
#             print(results.to_string(index=False))
#
#     else:
#         print("Error: Either --synth_file or --synth_dir must be provided")
#         parser.print_help()

def main():
    parser = argparse.ArgumentParser(description='Measure similarity between real and synthetic datasets.')

    # Real data source
    group_real = parser.add_mutually_exclusive_group(required=True)
    group_real.add_argument('--real', type=str, help='Path to the real dataset CSV')
    group_real.add_argument('--uci_id', type=int, help='UCI ML Repository dataset ID for real data')
    group_real.add_argument('--uci_name', type=str, help='UCI ML Repository dataset name for real data')

    # Synthetic data source
    group_synth = parser.add_mutually_exclusive_group(required=True)
    group_synth.add_argument('--synth_file', type=str, help='Path to a single synthetic dataset CSV')
    group_synth.add_argument('--synth_dir', type=str, help='Directory containing synthetic dataset CSVs')

    # Output and general options
    parser.add_argument('--output', type=str, help='Path to save comparison results CSV')
    parser.add_argument('--cat_threshold', type=int, default=10,
                        help='Threshold for automatic categorical column detection (unique values). Default: 10.')
    parser.add_argument('--tmp_dir', type=str, help='Directory for temporary files (e.g., UCI downloads)')

    # Mode selection
    parser.add_argument('--strict', action='store_true',
                        help='Enable strict comparison mode (column names must match, no implicit drops/trims).')

    # Original mode arguments
    parser.add_argument('--cat_cols', type=str,
                        help='(Original mode) Comma-separated indices of categorical columns (0-based).')
    parser.add_argument('--target_col', type=str,
                        help='(Original mode) Name of target column to exclude from comparison.')
    parser.add_argument('--drop_first_col', action='store_true',
                        help='(Original mode) Drop the first column (assumed index).')

    # Strict mode arguments
    parser.add_argument('--cat_col_names', type=str,
                        help='(Strict mode) Comma-separated names of categorical columns.')
    parser.add_argument('--exclude_cols_cat_detect', type=str,
                        help='(Strict mode) Comma-separated names of columns to exclude from auto categorical detection (e.g., target).')

    args = parser.parse_args()

    # Resolve real_data_path
    real_data_path = args.real
    if args.uci_id is not None:
        if not UCI_SUPPORT: parser.error("UCI_ID provided, but ucimlrepo is not installed. Please install it.")
        real_data_path = fetch_uci_dataset_by_id(args.uci_id, args.target_col,
                                                 args.tmp_dir)  # target_col for filename in original
    elif args.uci_name is not None:
        if not UCI_SUPPORT: parser.error("UCI_NAME provided, but ucimlrepo is not installed. Please install it.")
        real_data_path = fetch_uci_dataset_by_name(args.uci_name, args.target_col,
                                                   args.tmp_dir)  # target_col for filename in original

    if real_data_path is None or not os.path.exists(real_data_path):
        parser.error(f"Real data path could not be resolved or does not exist: {real_data_path}")
        return

    if args.strict:
        # --- Strict Mode ---
        print("--- Running in STRICT comparison mode ---")
        if args.target_col: print("Warning (Strict Mode): --target_col is ignored in strict mode.")
        if args.drop_first_col: print("Warning (Strict Mode): --drop_first_col is ignored in strict mode.")
        if args.cat_cols: print(
            "Warning (Strict Mode): --cat_cols (indices) is ignored. Use --cat_col_names for strict mode.")

        cat_col_names_list = [name.strip() for name in args.cat_col_names.split(',')] if args.cat_col_names else None
        exclude_cols_list = [name.strip() for name in
                             args.exclude_cols_cat_detect.split(',')] if args.exclude_cols_cat_detect else None

        if args.synth_file:
            # Auto-detect cat_col_names if not provided for single file comparison
            if cat_col_names_list is None:
                print("Auto-detecting categorical columns (by name) for strict single file comparison...")
                cat_col_names_list = identify_categorical_columns_by_name(real_data_path, args.cat_threshold,
                                                                          exclude_cols_list)
                print(f"Identified {len(cat_col_names_list)} categorical columns (names): {cat_col_names_list}")

            similarity = measure_dataset_similarity_strict(
                real_data_path,
                args.synth_file,
                categorical_col_names=cat_col_names_list
            )
            print(f"\nStrict Similarity between real data and {os.path.basename(args.synth_file)}:")
            if similarity.get('error'):
                print(f"Error: {similarity['error']}")
            else:
                print(f"  Average JSD (Categorical): {similarity['avg_jsd']:.4f}")
                print(f"  Average WD (Numerical): {similarity['avg_wd']:.4f}")
                print(f"  Overall Score: {similarity['overall_score']:.4f}")
                # print(f"  JSD Scores by Name: {similarity['jsd_scores_by_name']}") # Optional: too verbose
                # print(f"  WD Scores by Name: {similarity['wd_scores_by_name']}")   # Optional: too verbose

            if args.output:
                df_out = pd.DataFrame([{
                    'Model': os.path.basename(args.synth_file).replace("_synthetic.csv", ""),
                    'Avg JSD (Categorical)': similarity['avg_jsd'],
                    'Avg WD (Numerical)': similarity['avg_wd'],
                    'Overall Score': similarity['overall_score'],
                    'Error': similarity.get('error')
                }])
                df_out.to_csv(args.output, index=False)
                print(f"Strict single comparison results saved to {args.output}")

        elif args.synth_dir:
            results_df = compare_models_strict(
                real_data_path,
                args.synth_dir,
                args.output,
                categorical_col_names=cat_col_names_list,
                # Pass directly, compare_models_strict handles None for auto-detection
                categorical_threshold=args.cat_threshold,
                exclude_cols_from_cat_detection=exclude_cols_list
            )
            if results_df is not None and not results_df.empty:
                print("\nStrict Results summary (sorted by Overall Score):")
                print(results_df.to_string(index=False))
        else:  # Should not happen due to mutually_exclusive_group
            parser.error("Error: --synth_file or --synth_dir must be provided.")

    else:
        # --- Original Mode ---
        print("--- Running in ORIGINAL comparison mode ---")
        if args.cat_col_names: print(
            "Warning (Original Mode): --cat_col_names is ignored. Use --cat_cols (indices) for original mode.")
        if args.exclude_cols_cat_detect: print("Warning (Original Mode): --exclude_cols_cat_detect is ignored.")

        cat_cols_list = [int(i.strip()) for i in args.cat_cols.split(',')] if args.cat_cols else None

        if args.synth_file:
            # Auto-detect cat_cols (indices) if not provided for single file comparison
            if cat_cols_list is None:
                print("Auto-detecting categorical columns (by index) for original single file comparison...")
                cat_cols_list = identify_categorical_columns(real_data_path, args.cat_threshold, args.drop_first_col)
                print(f"Identified {len(cat_cols_list)} categorical columns (indices): {cat_cols_list}")

            similarity = measure_dataset_similarity(
                real_data_path,
                args.synth_file,
                categorical_cols=cat_cols_list,
                target_col=args.target_col,
                drop_first_col=args.drop_first_col
            )
            print(f"\nOriginal Similarity between real data and {os.path.basename(args.synth_file)}:")
            print(f"  Average JSD (Categorical): {similarity['avg_jsd']:.4f}")
            print(f"  Average WD (Numerical): {similarity['avg_wd']:.4f}")
            print(f"  Overall Score: {similarity['overall_score']:.4f}")

            if args.output:
                pd.DataFrame([{
                    'Model': os.path.basename(args.synth_file).replace("_synthetic.csv", ""),
                    'Avg JSD (Categorical)': similarity['avg_jsd'],
                    'Avg WD (Numerical)': similarity['avg_wd'],
                    'Overall Score': similarity['overall_score']
                }]).to_csv(args.output, index=False)
                print(f"Original single comparison results saved to {args.output}")

        elif args.synth_dir:
            results_df = compare_models(
                real_data_path,
                args.synth_dir,
                args.output,
                categorical_cols=cat_cols_list,  # Pass directly, compare_models handles None for auto-detection
                categorical_threshold=args.cat_threshold,
                target_col=args.target_col,
                drop_first_col=args.drop_first_col
            )
            if results_df is not None and not results_df.empty:
                print("\nOriginal Results summary (sorted by Overall Score):")
                print(results_df.to_string(index=False))
        else:  # Should not happen
            parser.error("Error: --synth_file or --synth_dir must be provided.")


if __name__ == "__main__":
    main()