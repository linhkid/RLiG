import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tabsyn_proper_wrapper import TabSynWrapper

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


def main(args):
    """Main function to test TabSyn wrapper"""
    # Load dataset
    X, y = load_dataset(args.dataset, uci_id=args.uci, verbose=args.verbose)
    
    # Preprocess data
    data = preprocess_data(X, y, discretize=args.discretize, verbose=args.verbose)
    
    # Initialize TabSyn wrapper
    tabsyn = TabSynWrapper(
        dataset_name=args.dataset,
        epochs=args.epochs,
        gpu=args.gpu,
        nfe=args.nfe
    )
    
    # Fit TabSyn
    if args.verbose:
        print(f"Fitting TabSyn on {args.dataset}...")
    
    success = tabsyn.fit(X, y)
    
    if not success:
        print("TabSyn training failed.")
        return
    
    # Generate samples
    if args.verbose:
        print(f"Generating {args.n_samples} samples...")
    
    synthetic_data = tabsyn.sample(args.n_samples)
    
    if synthetic_data is None:
        print("Failed to generate synthetic data.")
        return
    
    # Save synthetic data
    output_dir = 'train_data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'tabsyn_{args.dataset}_synthetic.csv')
    synthetic_data.to_csv(output_path, index=False)
    
    if args.verbose:
        print(f"Saved synthetic data to {output_path}")
        print(f"Synthetic data shape: {synthetic_data.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test TabSyn wrapper')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='adult', help='Dataset name')
    parser.add_argument('--uci', type=int, default=None, help='UCI dataset ID (if applicable)')
    
    # TabSyn parameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--nfe', type=int, default=50, help='Number of function evaluations')
    
    # Sampling parameters
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to generate')
    
    # Other options
    parser.add_argument('--discretize', action='store_true', help='Discretize continuous target variable')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    main(args)