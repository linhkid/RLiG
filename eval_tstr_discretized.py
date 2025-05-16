"""
TSTR (Train on Synthetic, Test on Real) Evaluation Framework with Discretization

This script implements a proper TSTR evaluation for generative models with explicit discretization
of continuous variables using quantile-based binning (7 bins).

This version is specifically designed to improve performance for models that work well
with discretized data, particularly GReaT and TabSyn, while maintaining a fair comparison
across all models. Quantile-based binning is used instead of uniform binning to better
preserve the distribution of the original data, which is important for transformer-based 
models and statistical models.

Models evaluated:
- RLiG: Reinforcement Learning inspired Generative Bayesian Networks
- GANBLR: Generative Adversarial Network with Bayesian Networks using Tree Search structure
- GANBLR++: Generative Adversarial Network with Bayesian Networks using Hill Climbing structure
- CTGAN: Conditional Tabular GAN
- NaiveBayes: Simple baseline generative model
- GReaT: Generation of Realistic Tabular data with transformers
- TabSyn: Tabular data synthesis with statistical modeling

The TSTR methodology:
1. Train a generative model on real data (with discretized continuous variables)
2. Generate synthetic data from the trained model 
3. Train classification models (LR, MLP, RF) on the synthetic data
4. Test these classification models on real test data
5. Measure accuracy (how well models trained on synthetic data perform on real data)
"""

import os
import gc
import torch
import time
import warnings
import logging
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.io.arff import loadarff

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}. Using device: {device}")
# Suppress warnings and verbose logs
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pgmpy').setLevel(logging.ERROR)

# scikit-learn imports
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# XGBoost for additional classifier
try:
    import xgboost as xgb
    # Check compatibility with sklearn wrapper
    try:
        xgb_test = xgb.XGBClassifier(n_estimators=5)
        hasattr(xgb_test, 'fit')  # Test if basic interface is available
        XGBOOST_AVAILABLE = True
    except (ImportError, AttributeError, TypeError) as e:
        print(f"XGBoost available but sklearn wrapper has compatibility issues: {e}")
        XGBOOST_AVAILABLE = False
except ImportError:
    print("XGBoost not available. Will use other classifiers only.")
    XGBOOST_AVAILABLE = False
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Check for required packages
try:
    from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.metrics import structure_score
    PGMPY_AVAILABLE = True
except ImportError:
    print("pgmpy not available. Bayesian Network models will be skipped.")
    PGMPY_AVAILABLE = False

try:
    from ganblr.models import RLiG
    RLIG_AVAILABLE = True
except ImportError:
    print("RLiG not available. Will be skipped.")
    RLIG_AVAILABLE = False

try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except ImportError:
    print("CTGAN not available. CTGAN model will be skipped.")
    CTGAN_AVAILABLE = False

try:
    # Add the current directory to the path to make imports work
    import sys
    if '.' not in sys.path:
        sys.path.append('.')
    from ctabgan.model.ctabgan import CTABGAN
    CTABGAN_AVAILABLE = True
except ImportError as e:
    print(f"CTABGAN not available. CTABGAN model will be skipped. Error: {e}")
    CTABGAN_AVAILABLE = False

try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    print("ucimlrepo not available. Will use local datasets.")
    UCI_AVAILABLE = False

try:
    from be_great import GReaT
    GREAT_AVAILABLE = True
except ImportError:
    print("GReaT is not available. Will be skipped.")
    GREAT_AVAILABLE = False

try:
    # Add tabsyn directory to Python path so imports work
    import sys
    import os
    tabsyn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabsyn')
    if tabsyn_path not in sys.path:
        sys.path.append(tabsyn_path)
    
    from tabsyn.tabular_gan import TabularGAN
    TABSYN_AVAILABLE = True
except ImportError as e:
    print(f"TabSyn is not available. Will be skipped. Error: {e}")
    TABSYN_AVAILABLE = False


# ============= DATA HANDLING FUNCTIONS =============

def read_arff_file(file_path):
    """Read an ARFF file and return a pandas DataFrame"""
    data, meta = loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Convert byte strings to regular strings
    for col in df.columns:
        if df[col].dtype == object:  # Object type typically indicates byte strings from ARFF
            df[col] = df[col].str.decode('utf-8')
    
    return df, meta


def label_encode_cols(X, cols):
    """Label encode categorical columns"""
    X_encoded = X.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le
    return X_encoded, encoders


def preprocess_data(X, y, discretize=True, model_name=None, cv_fold=None, n_folds=None):
    """Preprocess data: optionally discretize continuous variables and encode categoricals
    
    This version can selectively apply discretization using quantile binning with 7 bins,
    which better preserves the distribution of the original data. This is especially useful
    for certain models, while others may perform better with non-discretized data.
    
    Parameters:
    -----------
    X : DataFrame
        Features to preprocess
    y : DataFrame or Series
        Target variable
    discretize : bool, default=True
        Whether to apply discretization to continuous features
    model_name : str, optional
        Name of the model being trained, used for model-specific preprocessing decisions
    cv_fold : int, optional
        Current fold number when doing k-fold cross-validation (0-indexed)
    n_folds : int, optional
        Total number of folds when doing k-fold cross-validation
    """
    # First, handle missing values
    # Check if there are any missing values
    if X.isnull().any().any():
        print("Handling missing values in the dataset...")
        
        # For categorical columns, fill with the most frequent value
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0])
            
        # For numeric columns, fill with the median
        for col in X.select_dtypes(include=['number']).columns:
            X[col] = X[col].fillna(X[col].median())
            
        print("Missing values have been imputed")
    
    # Identify column types after imputation
    continuous_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    print("Continuous columns: ", continuous_cols)
    print("Categorical columns: ", categorical_cols)
    
    # Apply discretization based on the flag
    apply_discretization = discretize
    
    # Log the discretization status for the current model
    if model_name:
        if apply_discretization:
            print(f"Note: Using discretized features for {model_name}")
        else:
            print(f"Note: Using non-discretized features for {model_name}")
    
    # Create transformation pipeline with optional discretization
    transformers = []
    if len(continuous_cols) > 0:
        if apply_discretization:
            # Add discretization step to the pipeline
            continuous_transformer = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
            ])
            print("Using discretization with uniform binning (5 bins)")
        else:
            # Only standardize without discretization
            continuous_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            print("Using standardization without discretization")
            
        transformers.append(('num', continuous_transformer, continuous_cols))
    
    # Handle categorical columns
    if len(categorical_cols) > 0:
        X, encoders = label_encode_cols(X, categorical_cols)

    # Apply transformations
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=continuous_cols.tolist() + categorical_cols.tolist())

    # Handle target variable
    if y.isnull().any().any():
        print("Handling missing values in target variable...")
        if y.dtypes[0] == 'object':
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.median())
    
    if y.dtypes[0] == 'object':
        label_encoder = LabelEncoder()
        y_transformed = pd.DataFrame(label_encoder.fit_transform(y.values.ravel()), columns=y.columns)
    else:
        y_transformed = y
    
    # Split data based on whether we're using cross-validation or traditional train-test split
    if cv_fold is not None and n_folds is not None:
        from sklearn.model_selection import KFold
        print(f"Using {n_folds}-fold cross-validation (fold {cv_fold+1}/{n_folds})")
        
        # Create fold indices
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Convert to arrays for indexing
        X_array = X_transformed_df.values
        y_array = y_transformed.values
        
        # Get the train/test indices for this fold
        train_indices = []
        test_indices = []
        
        for i, (train_idx, test_idx) in enumerate(kf.split(X_array)):
            if i == cv_fold:
                train_indices = train_idx
                test_indices = test_idx
                break
        
        # Split the data using the indices
        X_train = pd.DataFrame(X_array[train_indices], columns=X_transformed_df.columns)
        X_test = pd.DataFrame(X_array[test_indices], columns=X_transformed_df.columns)
        y_train = pd.DataFrame(y_array[train_indices], columns=y_transformed.columns)
        y_test = pd.DataFrame(y_array[test_indices], columns=y_transformed.columns)
        
        return X_train, X_test, y_train, y_test
    else:
        # Traditional split
        return train_test_split(X_transformed_df, y_transformed, test_size=0.2, random_state=42, stratify=y)


def load_dataset(name, dataset_info):
    """Load dataset from UCI repository or local file"""
    if isinstance(dataset_info, int) and UCI_AVAILABLE:
        try:
            data = fetch_ucirepo(id=dataset_info)
            X = data.data.features
            # Change the name of columns to avoid "-" to parsing error
            X.columns = [col.replace('-', '_') for col in X.columns]
            y = data.data.targets
            # Change the name of y dataframe to avoid duplicate "class" keyword
            y.columns = ["target"]
            
            # Special handling for Credit dataset which is known to have NaN values
            if name == "Credit":
                print(f"Special handling for {name} dataset")
                # Check for missing values
                missing_X = X.isnull().sum().sum()
                missing_y = y.isnull().sum().sum()
                print(f"Missing values detected: {missing_X} in features, {missing_y} in target")
                
                # Drop rows with NaN values if there are not too many
                if missing_X + missing_y > 0 and missing_X + missing_y < len(X) * 0.1:  # If less than 10% are missing
                    print(f"Dropping {missing_X + missing_y} rows with missing values")
                    # Combine X and y for dropping rows with any NaN
                    combined = pd.concat([X, y], axis=1)
                    combined_clean = combined.dropna()
                    
                    # Split back to X and y
                    X = combined_clean.iloc[:, :-1]
                    y = combined_clean.iloc[:, -1:].copy()
                    y.columns = ["target"]
                    
                    print(f"After dropping rows: X shape = {X.shape}, y shape = {y.shape}")
            
            return X, y
        except Exception as e:
            print(f"Error loading UCI dataset {name} (id={dataset_info}): {e}")
            return None, None
    elif isinstance(dataset_info, str):
        try:
            if dataset_info.endswith(".csv"):
                df = pd.read_csv(dataset_info)
                
                # Check for NaN values
                if df.isnull().any().any():
                    print(f"Dataset {name} has missing values. Handling...")
                    # For Credit Card dataset specifically
                    if "Credit" in name or "credit" in name or "UCI_Credit_Card.csv" in dataset_info:
                        print(f"Special handling for Credit dataset")
                        # Drop rows with NaN if there are not too many
                        missing_count = df.isnull().sum().sum()
                        if missing_count < len(df) * 0.1:  # If less than 10% are missing
                            print(f"Dropping {missing_count} rows with missing values")
                            df = df.dropna()
                        # Otherwise, we'll use imputation in the preprocess_data function
                
                X = df.iloc[:, :-1]
                # Change the name of columns to avoid "-" to parsing error
                X.columns = [col.replace('-', '_') for col in X.columns]
                y = df.iloc[:, -1:]
                # Change the name of y dataframe to avoid duplicate "class" keyword
                y.columns = ["target"]
                return X, y
            else:
                # Read arff file
                df, meta = read_arff_file(dataset_info)
                
                # Check for NaN values
                if df.isnull().any().any():
                    print(f"ARFF file {name} has missing values. Handling...")
                    # Drop rows with NaN if there are not too many
                    missing_count = df.isnull().sum().sum()
                    if missing_count < len(df) * 0.1:  # If less than 10% are missing
                        print(f"Dropping {missing_count} rows with missing values")
                        df = df.dropna()
                    # Otherwise, we'll use imputation in the preprocess_data function
                
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
                return X, y
        except Exception as e:
            print(f"Error loading dataset from file {dataset_info}: {e}")
            return None, None
    else:
        print(f"Invalid dataset specification for {name}")
        return None, None


# ============= MODEL TRAINING FUNCTIONS =============

def train_bn(model, data):
    """Train a Bayesian Network model"""
    if not PGMPY_AVAILABLE:
        return None
        
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(model.nodes())
    bn.add_edges_from(model.edges())
    print("Bayesian Network structure:", bn)
    
    # Fit model using Maximum Likelihood Estimation
    try:
        bn.fit(data, estimator=MaximumLikelihoodEstimator)
        return bn
    except Exception as e:
        print(f"Error fitting Bayesian Network: {e}")
        return None


def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes model"""
    nb = GaussianNB()
    try:
        nb.fit(X_train, y_train.values.ravel())
        return nb
    except Exception as e:
        print(f"Error training Naive Bayes: {e}")
        return None


def train_ctgan(X_train, discrete_columns=None, epochs=100, batch_size=500):
    """Train a CTGAN model with M1/M2 Mac compatibility fixes"""
    if not CTGAN_AVAILABLE:
        return None
        
    try:
        import os
        # Set environment variables to limit TensorFlow memory usage
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
        
        # For Apple Silicon compatibility
        if hasattr(os, 'uname') and os.uname().machine == 'arm64':
            print("Apple Silicon detected - using compatibility settings for CTGAN")
            # Reduce batch size further for M1/M2 Macs
            batch_size = min(batch_size, 200)
            # Use fewer epochs by default on M1/M2
            epochs = min(epochs, 20)
            
        # Convert X_train to DataFrame if it's not already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        

        # Small: =< 15,000 rows
        # Medium: 20,000 to 50,000 rows
        # Large: >= 50,000 rows
        
        # For large datasets, use stratified sampling
        if len(X_train) >= 50000:  # Only sub-sample for large datasets per paper definition
            print(f"Large dataset detected ({len(X_train)} rows). Using stratified sampling for CTGAN training.")
            
            # Get target column if it exists in the dataframe
            target_col = None
            for col in X_train.columns:
                if col.lower() in ['target', 'label', 'class', 'y']:
                    target_col = col
                    break
            
            # If we have a target column, use stratified sampling
            if target_col:
                # Use sklearn's stratified sampling
                from sklearn.model_selection import train_test_split
                
                # Get features and target
                X = X_train.drop(columns=[target_col])
                y = X_train[target_col]
                
                # For large datasets, take 30% or 25,000 samples, whichever is larger
                sample_size = max(25000, int(0.3 * len(X_train)))
                _, X_sampled, _, y_sampled = train_test_split(
                    X, y, 
                    test_size=min(sample_size / len(X_train), 0.5),  # Take at most 50% of data
                    stratify=y,
                    random_state=42
                )
                
                # Recombine features and target
                X_train = pd.concat([X_sampled, y_sampled], axis=1)
                print(f"Using {len(X_train)} stratified samples for CTGAN training (as per paper's methodology).")
            else:
                # If no target column, use simple random sampling
                sample_size = max(25000, int(0.3 * len(X_train)))
                X_train = X_train.sample(min(sample_size, len(X_train)), random_state=42)
                print(f"Using {len(X_train)} random samples for CTGAN training (as per paper's methodology).")
        
        # Identify categorical columns if not provided
        if discrete_columns is None:
            discrete_columns = []
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or len(np.unique(X_train[col])) < 10:
                    discrete_columns.append(col)
        
        # Initialize CTGAN model with conservative settings
        ctgan_model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        print(f"Training CTGAN with {epochs} epochs, batch_size={batch_size}, categorical columns: {discrete_columns}")
        print("This may take a while. To skip CTGAN, use --models ganblr ganblr++ nb rlig")
        
        # Train with reduced data
        ctgan_model.fit(X_train, discrete_columns)
        return ctgan_model
    except Exception as e:
        print(f"Error training CTGAN model: {e}")
        return None

def train_ctabgan(X_train, y_train, categorical_columns=None, epochs=50):
    """Train a CTABGAN model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.DataFrame or Series
        Training targets
    categorical_columns : list, optional
        List of categorical column names
    epochs : int
        Number of training epochs
    """
    if not CTABGAN_AVAILABLE:
        return None
        
    try:
        # Small: < 15,000 rows
        # Medium: 20,000 to 50,000 rows
        # Large: >= 50,000 rows
        
        # For large datasets, use stratified sampling
        if len(X_train) >= 50000:  # Only subsample for large datasets per paper definition
            print(f"Large dataset detected ({len(X_train)} rows). Using stratified sampling for CTABGAN training.")
            
            # Standard practice for these models is to use a large enough sample
            # that represents the data distribution well
            
            # Check class distribution
            if isinstance(y_train, pd.DataFrame):
                y_values = y_train.iloc[:, 0]
            else:
                y_values = y_train
                
            class_counts = y_values.value_counts()
            min_class_count = class_counts.min()
            
            # Use sklearn's stratified sampling
            from sklearn.model_selection import train_test_split
            
            # Calculate appropriate sample size (20% of data or at least 10,000 samples)
            sample_size = max(10000, int(0.2 * len(X_train)))
            
            # Ensure we have at least 5 samples from each class
            if min_class_count < 5:
                print(f"Warning: Minimum class count is very low ({min_class_count}). Using special sampling strategy.")
                
                try:
                    # Keep all instances of rare classes and sample from common classes
                    rare_classes = class_counts[class_counts < 5].index.tolist()
                    
                    # If there are too many rare classes, ensure at least 2 samples per class
                    if len(rare_classes) > 50:  # More than 50 rare classes is unusual
                        indices = []
                        
                        # First, ensure at least 2 samples from each class
                        for class_val in class_counts.index:
                            class_indices = y_values[y_values == class_val].index.tolist()
                            # Sample with replacement if needed
                            if len(class_indices) == 1:
                                indices.extend([class_indices[0], class_indices[0]])  # Duplicate the single instance
                            else:
                                # Take at least 2 samples
                                n_samples = max(2, min(5, len(class_indices)))  # Take 2-5 samples from each class
                                sample_indices = np.random.choice(class_indices, size=n_samples, replace=False)
                                indices.extend(sample_indices)
                        
                        # Fill the rest with stratified samples up to desired sample size
                        if len(indices) < sample_size:
                            # Create mask for indices already selected
                            selected_mask = np.zeros(len(X_train), dtype=bool)
                            selected_mask[indices] = True
                            
                            # Get remaining data
                            X_remaining = X_train[~selected_mask]
                            y_remaining = y_values[~selected_mask]
                            
                            if len(y_remaining) > 0:
                                # Calculate how many more samples we need
                                remaining_size = min(sample_size - len(indices), len(X_remaining))
                                
                                if remaining_size > 0:
                                    # Use stratified sampling for the rest
                                    _, X_extra, _, y_extra = train_test_split(
                                        X_remaining, y_remaining,
                                        test_size=remaining_size/len(X_remaining),
                                        stratify=y_remaining if len(np.unique(y_remaining)) > 1 else None,
                                        random_state=42
                                    )
                                    
                                    # Combine with our selected indices
                                    extra_indices = X_extra.index.tolist()
                                    indices.extend(extra_indices)
                    else:
                        # Get indices of all rare classes
                        rare_indices = y_values[y_values.isin(rare_classes)].index.tolist()
                        
                        # Get indices of common classes
                        common_classes = [c for c in class_counts.index if c not in rare_classes]
                        common_indices = y_values[y_values.isin(common_classes)].index.tolist()
                        
                        # Determine how many samples to take from common classes
                        remaining_slots = sample_size - len(rare_indices)
                        
                        if remaining_slots > 0 and common_indices:
                            # Use stratified sampling for common classes
                            common_y = y_values.loc[common_indices]
                            _, sampled_common_indices = train_test_split(
                                common_indices,
                                test_size=min(remaining_slots/len(common_indices), 1.0),
                                stratify=common_y if len(np.unique(common_y)) > 1 else None,
                                random_state=42
                            )
                            
                            # Combine rare and sampled common indices
                            indices = rare_indices + sampled_common_indices.tolist()
                        else:
                            # Just use the rare indices
                            indices = rare_indices
                    
                    # Sample the data
                    X_train = X_train.loc[indices]
                    if isinstance(y_train, pd.DataFrame):
                        y_train = y_train.loc[indices]
                    else:
                        y_train = y_train.loc[indices]
                    
                    print(f"Using {len(X_train)} samples with special class balancing for CTABGAN training.")
                except Exception as e:
                    print(f"Error during special sampling: {e}")
                    # Fall back to regular stratified sampling
                    X_indices = list(range(len(X_train)))
                    _, sampled_indices = train_test_split(
                        X_indices, 
                        test_size=min(sample_size/len(X_train), 0.8),
                        stratify=y_values if len(np.unique(y_values)) > 1 else None,
                        random_state=42
                    )
                    
                    X_train = X_train.iloc[sampled_indices]
                    if isinstance(y_train, pd.DataFrame):
                        y_train = y_train.iloc[sampled_indices]
                    else:
                        y_train = y_train.iloc[sampled_indices]
                    
                    print(f"Using {len(X_train)} stratified samples for CTABGAN training.")
            else:
                # Normal stratified sampling
                X_indices = list(range(len(X_train)))
                _, sampled_indices = train_test_split(
                    X_indices, 
                    test_size=min(sample_size/len(X_train), 0.8),
                    stratify=y_values if len(np.unique(y_values)) > 1 else None,
                    random_state=42
                )
                
                X_train = X_train.iloc[sampled_indices]
                if isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[sampled_indices]
                else:
                    y_train = y_train.iloc[sampled_indices]
                
                print(f"Using {len(X_train)} stratified samples for CTABGAN training.")
        
        # Convert X_train to DataFrame if it's not already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            
        # Combine X and y
        target_name = y_train.name if hasattr(y_train, 'name') else "target"
        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] == 1:
                target_name = y_train.columns[0]
                train_data = pd.concat([X_train, y_train], axis=1)
            else:
                # If y has multiple columns, use the first one
                target_name = y_train.columns[0]
                train_data = pd.concat([X_train, y_train.iloc[:, 0]], axis=1)
        else:
            # Convert Series to DataFrame
            y_df = pd.DataFrame(y_train, columns=[target_name])
            train_data = pd.concat([X_train, y_df], axis=1)
        
        # Create a temporary CSV file to use with CTABGAN
        temp_csv_path = "temp_train_data.csv"
        train_data.to_csv(temp_csv_path, index=False)
        
        # Use fewer epochs for Apple Silicon compatibility
        import os
        if hasattr(os, 'uname') and os.uname().machine == 'arm64':
            print("Apple Silicon detected - using reduced settings for CTABGAN")
            epochs = min(epochs, 5)  # Even fewer epochs on M1/M2
        
        # Identify categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = []
            for col in train_data.columns:
                if train_data[col].dtype == 'object' or len(np.unique(train_data[col])) < 10:
                    categorical_columns.append(col)
        
        # Identify integer columns
        integer_columns = []
        for col in train_data.columns:
            if col not in categorical_columns and pd.api.types.is_integer_dtype(train_data[col]):
                integer_columns.append(col)
        
        # Initialize CTABGAN model
        print(f"Training CTABGAN with {epochs} epochs")
        print(f"Categorical columns: {categorical_columns}")
        print(f"Integer columns: {integer_columns}")
        
        # Define problem type (classification by default)
        problem_type = {"Classification": target_name}
        
        # Check for class counts before training
        if isinstance(y_train, pd.DataFrame):
            target_values = y_train.iloc[:, 0]
        else:
            target_values = y_train
            
        class_counts = target_values.value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < 2:
            print(f"Warning: Target class {class_counts.idxmin()} has only {min_class_count} instances.")
            print("Applying oversampling to ensure at least 2 instances per class.")
            
            # Create a new DataFrame with oversampled rare classes
            oversampled_data = []
            
            # Process each row of the original data
            for i in range(len(X_train)):
                # Get the class label for this row
                if isinstance(y_train, pd.DataFrame):
                    class_label = y_train.iloc[i, 0]
                else:
                    class_label = y_train.iloc[i]
                
                # Add the original row
                row_data = X_train.iloc[i].tolist() + [class_label]
                oversampled_data.append(row_data)
                
                # If this is a rare class with only 1 instance, duplicate it
                if class_counts[class_label] == 1:
                    oversampled_data.append(row_data)  # Add it again
            
            # Create a new DataFrame with all columns including target
            all_columns = list(X_train.columns) + [target_name]
            oversampled_df = pd.DataFrame(oversampled_data, columns=all_columns)
            
            # Save the oversampled data
            oversampled_df.to_csv(temp_csv_path, index=False)
            print(f"Saved oversampled data with {len(oversampled_df)} rows to {temp_csv_path}")
            
            # Verify class counts after oversampling
            over_class_counts = oversampled_df[target_name].value_counts()
            print(f"Minimum class count after oversampling: {over_class_counts.min()}")
        
        ctabgan_model = CTABGAN(
            raw_csv_path=temp_csv_path,
            test_ratio=0.2,  # Keep consistent with the evaluation framework's test split
            categorical_columns=categorical_columns,
            integer_columns=integer_columns,
            problem_type=problem_type,
            epochs=epochs
        )
        
        # Train the model with try-except to handle potential issues
        try:
            ctabgan_model.fit()
        except Exception as e:
            print(f"Error during CTABGAN fitting: {e}")
            
            # Special handling for "least populated class" error
            if "least populated class" in str(e):
                print("Trying to fix the class imbalance issue...")
                
                # Create a more balanced dataset by oversampling rare classes even more
                if isinstance(y_train, pd.DataFrame):
                    # Get class distribution from the DataFrame
                    class_distribution = y_train.iloc[:, 0].value_counts()
                else:
                    # Get class distribution from the Series
                    class_distribution = y_train.value_counts()
                
                # Identify the minimum count needed for each class
                # Duplicate each class to have at least 3 instances
                balanced_data = []
                
                for i in range(len(X_train)):
                    # Get the class label for this row
                    if isinstance(y_train, pd.DataFrame):
                        class_label = y_train.iloc[i, 0]
                    else:
                        class_label = y_train.iloc[i]
                    
                    # Get the original data row
                    if isinstance(y_train, pd.DataFrame):
                        row_data = X_train.iloc[i].tolist() + [y_train.iloc[i, 0]]
                    else:
                        row_data = X_train.iloc[i].tolist() + [y_train.iloc[i]]
                    
                    # Always add the original row
                    balanced_data.append(row_data)
                    
                    # Add duplicates for rare classes
                    if class_distribution[class_label] < 3:
                        # Add enough duplicates to get to 3
                        for _ in range(3 - class_distribution[class_label]):
                            balanced_data.append(row_data)
                
                # Create a new balanced DataFrame
                balanced_df = pd.DataFrame(balanced_data, columns=list(X_train.columns) + [target_name])
                
                # Save balanced data to temporary CSV
                balanced_df.to_csv(temp_csv_path, index=False)
                print(f"Created a balanced dataset with {len(balanced_df)} rows")
                
                # Try recreating and fitting the model with the new balanced data
                try:
                    ctabgan_model = CTABGAN(
                        raw_csv_path=temp_csv_path,
                        test_ratio=0.2,
                        categorical_columns=categorical_columns,
                        integer_columns=integer_columns,
                        problem_type=problem_type,
                        epochs=epochs
                    )
                    ctabgan_model.fit()
                except Exception as e2:
                    print(f"Second attempt at fitting CTABGAN failed: {e2}")
                    # If it still fails, return None
                    return None
        
        return ctabgan_model
    except Exception as e:
        print(f"Error training CTABGAN model: {e}")
        return None


def train_rlig(X_train, y_train, episodes=2, epochs=5):
    """Train a RLiG model"""
    if not RLIG_AVAILABLE:
        return None
        
    try:
        # Initialize and train RLiG model
        rlig_model = RLiG()
        
        # Ensure the data is properly formatted
        if isinstance(y_train, pd.DataFrame):
            y_series = y_train.iloc[:, 0] if y_train.shape[1] == 1 else y_train
        else:
            y_series = y_train
        
        print(f"Training RLiG with {episodes} episodes and {epochs} epochs")
        rlig_model.fit(X_train, y_series, episodes=episodes, gan=1, k=0, epochs=epochs, n=1)
        return rlig_model
    except Exception as e:
        print(f"Error training RLiG model: {e}")
        return None


def train_great(X_train, y_train, batch_size=1, epochs=1):
    """Train a Generation of Realistic Tabular data
    with pretrained Transformer-based language models"""
    if not GREAT_AVAILABLE:
        return None

    try:
        # Initialize and train GReaT model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}. Using device: {device}")
        
        # Configure GReaT with appropriate parameters and suppress warnings
        great_model = GReaT(
            llm='distilgpt2', 
            batch_size=2, 
            epochs=10, 
            fp16=True,
            gradient_accumulation_steps=8,
            metric_for_best_model="accuracy"
        )
        
        # Set pad token explicitly to address the attention mask warning
        if hasattr(great_model, 'tokenizer') and great_model.tokenizer is not None:
            if great_model.tokenizer.pad_token is None:
                great_model.tokenizer.pad_token = great_model.tokenizer.eos_token
                print("Set pad_token to eos_token to fix attention mask warning")

        #otherwise for Mac, use this
        # great_model = GReaT(llm='unsloth/Llama-3.2-1B', batch_size=batch_size, epochs=epochs,
        #                     metric_for_best_model="accuracy",
        #                     # # For weak machine, add more 3 following lines
        #                     dataloader_num_workers=0,  # 0 means no parallelism in data loading
        #                     gradient_accumulation_steps=8,
        #                     efficient_finetuning="lora",
        #                     lora_target_modules=["q_proj", "v_proj"]
        #                     )
        # 
        # # Set pad token explicitly to address the attention mask warning
        # if hasattr(great_model, 'tokenizer') and great_model.tokenizer is not None:
        #     if great_model.tokenizer.pad_token is None:
        #         great_model.tokenizer.pad_token = great_model.tokenizer.eos_token
        #         print("Set pad_token to eos_token to fix attention mask warning")
        # Ensure the data is properly formatted
        if isinstance(y_train, pd.DataFrame):
            y_series = y_train.iloc[:, 0] if y_train.shape[1] == 1 else y_train
        else:
            y_series = y_train

        print(f"Training GReaT with {epochs} epochs")
        great_model.fit(X_train, y_series)
        return great_model
    except Exception as e:
        print(f"Error training GReaT model: {e}")
        return None


def train_tabsyn(X_train, y_train, epochs=50, random_seed=42):
    """Train the TabSyn tabular data synthesizer from Amazon Science
    
    TabSyn is a tabular data synthesis method that uses a GAN architecture
    with a pre-trained transformer encoder and a distribution-aware decoder.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.DataFrame
        Training targets
    epochs : int
        Number of training epochs
    random_seed : int
        Random seed for reproducibility
    """
    if not TABSYN_AVAILABLE:
        return None
        
    try:
        # Prepare data for TabSyn (we need to combine X and y)
        combined_data = pd.concat([X_train, y_train], axis=1)
        
        # Identify categorical columns
        categorical_cols = []
        for col in combined_data.columns:
            if len(np.unique(combined_data[col])) < 10:  # Heuristic for categorical columns
                categorical_cols.append(col)
        
        # Initialize TabularGAN with conservative settings and consistent random seed
        print(f"Training TabSyn with {epochs} epochs and random_seed={random_seed}")
        tabsyn_model = TabularGAN(
            train_data=combined_data,
            categorical_columns=categorical_cols,
            epochs=epochs,
            verbose=True,
            random_seed=random_seed
        )
        
        # Train the model
        tabsyn_model.fit()
        return tabsyn_model
    except Exception as e:
        print(f"Error training TabSyn model: {e}")
        return None


# ============= SYNTHETIC DATA SAVING HELPER =============

def save_synthetic_data(synthetic_data, model_name, dataset_name, directory="train_data"):
    """Save synthetic data according to dataset size classifications in the paper
    
    Parameters:
    -----------
    synthetic_data : pandas.DataFrame
        The synthetic data to save
    model_name : str
        The name of the model (e.g., 'ganblr', 'ctgan')
    dataset_name : str
        The name of the dataset
    directory : str, optional
        The directory to save the data to
    """
    os.makedirs(directory, exist_ok=True)
    file_path = f"{directory}/{model_name}_{dataset_name}_synthetic.csv"

    # Small: =< 15,000 rows
    # Medium: 20,000 to 50,000 rows
    # Large: >= 50,000 rows
    
    # For small and medium datasets, save all synthetic data
    # For large datasets, save a meaningful sample
    if len(synthetic_data) < 50000:
        # Small or medium dataset - save everything
        synthetic_data.to_csv(file_path, index=False)
        print(f"Saved complete {model_name.upper()} synthetic dataset ({len(synthetic_data)} samples) to {file_path}")
    else:
        # Large dataset - save a stratified sample
        # Try to find the target column for stratified sampling
        target_col = None
        for col in synthetic_data.columns:
            if col.lower() in ['target', 'label', 'class', 'y']:
                target_col = col
                break
                
        # For large synthetic datasets, save a stratified sample of 25,000
        if target_col and len(np.unique(synthetic_data[target_col])) > 1:
            from sklearn.model_selection import train_test_split
            
            # Use stratified sampling
            _, sampled_data = train_test_split(
                synthetic_data, 
                test_size=min(25000/len(synthetic_data), 0.5),  # Take at most 50% or 25,000
                stratify=synthetic_data[target_col],
                random_state=42
            )
        else:
            # Simple random sample if no target column or only one class
            sample_size = min(25000, len(synthetic_data))
            sampled_data = synthetic_data.sample(sample_size, random_state=42)
            
        sampled_data.to_csv(file_path, index=False)
        print(f"Saved representative sample of {model_name.upper()} synthetic data ({len(sampled_data)} of {len(synthetic_data)} samples) to {file_path}")
    
    return file_path


# ============= SYNTHETIC DATA GENERATION FUNCTIONS =============

def generate_bn_synthetic_data(bn_model, train_data, n_samples=None):
    """Generate synthetic data from a Bayesian Network model"""
    if not PGMPY_AVAILABLE or bn_model is None:
        return None
    
    if n_samples is None:
        n_samples = len(train_data)
    
    try:
        # Sample from the Bayesian Network
        sampler = BayesianModelSampling(bn_model)
        synthetic_data = sampler.forward_sample(size=n_samples)
        
        # Ensure synthetic data has the same column order as train_data
        col_order = list(train_data.columns)
        synthetic_data = synthetic_data[col_order]
        
        print(f"Generated {len(synthetic_data)} synthetic samples from Bayesian Network")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from BN: {e}")
        return None


def generate_nb_synthetic_data(nb_model, X_train, y_train, n_samples=None):
    """Generate synthetic data from a Naive Bayes model"""
    if nb_model is None:
        return None
    
    if n_samples is None:
        n_samples = len(X_train)
    
    try:
        # Get unique classes and their probabilities
        classes, class_counts = np.unique(y_train, return_counts=True)
        class_probs = class_counts / len(y_train)
        
        # Generate synthetic class labels
        synthetic_y = np.random.choice(classes, size=n_samples, p=class_probs)
        
        # Generate synthetic features for each class
        synthetic_X = np.zeros((n_samples, X_train.shape[1]))
        
        for i, c in enumerate(classes):
            # Get indices of samples with this class
            indices = synthetic_y == c
            n_class_samples = indices.sum()
            
            # For each feature, sample from Gaussian distribution with the class's mean and var
            for j in range(X_train.shape[1]):
                mean = nb_model.theta_[i, j]
                var = nb_model.var_[i, j]
                synthetic_X[indices, j] = np.random.normal(mean, np.sqrt(var), n_class_samples)
        
        # Create DataFrame with same column names
        if isinstance(X_train, pd.DataFrame):
            synthetic_X = pd.DataFrame(synthetic_X, columns=X_train.columns)
        
        # Combine features and target
        if isinstance(y_train, pd.DataFrame):
            synthetic_y = pd.DataFrame(synthetic_y, columns=y_train.columns)
        else:
            synthetic_y = pd.Series(synthetic_y, name=y_train.name)
        
        print(f"Generated {n_samples} synthetic samples from Naive Bayes")
        return pd.concat([synthetic_X, synthetic_y], axis=1)
    except Exception as e:
        print(f"Error generating synthetic data from Naive Bayes: {e}")
        return None


def generate_ctgan_synthetic_data(ctgan_model, train_data, n_samples=None):
    """Generate synthetic data from CTGAN model"""
    if not CTGAN_AVAILABLE or ctgan_model is None:
        return None
        
    if n_samples is None:
        n_samples = len(train_data)
    
    try:
        # For M1/M2 Macs, generate in smaller batches
        import os
        if hasattr(os, 'uname') and os.uname().machine == 'arm64' and n_samples > 500:
            print(f"Generating {n_samples} samples in smaller batches for Apple Silicon compatibility")
            batch_size = 500
            num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
            
            # Generate in batches and concatenate
            batches = []
            for i in range(num_batches):
                print(f"Generating batch {i+1}/{num_batches}")
                this_batch_size = min(batch_size, n_samples - i*batch_size)
                batch = ctgan_model.sample(this_batch_size)
                batches.append(batch)
            
            synthetic_data = pd.concat(batches, ignore_index=True)
        else:
            # Regular generation for other platforms
            synthetic_data = ctgan_model.sample(n_samples)
            
        print(f"Generated {len(synthetic_data)} synthetic samples from CTGAN")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from CTGAN: {e}")
        
        # Fallback: if sampling fails, try to sample a smaller number
        try:
            fallback_samples = min(n_samples, 500)
            print(f"Trying fallback with {fallback_samples} samples")
            synthetic_data = ctgan_model.sample(fallback_samples)
            print(f"Generated {len(synthetic_data)} synthetic samples as fallback")
            return synthetic_data
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return None

def generate_ctabgan_synthetic_data(ctabgan_model, train_data, n_samples=None):
    """Generate synthetic data from CTABGAN model
    
    Parameters:
    -----------
    ctabgan_model : CTABGAN
        Trained CTABGAN model
    train_data : pandas.DataFrame
        Training data used for context
    n_samples : int, optional
        Number of samples to generate. If None, uses train_data length
    """
    if not CTABGAN_AVAILABLE or ctabgan_model is None:
        return None
        
    if n_samples is None:
        n_samples = len(train_data)
    
    try:
        # CTABGAN's generate_samples() function generates data with the same size as the original dataset
        # We need to adjust the approach to generate the requested number of samples
        
        # Calculate how many times we need to call generate_samples
        batch_size = len(ctabgan_model.raw_df)
        num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        if num_batches > 1:
            print(f"Generating {n_samples} samples in {num_batches} batches")
            batches = []
            for i in range(num_batches):
                print(f"Generating batch {i+1}/{num_batches}")
                batch = ctabgan_model.generate_samples()
                # If it's the last batch and we need fewer samples
                if i == num_batches - 1 and n_samples % batch_size != 0:
                    batch = batch.head(n_samples % batch_size)
                batches.append(batch)
            synthetic_data = pd.concat(batches, ignore_index=True)
        else:
            # Generate samples and take only what we need
            synthetic_data = ctabgan_model.generate_samples()
            if n_samples < len(synthetic_data):
                synthetic_data = synthetic_data.head(n_samples)
        
        # Ensure all columns have the correct types
        # Convert any string columns to numeric if they should be numeric
        for col in synthetic_data.columns:
            # Skip the target column
            if col == 'target':
                continue
                
            # Try to convert to numeric if it's not already
            if synthetic_data[col].dtype == 'object':
                try:
                    synthetic_data[col] = pd.to_numeric(synthetic_data[col])
                except:
                    # If conversion fails, leave as is
                    pass
                    
        print(f"Generated {len(synthetic_data)} synthetic samples from CTABGAN")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from CTABGAN: {e}")
        
        # Fallback: if sampling fails, try a simple approach
        try:
            print("Trying fallback generation approach")
            synthetic_data = ctabgan_model.generate_samples()
            if len(synthetic_data) > n_samples:
                synthetic_data = synthetic_data.head(n_samples)
                
            # Same type conversion as above
            for col in synthetic_data.columns:
                if col == 'target':
                    continue
                if synthetic_data[col].dtype == 'object':
                    try:
                        synthetic_data[col] = pd.to_numeric(synthetic_data[col])
                    except:
                        pass
                        
            print(f"Generated {len(synthetic_data)} synthetic samples as fallback")
            return synthetic_data
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return None


def generate_great_synthetic_data(great_model, train_data, n_samples=None):
    """Generate synthetic data from GReaT model"""
    if not GREAT_AVAILABLE or great_model is None:
        return None

    if n_samples is None:
        n_samples = len(train_data)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}. Using device: {device}")

        # For M1/M2 Macs, generate in smaller batches
        import os
        if hasattr(os, 'uname') and os.uname().machine == 'arm64' and n_samples > 500:
            print(f"Generating {n_samples} samples in smaller batches for Apple Silicon compatibility")
            batch_size = 500
            num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

            # Generate in batches and concatenate
            batches = []
            for i in range(num_batches):
                print(f"Generating batch {i + 1}/{num_batches}")
                this_batch_size = min(batch_size, n_samples - i * batch_size)
                # Sample with standard parameters
                batch = great_model.sample(this_batch_size, device=device)
                batches.append(batch)

            synthetic_data = pd.concat(batches, ignore_index=True)
        else:
            # Regular generation for other platforms
            # Sample with standard parameters
            synthetic_data = great_model.sample(n_samples, device=device)

        print(f"Generated {len(synthetic_data)} synthetic samples from GReaT")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from GReaT: {e}")

        # Fallback: if sampling fails, try to sample a smaller number
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"CUDA available: {torch.cuda.is_available()}. Using device: {device}")

            fallback_samples = min(n_samples, 500)
            print(f"Trying fallback with {fallback_samples} samples")
            # Sample with standard parameters for fallback
            synthetic_data = great_model.sample(fallback_samples, device=device)
            print(f"Generated {len(synthetic_data)} synthetic samples as fallback")
            return synthetic_data
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return None


def generate_tabsyn_synthetic_data(tabsyn_model, train_data, n_samples=None):
    """Generate synthetic data from TabSyn model"""
    if not TABSYN_AVAILABLE or tabsyn_model is None:
        return None

    if n_samples is None:
        n_samples = len(train_data)

    try:
        # For M1/M2 Macs, generate in smaller batches for memory management
        import os
        if hasattr(os, 'uname') and os.uname().machine == 'arm64' and n_samples > 500:
            print(f"Generating {n_samples} samples in smaller batches for Apple Silicon compatibility")
            batch_size = 500
            num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
            
            # Generate in batches and concatenate
            batches = []
            for i in range(num_batches):
                print(f"Generating batch {i+1}/{num_batches}")
                this_batch_size = min(batch_size, n_samples - i*batch_size)
                batch = tabsyn_model.sample(this_batch_size)
                batches.append(batch)
                
            synthetic_data = pd.concat(batches, ignore_index=True)
        else:
            # Regular generation for other platforms
            synthetic_data = tabsyn_model.sample(n_samples)
            
        print(f"Generated {len(synthetic_data)} synthetic samples from TabSyn")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from TabSyn: {e}")
        
        # Fallback: if sampling fails, try to sample a smaller number
        try:
            fallback_samples = min(n_samples, 500)
            print(f"Trying fallback with {fallback_samples} samples")
            synthetic_data = tabsyn_model.sample(fallback_samples)
            print(f"Generated {len(synthetic_data)} synthetic samples as fallback")
            return synthetic_data
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return None


# ============= EVALUATION FUNCTIONS =============

def evaluate_models_on_fold(dataset_name, synthetic_data_cache, X_test, y_test, models):
    """Helper function to evaluate models on a specific fold
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset being evaluated
    synthetic_data_cache : dict
        Dictionary containing synthetic data for all models
    X_test : DataFrame
        Test features for this fold
    y_test : DataFrame
        Test target for this fold
    models : list
        List of model names to evaluate
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results for this fold
    """
    model_results = {
        'metrics': {},
        'times': {},
        'bic_scores': {},
        'dataset_name': dataset_name
    }
    
    # Get the models from the cache
    models_cache = synthetic_data_cache[dataset_name]['models']
    
    # Evaluate each model's synthetic data
    for model_name, model_cache in models_cache.items():
        if model_name not in models:
            continue
            
        print(f"\n-- Evaluating {model_name.upper()} synthetic data --")
        
        # Get synthetic data
        synthetic_data = model_cache['data']
        
        # Train classifiers on synthetic data and evaluate on real test data
        try:
            # Call evaluate_tstr with the full synthetic data
            metrics = evaluate_tstr(synthetic_data, X_test, y_test)
            
            # Store the metrics in our results structure
            for classifier_name, accuracy in metrics.items():
                metric_key = f"{classifier_name}_accuracy"
                if metric_key not in model_results['metrics']:
                    model_results['metrics'][metric_key] = {}
                model_results['metrics'][metric_key][model_name] = accuracy
            
            # Store training time directly with model name as key in the format MODEL-TIME
            model_upper = model_name.upper()
            if model_name.lower() == 'ganblr++':
                model_upper = 'GANBLR++'
                
            # Check for any time information in model_cache
            time_value = 0.0  # Default placeholder
            for possible_key in ['train_time', 'training_time', 'time']:
                if possible_key in model_cache:
                    time_value = model_cache[possible_key]
                    break
                    
            # Store with the model name as the key
            # Format the key properly to avoid CSV formatting issues
            time_key = f"{model_upper}-training_time"
            
            # Ensure times dictionary is initialized properly
            if 'training_time' not in model_results['times']:
                model_results['times']['training_time'] = {}
                
            # Store the time value in the dictionary
            model_results['times']['training_time'][model_name] = time_value
                
            # Store BIC score if available
            if 'bic' in model_cache:
                if 'bic' not in model_results['bic_scores']:
                    model_results['bic_scores']['bic'] = {}
                model_results['bic_scores']['bic'][model_name] = model_cache['bic']
        except Exception as e:
            print(f"Error evaluating {model_name} for {dataset_name}: {e}")
    
    return model_results

def average_fold_results(fold_results):
    """Average results across multiple folds
    
    Parameters:
    -----------
    fold_results : list
        List of dictionaries containing results for each fold
        
    Returns:
    --------
    dict
        Dictionary containing averaged results
    """
    if not fold_results:
        return {'metrics': {}, 'times': {}, 'bic_scores': {}}
    
    # Initialize results structure
    avg_results = {
        'metrics': {},
        'times': {},
        'bic_scores': {},
        'dataset_name': fold_results[0]['dataset_name']
    }
    
    # Accumulate results from all folds
    for fold_result in fold_results:
        # Metrics
        for metric_name, metric_data in fold_result['metrics'].items():
            if metric_name not in avg_results['metrics']:
                avg_results['metrics'][metric_name] = {}
                
            for model_name, metric_value in metric_data.items():
                # Skip None values
                if metric_value is None:
                    continue
                    
                if model_name not in avg_results['metrics'][metric_name]:
                    avg_results['metrics'][metric_name][model_name] = 0
                avg_results['metrics'][metric_name][model_name] += metric_value
        
        # Times
        for time_name, time_data in fold_result['times'].items():
            # Check if time_data is a dictionary or a direct value
            if isinstance(time_data, dict):
                # Handle dictionary case
                if time_name not in avg_results['times']:
                    avg_results['times'][time_name] = {}
                    
                for model_name, time_value in time_data.items():
                    # Skip None values
                    if time_value is None:
                        continue
                        
                    if model_name not in avg_results['times'][time_name]:
                        avg_results['times'][time_name][model_name] = 0
                    avg_results['times'][time_name][model_name] += time_value
            else:
                # Handle direct value case (float or other non-dict)
                if time_name not in avg_results['times']:
                    avg_results['times'][time_name] = 0
                # Skip None values
                if time_data is not None:
                    avg_results['times'][time_name] += time_data
        
        # BIC scores
        for bic_name, bic_data in fold_result['bic_scores'].items():
            # Check if bic_data is a dictionary or a direct value
            if isinstance(bic_data, dict):
                # Handle dictionary case
                if bic_name not in avg_results['bic_scores']:
                    avg_results['bic_scores'][bic_name] = {}
                    
                for model_name, bic_value in bic_data.items():
                    # Skip None values
                    if bic_value is None:
                        continue
                        
                    if model_name not in avg_results['bic_scores'][bic_name]:
                        avg_results['bic_scores'][bic_name][model_name] = 0
                    avg_results['bic_scores'][bic_name][model_name] += bic_value
            else:
                # Handle direct value case (float or other non-dict)
                if bic_name not in avg_results['bic_scores']:
                    avg_results['bic_scores'][bic_name] = 0
                # Skip None values
                if bic_data is not None:
                    avg_results['bic_scores'][bic_name] += bic_data
    
    # Average the accumulated results
    n_folds = len(fold_results)
    
    # Average metrics
    for metric_name in avg_results['metrics']:
        for model_name in avg_results['metrics'][metric_name]:
            avg_results['metrics'][metric_name][model_name] /= n_folds
    
    # Average times
    for time_name in avg_results['times']:
        if isinstance(avg_results['times'][time_name], dict):
            # Handle dictionary case
            for model_name in avg_results['times'][time_name]:
                avg_results['times'][time_name][model_name] /= n_folds
        else:
            # Handle direct value case
            avg_results['times'][time_name] /= n_folds
    
    # Average BIC scores
    for bic_name in avg_results['bic_scores']:
        if isinstance(avg_results['bic_scores'][bic_name], dict):
            # Handle dictionary case
            for model_name in avg_results['bic_scores'][bic_name]:
                avg_results['bic_scores'][bic_name][model_name] /= n_folds
        else:
            # Handle direct value case
            avg_results['bic_scores'][bic_name] /= n_folds
    
    return avg_results

def evaluate_tstr(synthetic_data, X_test, y_test, target_col='target'):
    """
    Evaluate models using TSTR (Train on Synthetic, Test on Real) methodology
    
    Parameters:
    -----------
    synthetic_data : DataFrame with features and target
    X_test : Real test features
    y_test : Real test target
    target_col : Name of the target column in synthetic_data
    
    Returns:
    --------
    Dictionary of model accuracies for LR, MLP, and RF classifiers
    and an average across all models
    """
    if synthetic_data is None:
        return {'LR': None, 'MLP': None, 'RF': None, 'XGB': None, 'AVG': None}
    
    try:
        # Split synthetic data into features and target
        if target_col in synthetic_data.columns:
            syn_X = synthetic_data.drop(target_col, axis=1)
            syn_y = synthetic_data[target_col]
        else:
            # If target column isn't found, assume last column is target
            syn_X = synthetic_data.iloc[:, :-1]
            syn_y = synthetic_data.iloc[:, -1]
            
        # Debug target variables
        print(f"Synthetic target type: {type(syn_y)}, values: {syn_y.head()}")
        print(f"Test target type: {type(y_test)}, values: {y_test.head()}")
        
        # Ensure targets are in a format classifiers can work with
        for y_var, name in [(syn_y, "synthetic"), (y_test, "test")]:
            if isinstance(y_var, pd.Series) or isinstance(y_var, pd.DataFrame):
                # Try to convert to numeric if possible
                try:
                    if name == "synthetic":
                        syn_y = pd.to_numeric(y_var)
                    else:
                        y_test = pd.to_numeric(y_var)
                except:
                    # If conversion fails, just use as is
                    pass
                    
                # If it's still a DataFrame with 1 column, convert to Series
                if isinstance(y_var, pd.DataFrame) and y_var.shape[1] == 1:
                    if name == "synthetic":
                        syn_y = y_var.iloc[:, 0]
                    else:
                        y_test = y_var.iloc[:, 0]
        
        # Ensure synthetic data has the same types as test data
        for col in X_test.columns:
            if col in syn_X.columns:
                # Convert synthetic columns to the same type as test columns
                try:
                    syn_X[col] = syn_X[col].astype(X_test[col].dtype)
                except:
                    # If conversion fails, try to convert both to float
                    try:
                        syn_X[col] = syn_X[col].astype(float)
                        if X_test[col].dtype != float:
                            X_test[col] = X_test[col].astype(float)
                    except:
                        # If that still fails, just continue (will use available columns only)
                        print(f"Warning: Could not convert column {col} to matching type")
            
        # Ensure column orders match exactly between synthetic and test data
        print(f"Synthetic X columns: {syn_X.columns.tolist()}")
        print(f"Test X columns: {X_test.columns.tolist()}")
        
        # Reorder synthetic columns to match test data if needed
        if list(syn_X.columns) != list(X_test.columns):
            print("Reordering synthetic data columns to match test data...")
            try:
                syn_X = syn_X[X_test.columns]
            except KeyError as e:
                print(f"Column mismatch between synthetic and test data: {e}")
                print("Using available columns only...")
                common_cols = list(set(syn_X.columns).intersection(set(X_test.columns)))
                syn_X = syn_X[common_cols]
                X_test = X_test[common_cols]
        
        # Define classification models as used in the paper
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'MLP': MLPClassifier(max_iter=500, early_stopping=True),
            'RF': RandomForestClassifier(n_estimators=100)
        }
        
        # Add XGBoost if available (with compatibility settings)
        if XGBOOST_AVAILABLE:
            try:
                # Try different parameter combinations based on XGBoost version
                try:
                    # Newer XGBoost versions
                    models['XGB'] = xgb.XGBClassifier(
                        n_estimators=100, 
                        learning_rate=0.1,
                        enable_categorical=False,  # Avoid categorical feature warning
                        use_label_encoder=False    # Compatibility for older versions
                    )
                except TypeError:
                    # Older XGBoost versions
                    models['XGB'] = xgb.XGBClassifier(
                        n_estimators=100, 
                        learning_rate=0.1
                    )
            except Exception as e:
                print(f"Could not initialize XGBoost classifier: {e}")
                # Don't add XGB to models in case of error
        
        results = {}
        
        # Fix data types for OneHotEncoder
        # Convert all columns to numeric if possible
        for col in syn_X.columns:
            try:
                syn_X[col] = pd.to_numeric(syn_X[col])
                X_test[col] = pd.to_numeric(X_test[col])
            except:
                # If can't convert to numeric, will be treated as categorical by encoder
                pass
                
        # Get feature categories for one-hot encoding (fixing numeric types)
        categories = []
        for col in X_test.columns:
            try:
                # Handling numeric types properly
                if pd.api.types.is_numeric_dtype(syn_X[col]) and pd.api.types.is_numeric_dtype(X_test[col]):
                    # For numeric columns, need to ensure all values are of the same type
                    unique_vals = np.unique(np.concatenate([
                        syn_X[col].astype(float).unique(), 
                        X_test[col].astype(float).unique()
                    ]))
                else:
                    # For categorical columns, handle as strings for consistency
                    unique_vals = np.unique(np.concatenate([
                        syn_X[col].astype(str).unique(), 
                        X_test[col].astype(str).unique()
                    ]))
                categories.append(unique_vals)
            except Exception as e:
                print(f"Error processing column {col} for OneHotEncoder: {e}")
                # Fallback: use union of unique values directly
                categories.append(np.union1d(syn_X[col].unique(), X_test[col].unique()))

        for name, model in models.items():
            try:
                print(f"Training {name} on synthetic data...")
                pipeline = Pipeline([
                    ('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore')),
                    ('model', model)
                ])
                
                # Train on synthetic data
                try:
                    # Try to fit with sklearn's error handling
                    pipeline.fit(syn_X, syn_y)
                    
                    # Test on real data
                    y_pred = pipeline.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    results[name] = acc
                    print(f"{name} TSTR accuracy: {acc:.4f}")
                except ValueError as ve:
                    print(f"Fitting error for {name}: {ve}")
                    # Try a backup approach if the error is about label types
                    if "Unknown label type" in str(ve):
                        try:
                            print(f"Attempting to convert target variables to int")
                            # Convert both target variables to integers
                            syn_y_int = syn_y.astype(int)
                            y_test_int = y_test.astype(int)
                            
                            # Try fitting with the converted targets
                            pipeline.fit(syn_X, syn_y_int)
                            y_pred = pipeline.predict(X_test)
                            acc = accuracy_score(y_test_int, y_pred)
                            results[name] = acc
                            print(f"{name} TSTR accuracy (with int conversion): {acc:.4f}")
                        except Exception as backup_error:
                            print(f"Backup approach failed: {backup_error}")
                            results[name] = None
                    else:
                        results[name] = None
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = None
        
        # Calculate average accuracy across all models (as done in the paper)
        valid_accs = [acc for acc in results.values() if acc is not None]
        if valid_accs:
            results['AVG'] = sum(valid_accs) / len(valid_accs)
            print(f"Average TSTR accuracy: {results['AVG']:.4f}")
        else:
            results['AVG'] = None
        
        return results
    except Exception as e:
        print(f"Error in TSTR evaluation: {e}")
        return {'LR': None, 'MLP': None, 'RF': None, 'XGB': None, 'AVG': None}


def get_gaussianNB_bic_score(model, data):
    """Calculate the BIC score for a GaussianNB model"""
    try:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Get predicted probabilities
        probs = model.predict_proba(X)
        
        # Compute log-likelihood
        log_likelihood = -log_loss(y, probs, labels=model.classes_, normalize=False)
        
        # Estimate number of parameters:
        # For GaussianNB:
        # - Each feature per class has a mean and variance => 2 * n_features
        # - Plus class priors (n_classes - 1 independent values)
        k = n_classes * 2 * n_features + (n_classes - 1)
        
        # Compute BIC
        bic = -2 * log_likelihood + k * np.log(n_samples)
        
        return bic
    except Exception as e:
        print(f"Error calculating BIC score: {e}")
        return None


# ============= MODEL EVALUATION IMPLEMENTATIONS =============

def train_and_evaluate_rlig(X_train, y_train, X_test, y_test, model_results, n_samples, episodes=2, epochs=5):
    """Train and evaluate RLiG model"""
    print("\n--------------------------------------------------")
    print("EVALUATING RLiG")
    print("--------------------------------------------------")
    start_time = time.time()
    try:
        # Initialize and train RLiG model
        rlig_model = train_rlig(X_train, y_train, episodes=episodes, epochs=epochs)
        rlig_time = time.time() - start_time
        
        if rlig_model is None:
            return
        
        # Note: RLiG's evaluate method already implements TSTR
        print("Evaluating RLiG model using built-in TSTR...")
        rlig_results = {}
        
        try:
            if isinstance(y_test, pd.DataFrame):
                y_test_series = y_test.iloc[:, 0] if y_test.shape[1] == 1 else y_test
            else:
                y_test_series = y_test
            
            # RLiG's built-in evaluate method implements TSTR
            lr_result = rlig_model.evaluate(X_test, y_test_series, model='lr')
            mlp_result = rlig_model.evaluate(X_test, y_test_series, model='mlp')
            rf_result = rlig_model.evaluate(X_test, y_test_series, model='rf')
            
            # We'll skip built-in XGBoost evaluation as it's not directly supported by RLiG
            xgb_result = None
            
            # Store individual results
            rlig_results = {
                'LR': lr_result,
                'MLP': mlp_result,
                'RF': rf_result,
                'AVG': (lr_result + mlp_result + rf_result) / 3
            }
            
            # Add XGBoost result if available
            if xgb_result is not None:
                rlig_results['XGB'] = xgb_result
                # Recalculate average with XGBoost
                valid_results = [lr_result, mlp_result, rf_result, xgb_result]
                rlig_results['AVG'] = sum(valid_results) / len(valid_results)
            
            for model_name, acc in rlig_results.items():
                model_results['metrics'][f'RLiG-{model_name}'] = acc
            
            # Ensure times dictionary is initialized properly
            if 'training_time' not in model_results['times']:
                model_results['times']['training_time'] = {}
                
            # Store the time value in the dictionary
            model_results['times']['training_time']['rlig'] = rlig_time
            model_results['bic_scores']['RLiG'] = rlig_model.best_score if hasattr(rlig_model, 'best_score') else None
            
            print(f"RLiG TSTR results: {rlig_results}")
            print(f"RLiG - Time: {rlig_time:.2f}s")
            
            # Save the network structure visualization and synthetic data sample
            dataset_name = model_results.get('dataset_name', 'unknown')
            try:
                # Create img directory if it doesn't exist
                os.makedirs("img", exist_ok=True)
                os.makedirs("train_data", exist_ok=True)
                
                # Save network visualization
                model_graphviz = rlig_model.bayesian_network.to_graphviz()
                model_graphviz.draw(f"img/rlig_{dataset_name}_network.png", prog="dot")
                print(f"RLiG network visualization saved to img/rlig_{dataset_name}_network.png")
                
                # Save synthetic data sample
                synthetic_data = rlig_model.sample(1000)
                # Convert to DataFrame if it's a numpy array
                if isinstance(synthetic_data, np.ndarray):
                    # Create DataFrame with original column names
                    columns = list(X_train.columns) + ['target']
                    synthetic_data = pd.DataFrame(synthetic_data, columns=columns)
                synthetic_data.to_csv(f"train_data/rlig_{dataset_name}_synthetic.csv", index=False)
                print(f"RLiG synthetic data sample saved to train_data/rlig_{dataset_name}_synthetic.csv")
            except Exception as e:
                print(f"Error saving RLiG outputs: {e}")
        except Exception as e:
            print(f"Error evaluating RLiG model: {e}")
    except Exception as e:
        print(f"Error with RLiG: {e}")
    
    # Clean up to prevent memory issues
    gc.collect()


def train_and_evaluate_ganblr(train_data, X_test, y_test, model_results, n_samples):
    """Train and evaluate GANBLR (Tree Search) model"""
    if not PGMPY_AVAILABLE:
        return
        
    print("\n--------------------------------------------------")
    print("RUNNING TREE SEARCH FOR GANBLR STRUCTURE")
    print("--------------------------------------------------")
    start_time = time.time()
    try:
        ts = TreeSearch(train_data)
        best_model_ts = ts.estimate()
        bn_ts = train_bn(best_model_ts, train_data)
        ts_time = time.time() - start_time
        
        if bn_ts is None:
            return
            
        # Store BIC score and time
        ts_bic = structure_score(bn_ts, train_data, scoring_method="bic-cg") if bn_ts else None
        # Ensure times dictionary is initialized properly
        if 'training_time' not in model_results['times']:
            model_results['times']['training_time'] = {}
            
        # Store the time value in the dictionary
        model_results['times']['training_time']['ganblr'] = ts_time
        model_results['bic_scores']['TS'] = ts_bic
        
        print(f"Tree Search - Time: {ts_time:.2f}s, BIC: {ts_bic}")
        
        # Evaluate GANBLR (BN with Tree Search)
        print("\nEvaluating GANBLR (BN with Tree Search) using TSTR...")
        try:
            # Generate synthetic data
            ts_synthetic = generate_bn_synthetic_data(bn_ts, train_data, n_samples=n_samples)
            
            # TSTR evaluation
            ts_tstr = evaluate_tstr(ts_synthetic, X_test, y_test)
            
            # Store results
            for model_name, acc in ts_tstr.items():
                model_results['metrics'][f'GANBLR-{model_name}'] = acc
            # Time is already stored using the training_time dictionary
            
            print(f"GANBLR - Time: {ts_time:.2f}s")
            
            # Save the network structure visualization and synthetic data sample
            dataset_name = model_results.get('dataset_name', 'unknown')
            try:
                # Create directories if they don't exist
                os.makedirs("img", exist_ok=True)
                os.makedirs("train_data", exist_ok=True)
                
                # Save network visualization
                bn_ts_viz = bn_ts.to_graphviz()
                bn_ts_viz.draw(f"img/ganblr_{dataset_name}_network.png", prog="dot")
                print(f"GANBLR network visualization saved to img/ganblr_{dataset_name}_network.png")
                
                # Save synthetic data using the helper function
                save_synthetic_data(ts_synthetic, "ganblr", dataset_name)
            except Exception as e:
                print(f"Error saving GANBLR outputs: {e}")
        except Exception as e:
            print(f"Error evaluating GANBLR model: {e}")
    except Exception as e:
        print(f"Error with Tree Search: {e}")


def train_and_evaluate_ganblrpp(train_data, X_test, y_test, model_results, n_samples):
    """Train and evaluate GANBLR++ (Hill Climbing) model"""
    if not PGMPY_AVAILABLE:
        return
        
    print("\n--------------------------------------------------")
    print("RUNNING HILL CLIMBING FOR GANBLR++ STRUCTURE")
    print("--------------------------------------------------")
    start_time = time.time()
    try:
        hc = HillClimbSearch(train_data)
        best_model_hc = hc.estimate(scoring_method=BIC(train_data))
        bn_hc = train_bn(best_model_hc, train_data)
        hc_time = time.time() - start_time
        
        if bn_hc is None:
            return
            
        # Store BIC score and time
        hc_bic = structure_score(bn_hc, train_data, scoring_method="bic-cg") if bn_hc else None
        # Ensure times dictionary is initialized properly
        if 'training_time' not in model_results['times']:
            model_results['times']['training_time'] = {}
            
        # Store the time value in the dictionary
        model_results['times']['training_time']['ganblr++'] = hc_time
        model_results['bic_scores']['HC'] = hc_bic
        
        print(f"Hill Climbing - Time: {hc_time:.2f}s, BIC: {hc_bic}")
        
        # Evaluate GANBLR++ (BN with Hill Climbing)
        print("\nEvaluating GANBLR++ (BN with Hill Climbing) using TSTR...")
        try:
            # Generate synthetic data
            hc_synthetic = generate_bn_synthetic_data(bn_hc, train_data, n_samples=n_samples)
            
            # TSTR evaluation
            hc_tstr = evaluate_tstr(hc_synthetic, X_test, y_test)
            
            # Store results
            for model_name, acc in hc_tstr.items():
                model_results['metrics'][f'GANBLR++-{model_name}'] = acc
            # Time is already stored using the training_time dictionary
            
            print(f"GANBLR++ - Time: {hc_time:.2f}s")
            
            # Save the network structure visualization and synthetic data sample
            dataset_name = model_results.get('dataset_name', 'unknown')
            try:
                # Create directories if they don't exist
                os.makedirs("img", exist_ok=True)
                os.makedirs("train_data", exist_ok=True)
                
                # Save network visualization
                bn_hc_viz = bn_hc.to_graphviz()
                bn_hc_viz.draw(f"img/ganblrpp_{dataset_name}_network.png", prog="dot")
                print(f"GANBLR++ network visualization saved to img/ganblrpp_{dataset_name}_network.png")
                
                # Save synthetic data using the helper function
                save_synthetic_data(hc_synthetic, "ganblrpp", dataset_name)
            except Exception as e:
                print(f"Error saving GANBLR++ outputs: {e}")
        except Exception as e:
            print(f"Error evaluating GANBLR++ model: {e}")
    except Exception as e:
        print(f"Error with Hill Climbing: {e}")


def train_and_evaluate_ctgan(X_train, y_train, X_test, y_test, model_results, n_samples, epochs=50):
    """Train and evaluate CTGAN model"""
    if not CTGAN_AVAILABLE:
        return
        
    print("\n--------------------------------------------------")
    print("EVALUATING CTGAN")
    print("--------------------------------------------------")
    start_time = time.time()
    try:
        # Prepare data for CTGAN (combine features and target)
        ctgan_train_data = pd.concat([X_train, y_train], axis=1)
        
        # Identify categorical columns
        discrete_columns = []
        for col in ctgan_train_data.columns:
            if len(np.unique(ctgan_train_data[col])) < 10:  # Heuristic for categorical
                discrete_columns.append(col)
        
        # Train CTGAN
        ctgan_model = train_ctgan(
            ctgan_train_data, 
            discrete_columns=discrete_columns,
            epochs=epochs,  # Use parameter from arguments
            batch_size=min(500, len(ctgan_train_data))  # Adjust batch size for small datasets
        )
        ctgan_time = time.time() - start_time
        
        if ctgan_model is None:
            return
            
        # Generate synthetic data
        ctgan_synthetic = generate_ctgan_synthetic_data(ctgan_model, ctgan_train_data, n_samples=n_samples)
        
        # TSTR evaluation
        print("Performing TSTR evaluation for CTGAN...")
        ctgan_tstr = evaluate_tstr(ctgan_synthetic, X_test, y_test)
        
        # Store results
        for model_name, acc in ctgan_tstr.items():
            model_results['metrics'][f'CTGAN-{model_name}'] = acc
        # Ensure times dictionary is initialized properly
        if 'training_time' not in model_results['times']:
            model_results['times']['training_time'] = {}
            
        # Store the time value in the dictionary
        model_results['times']['training_time']['ctgan'] = ctgan_time
        
        print(f"CTGAN - Time: {ctgan_time:.2f}s")
        
        # Save synthetic data sample
        dataset_name = model_results.get('dataset_name', 'unknown')
        try:
            # Create directory if it doesn't exist
            os.makedirs("train_data", exist_ok=True)
            
            # Save synthetic data using the helper function
            save_synthetic_data(ctgan_synthetic, "ctgan", dataset_name)
        except Exception as e:
            print(f"Error saving CTGAN synthetic data: {e}")
    except Exception as e:
        print(f"Error evaluating CTGAN model: {e}")


def train_and_evaluate_ctabgan(X_train, y_train, X_test, y_test, model_results, n_samples, epochs=50):
    """Train and evaluate CTABGAN model"""
    if not CTABGAN_AVAILABLE:
        return
        
    print("\n--------------------------------------------------")
    print("EVALUATING CTABGAN")
    print("--------------------------------------------------")
    start_time = time.time()
    try:
        # Identify categorical columns
        categorical_columns = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or len(np.unique(X_train[col])) < 10:
                categorical_columns.append(col)
        
        # Train CTABGAN
        ctabgan_model = train_ctabgan(
            X_train, 
            y_train,
            categorical_columns=categorical_columns,
            epochs=epochs
        )
        ctabgan_time = time.time() - start_time
        
        if ctabgan_model is None:
            return
            
        # Prepare train data for synthetic data generation
        train_data = pd.concat([X_train, y_train], axis=1)
        
        # Generate synthetic data
        ctabgan_synthetic = generate_ctabgan_synthetic_data(ctabgan_model, train_data, n_samples=n_samples)
        
        # TSTR evaluation
        print("Performing TSTR evaluation for CTABGAN...")
        ctabgan_tstr = evaluate_tstr(ctabgan_synthetic, X_test, y_test)
        
        # Store results
        for model_name, acc in ctabgan_tstr.items():
            model_results['metrics'][f'CTABGAN-{model_name}'] = acc
        # Ensure times dictionary is initialized properly
        if 'training_time' not in model_results['times']:
            model_results['times']['training_time'] = {}
            
        # Store the time value in the dictionary
        model_results['times']['training_time']['ctabgan'] = ctabgan_time
        
        print(f"CTABGAN - Time: {ctabgan_time:.2f}s")
        
        # Save synthetic data sample
        dataset_name = model_results.get('dataset_name', 'unknown')
        try:
            # Create directory if it doesn't exist
            os.makedirs("train_data", exist_ok=True)
            
            # Save synthetic data using the helper function
            save_synthetic_data(ctabgan_synthetic, "ctabgan", dataset_name)
        except Exception as e:
            print(f"Error saving CTABGAN synthetic data: {e}")
    except Exception as e:
        print(f"Error evaluating CTABGAN model: {e}")


def train_and_evaluate_nb(X_train, y_train, X_test, y_test, train_data, model_results, n_samples):
    """Train and evaluate Naive Bayes model"""
    print("\n--------------------------------------------------")
    print("EVALUATING NAIVE BAYES")
    print("--------------------------------------------------")
    start_time = time.time()
    try:
        nb = train_naive_bayes(X_train, y_train)
        nb_time = time.time() - start_time
        
        if nb is None:
            return
            
        # Generate synthetic data
        nb_synthetic = generate_nb_synthetic_data(nb, X_train, y_train, n_samples=n_samples)
        
        # TSTR evaluation
        print("Performing TSTR evaluation for Naive Bayes...")
        nb_tstr = evaluate_tstr(nb_synthetic, X_test, y_test)
        
        # Store results
        for model_name, acc in nb_tstr.items():
            model_results['metrics'][f'NB-{model_name}'] = acc
        # Ensure times dictionary is initialized properly
        if 'training_time' not in model_results['times']:
            model_results['times']['training_time'] = {}
            
        # Store the time value in the dictionary
        model_results['times']['training_time']['nb'] = nb_time
        model_results['bic_scores']['NB'] = get_gaussianNB_bic_score(nb, train_data) if nb else None
        
        print(f"Naive Bayes - Time: {nb_time:.2f}s")
        
        # Save synthetic data sample
        dataset_name = model_results.get('dataset_name', 'unknown')
        try:
            # Create directory if it doesn't exist
            os.makedirs("train_data", exist_ok=True)
            
            # Save synthetic data using the helper function
            save_synthetic_data(nb_synthetic, "nb", dataset_name)
        except Exception as e:
            print(f"Error saving Naive Bayes synthetic data: {e}")
    except Exception as e:
        print(f"Error with Naive Bayes: {e}")


# ============= MAIN COMPARISON FUNCTION =============

def compare_models_tstr(datasets, models=None, n_rounds=3, seed=42, rlig_episodes=2, rlig_epochs=5, 
                  ctgan_epochs=50, great_bs=1, great_epochs=5, tabsyn_epochs=50, verbose=False, discretize=True,
                  use_cv=False, n_folds=2, nested_cv=False):
    """
    Compare generative models using TSTR methodology as described in the paper
    with multiple rounds of cross-validation for robustness
    
    Parameters:
    -----------
    datasets : dict
        Dictionary mapping dataset names to dataset sources
    models : list or None
        List of models to evaluate. If None, evaluate all available models.
        Options: 'rlig', 'ganblr', 'ganblr++', 'ctgan', 'nb', 'great', 'tabsyn'
    n_rounds : int
        Number of rounds of cross-validation to run (default: 3)
    seed : int
        Random seed for reproducibility
    rlig_episodes : int
        Number of episodes for RLiG training
    rlig_epochs : int
        Number of epochs for RLiG training
    ctgan_epochs : int
        Number of epochs for CTGAN training
    great_epochs : int
        Number of epochs for GReaT training
    tabsyn_epochs : int
        Number of epochs for TabSyn training
    verbose : bool
        Whether to print verbose output
    discretize : bool
        Whether to apply discretization to continuous features during preprocessing.
        When True, quantile binning with 7 bins is used.
        When False, only standardization is applied to continuous features.
    use_cv : bool
        Whether to use k-fold cross-validation (default: False)
    n_folds : int
        Number of folds to use when use_cv is True (default: 2)
    nested_cv : bool
        Whether to use k-fold cross-validation within each random seed round.
        When True, runs n_folds CV for each of the n_rounds random seeds.
        When False and use_cv is True, uses only CV without different random seeds.
    """
    # Default models to evaluate
    if models is None:
        models = ['rlig', 'ganblr', 'ganblr++', 'ctgan', 'ctabgan', 'nb', 'great', 'tabsyn']
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set seeds for other libraries if they are used
    try:
        import tensorflow as tf
        import random
        random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"Random seeds set to {seed} for numpy, torch, tensorflow, and Python")
    except ImportError:
        print(f"Random seeds set to {seed} for numpy and torch")
    
    print(f"Running {n_rounds} rounds of cross-validation for robust results...")
    print(f"Models to evaluate: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    
    if verbose:
        print(f"Training parameters:")
        print(f"  - RLiG episodes: {rlig_episodes}")
        print(f"  - RLiG epochs: {rlig_epochs}")
        print(f"  - CTGAN epochs: {ctgan_epochs}")
        print(f"  - GReaT epochs: {great_epochs}")
        print(f"  - TabSyn epochs: {tabsyn_epochs}")
    
    # Dictionary to store results from all rounds
    all_rounds_results = {}
    
    # Dictionary to store synthetic data for each model and dataset
    synthetic_data_cache = {}
    
    # First, generate synthetic data for all models once
    print("\n\n== GENERATING SYNTHETIC DATA FOR ALL MODELS ==\n")
    
    # Process each dataset
    for name, dataset_info in datasets.items():
        print(f"\n{'='*50}\nProcessing dataset: {name}\n{'='*50}")
        X, y = load_dataset(name, dataset_info)
        if X is None or y is None:
            continue
            
        # Preprocess data based on discretization flag
        try:
            if use_cv:
                # Initialize synthetic data cache for this dataset with folds
                synthetic_data_cache[name] = {
                    'folds': [],
                    'models': {}
                }
                
                # Create a fold for each cross-validation split
                for fold_idx in range(n_folds):
                    X_train, X_test, y_train, y_test = preprocess_data(
                        X, y, discretize=discretize, cv_fold=fold_idx, n_folds=n_folds
                    )
                    train_data = pd.concat([X_train, y_train], axis=1)
                    print(f"Fold {fold_idx+1}/{n_folds} data loaded and preprocessed. "
                          f"Training data shape: {train_data.shape}")
                    
                    # Add this fold to the dataset cache
                    synthetic_data_cache[name]['folds'].append({
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train, 
                        'y_test': y_test,
                        'train_data': train_data
                    })
            else:
                # Traditional train-test split
                X_train, X_test, y_train, y_test = preprocess_data(X, y, discretize=discretize)
                train_data = pd.concat([X_train, y_train], axis=1)
                print(f"Data loaded and preprocessed with discretize={discretize}. "
                      f"Training data shape: {train_data.shape}")
                
                # Initialize synthetic data cache for this dataset
                synthetic_data_cache[name] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train, 
                    'y_test': y_test,
                    'train_data': train_data,
                    'models': {}
                }
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            continue
        
        # Set number of synthetic samples to match training data size
        n_samples = len(train_data)
            
        # Generate synthetic data for each model
        if 'ganblr++' in models:
            print("\n-- Generating synthetic data for GANBLR++ --")
            try:
                hc = HillClimbSearch(train_data)
                best_model_hc = hc.estimate(scoring_method=BIC(train_data))
                bn_hc = train_bn(best_model_hc, train_data)
                
                if bn_hc:
                    # Store BIC score
                    hc_bic = structure_score(bn_hc, train_data, scoring_method="bic-cg")
                    
                    # Generate synthetic data
                    hc_synthetic = generate_bn_synthetic_data(bn_hc, train_data, n_samples=n_samples)
                    
                    if hc_synthetic is not None:
                        # Save the network visualization
                        try:
                            os.makedirs("img", exist_ok=True)
                            bn_hc.to_graphviz().draw(f"img/ganblrpp_{name}_network.png", prog="dot")
                            print(f"GANBLR++ network visualization saved to img/ganblrpp_{name}_network.png")
                        except Exception as e:
                            print(f"Error saving network visualization: {e}")
                            
                        # Store in cache
                        synthetic_data_cache[name]['models']['ganblr++'] = {
                            'data': hc_synthetic,
                            'bic': hc_bic
                        }
                        
                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(hc_synthetic, "ganblrpp", name)
            except Exception as e:
                print(f"Error generating GANBLR++ synthetic data: {e}")
                
        if 'ganblr' in models:
            print("\n-- Generating synthetic data for GANBLR --")
            try:
                ts = TreeSearch(train_data)
                best_model_ts = ts.estimate()
                bn_ts = train_bn(best_model_ts, train_data)
                
                if bn_ts:
                    # Store BIC score
                    ts_bic = structure_score(bn_ts, train_data, scoring_method="bic-cg")
                    
                    # Generate synthetic data
                    ts_synthetic = generate_bn_synthetic_data(bn_ts, train_data, n_samples=n_samples)
                    
                    if ts_synthetic is not None:
                        # Save the network visualization
                        try:
                            os.makedirs("img", exist_ok=True)
                            bn_ts.to_graphviz().draw(f"img/ganblr_{name}_network.png", prog="dot")
                            print(f"GANBLR network visualization saved to img/ganblr_{name}_network.png")
                        except Exception as e:
                            print(f"Error saving network visualization: {e}")
                            
                        # Store in cache
                        synthetic_data_cache[name]['models']['ganblr'] = {
                            'data': ts_synthetic,
                            'bic': ts_bic
                        }
                        
                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(ts_synthetic, "ganblr", name)
            except Exception as e:
                print(f"Error generating GANBLR synthetic data: {e}")
                
        if 'ctgan' in models:
            print("\n-- Generating synthetic data for CTGAN --")
            try:
                # Prepare data for CTGAN
                ctgan_train_data = pd.concat([X_train, y_train], axis=1)
                
                # Identify categorical columns
                discrete_columns = []
                for col in ctgan_train_data.columns:
                    if len(np.unique(ctgan_train_data[col])) < 10:  # Heuristic for categorical
                        discrete_columns.append(col)
                
                # Train CTGAN
                ctgan_model = train_ctgan(
                    ctgan_train_data, 
                    discrete_columns=discrete_columns,
                    epochs=ctgan_epochs,
                    batch_size=min(500, len(ctgan_train_data))
                )
                
                if ctgan_model:
                    # Generate synthetic data
                    ctgan_synthetic = generate_ctgan_synthetic_data(ctgan_model, ctgan_train_data, n_samples=n_samples)
                    
                    if ctgan_synthetic is not None:
                        # Store in cache
                        synthetic_data_cache[name]['models']['ctgan'] = {
                            'data': ctgan_synthetic
                        }
                        
                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(ctgan_synthetic, "ctgan", name)
            except Exception as e:
                print(f"Error generating CTGAN synthetic data: {e}")
                
        if 'ctabgan' in models:
            print("\n-- Generating synthetic data for CTABGAN --")
            try:
                # Identify categorical columns
                categorical_columns = []
                for col in X_train.columns:
                    if X_train[col].dtype == 'object' or len(np.unique(X_train[col])) < 10:
                        categorical_columns.append(col)
                
                # Train CTABGAN
                ctabgan_model = train_ctabgan(
                    X_train, 
                    y_train,
                    categorical_columns=categorical_columns,
                    epochs=ctgan_epochs  # Use the same parameter as CTGAN for consistency
                )
                
                if ctabgan_model:
                    # Prepare train data for synthetic data generation
                    train_data_combined = pd.concat([X_train, y_train], axis=1)
                    
                    # Generate synthetic data
                    ctabgan_synthetic = generate_ctabgan_synthetic_data(ctabgan_model, train_data_combined, n_samples=n_samples)
                    
                    if ctabgan_synthetic is not None:
                        # Store in cache
                        synthetic_data_cache[name]['models']['ctabgan'] = {
                            'data': ctabgan_synthetic
                        }
                        
                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(ctabgan_synthetic, "ctabgan", name)
            except Exception as e:
                print(f"Error generating CTABGAN synthetic data: {e}")
                
        if 'nb' in models:
            print("\n-- Generating synthetic data for Naive Bayes --")
            try:
                # Train NB model
                nb = train_naive_bayes(X_train, y_train)
                
                if nb:
                    # Calculate BIC score
                    nb_bic = get_gaussianNB_bic_score(nb, train_data)
                    
                    # Generate synthetic data
                    nb_synthetic = generate_nb_synthetic_data(nb, X_train, y_train, n_samples=n_samples)
                    
                    if nb_synthetic is not None:
                        # Store in cache
                        synthetic_data_cache[name]['models']['nb'] = {
                            'data': nb_synthetic,
                            'bic': nb_bic
                        }
                        
                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(nb_synthetic, "nb", name)
            except Exception as e:
                print(f"Error generating Naive Bayes synthetic data: {e}")
                
        if 'rlig' in models and RLIG_AVAILABLE:
            print("\n-- Generating synthetic data for RLiG --")
            try:
                # Train RLiG model
                rlig_model = train_rlig(X_train, y_train, episodes=rlig_episodes, epochs=rlig_epochs)
                
                if rlig_model:
                    # Store BIC score if available
                    rlig_bic = rlig_model.best_score if hasattr(rlig_model, 'best_score') else None
                    
                    # Generate synthetic data
                    synthetic_data = rlig_model.sample(1000)
                    
                    # Convert to DataFrame if it's a numpy array
                    if isinstance(synthetic_data, np.ndarray):
                        columns = list(X_train.columns) + ['target']
                        synthetic_data = pd.DataFrame(synthetic_data, columns=columns)
                    
                    if synthetic_data is not None:
                        # Save the network visualization
                        try:
                            os.makedirs("img", exist_ok=True)
                            rlig_model.bayesian_network.to_graphviz().draw(f"img/rlig_{name}_network.png", prog="dot")
                            print(f"RLiG network visualization saved to img/rlig_{name}_network.png")
                        except Exception as e:
                            print(f"Error saving network visualization: {e}")
                            
                        # Store in cache
                        synthetic_data_cache[name]['models']['rlig'] = {
                            'data': synthetic_data,
                            'bic': rlig_bic,
                            'model': rlig_model  # Store model for built-in evaluation
                        }
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        synthetic_data.to_csv(f"train_data/rlig_{name}_synthetic.csv", index=False)
                        print(f"RLiG synthetic data saved to train_data/rlig_{name}_synthetic.csv")
            except Exception as e:
                print(f"Error generating RLiG synthetic data: {e}")

        if 'great' in models and GREAT_AVAILABLE:
            print("\n-- Generating synthetic data for GReaT --")
            try:
                # Get data and preprocess based on discretization flag for GReaT
                X_train_great, X_test_great, y_train_great, y_test_great = preprocess_data(X, y, discretize=discretize, model_name='great')
                
                # Prepare data for GReaT
                great_train_data = pd.concat([X_train_great, y_train_great], axis=1)

                # Identify categorical columns
                discrete_columns = []
                for col in great_train_data.columns:
                    if len(np.unique(great_train_data[col])) < 10:  # Heuristic for categorical
                        discrete_columns.append(col)

                # Train GReaT model
                great_model = train_great(X_train_great, y_train_great, batch_size=great_bs, epochs=great_epochs)

                if great_model:
                    # Generate synthetic data
                    great_synthetic = generate_great_synthetic_data(great_model, great_train_data, n_samples=n_samples)

                    if great_synthetic is not None:
                        # Store in cache
                        synthetic_data_cache[name]['models']['great'] = {
                            'data': great_synthetic,
                            'X_test': X_test_great,
                            'y_test': y_test_great
                        }

                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(great_synthetic, "great", name)
            except Exception as e:
                print(f"Error generating GReaT synthetic data: {e}")
                
        if 'tabsyn' in models and TABSYN_AVAILABLE:
            print("\n-- Generating synthetic data for TabSyn --")
            try:
                # Get data and preprocess based on discretization flag for TabSyn
                X_train_tabsyn, X_test_tabsyn, y_train_tabsyn, y_test_tabsyn = preprocess_data(X, y, discretize=discretize, model_name='tabsyn')
                
                # Train TabSyn model with consistent random seed
                tabsyn_model = train_tabsyn(X_train_tabsyn, y_train_tabsyn, epochs=tabsyn_epochs, random_seed=seed)
                
                if tabsyn_model:
                    # Generate synthetic data
                    tabsyn_synthetic = generate_tabsyn_synthetic_data(tabsyn_model, train_data, n_samples=n_samples)
                    
                    if tabsyn_synthetic is not None:
                        # Store in cache
                        synthetic_data_cache[name]['models']['tabsyn'] = {
                            'data': tabsyn_synthetic,
                            'X_test': X_test_tabsyn,
                            'y_test': y_test_tabsyn
                        }
                        
                        # Save synthetic data using the helper function
                        os.makedirs("train_data", exist_ok=True)
                        save_synthetic_data(tabsyn_synthetic, "tabsyn", name)
            except Exception as e:
                print(f"Error generating TabSyn synthetic data: {e}")

    # Configure approach based on CV and nested CV options
    if nested_cv:
        print(f"\n\n{'='*20} USING {n_folds}-FOLD CROSS-VALIDATION WITHIN {n_rounds} RANDOM SEED ROUNDS {'='*20}\n")
    elif use_cv:
        print(f"\n\n{'='*20} USING {n_folds}-FOLD CROSS-VALIDATION {'='*20}\n")
    else:
        print(f"\n\n{'='*20} USING {n_rounds} ROUNDS WITH DIFFERENT RANDOM SEEDS {'='*20}\n")

    # Dictionary to store results from all rounds/folds
    all_rounds_results = {}
    
    # Run appropriate evaluation strategy
    if nested_cv:
        # Nested CV: Random seeds with k-fold CV within each
        for round_idx in range(n_rounds):
            all_rounds_results[round_idx] = {}
            
            print(f"\n\n{'='*20} RANDOM SEED ROUND {round_idx+1}/{n_rounds} {'='*20}\n")
            
            # Set a different seed for each round but in a deterministic way
            round_seed = seed + round_idx
            np.random.seed(round_seed)
            
            # For each dataset in this round
            for name in synthetic_data_cache.keys():
                X, y = load_dataset(name, datasets[name])
                if X is None or y is None:
                    continue
                
                # Initialize results for this dataset in this round
                all_rounds_results[round_idx][name] = {
                    'metrics': {},
                    'times': {},
                    'bic_scores': {},
                    'dataset_name': name
                }
                
                # Run k-fold CV for this dataset in this round
                fold_results = []
                
                for fold_idx in range(n_folds):
                    print(f"\n{'='*15} FOLD {fold_idx+1}/{n_folds} FOR {name} IN ROUND {round_idx+1} {'='*15}\n")
                    
                    # Prepare data for this fold
                    X_train, X_test, y_train, y_test = preprocess_data(
                        X, y, discretize=discretize, cv_fold=fold_idx, n_folds=n_folds
                    )
                    
                    # Evaluate each model using the test data
                    fold_model_results = evaluate_models_on_fold(
                        name, synthetic_data_cache, X_test, y_test, models
                    )
                    
                    # Store this fold's results
                    fold_results.append(fold_model_results)
                
                # Average the results across folds for this dataset in this round
                all_rounds_results[round_idx][name] = average_fold_results(fold_results)
                
    elif use_cv:
        # Single round of k-fold CV
        round_idx = 0
        all_rounds_results[round_idx] = {}
        
        # For each dataset
        for name in synthetic_data_cache.keys():
            X, y = load_dataset(name, datasets[name])
            if X is None or y is None:
                continue
            
            # Initialize results for this dataset
            all_rounds_results[round_idx][name] = {
                'metrics': {},
                'times': {},
                'bic_scores': {},
                'dataset_name': name
            }
            
            # Run k-fold CV for this dataset
            fold_results = []
            
            for fold_idx in range(n_folds):
                print(f"\n{'='*15} FOLD {fold_idx+1}/{n_folds} FOR {name} {'='*15}\n")
                
                # Prepare data for this fold
                X_train, X_test, y_train, y_test = preprocess_data(
                    X, y, discretize=discretize, cv_fold=fold_idx, n_folds=n_folds
                )
                
                # Evaluate each model using the test data
                fold_model_results = evaluate_models_on_fold(
                    name, synthetic_data_cache, X_test, y_test, models
                )
                
                # Store this fold's results
                fold_results.append(fold_model_results)
            
            # Average the results across folds for this dataset
            all_rounds_results[round_idx][name] = average_fold_results(fold_results)
            
    else:
        # Traditional approach: Multiple rounds with different random seeds
        for round_idx in range(n_rounds):
            all_rounds_results[round_idx] = {}
            
            print(f"\n\n{'='*20} RANDOM SEED ROUND {round_idx+1}/{n_rounds} {'='*20}\n")
            
            # Set a different seed for each round but in a deterministic way
            round_seed = seed + round_idx
            np.random.seed(round_seed)
            
            # For each dataset in this round
            for name in synthetic_data_cache.keys():
                print(f"\n{'='*50}\nEvaluating dataset: {name} (Round {round_idx+1})\n{'='*50}")
                
                # Get cached data
                cached_data = synthetic_data_cache[name]
                X_test = cached_data['X_test'] 
                y_test = cached_data['y_test']
                
                # Evaluate models
                model_results = {
                    'metrics': {},
                    'times': {},
                    'bic_scores': {},
                    'dataset_name': name
                }
                
                # Evaluate each model's synthetic data
                models_cache = cached_data['models']
                
                for model_name, model_cache in models_cache.items():
                    if model_name not in models:
                        continue
                        
                    print(f"\n-- Evaluating {model_name.upper()} synthetic data --")
                    
                    # Get synthetic data
                    synthetic_data = model_cache['data']
                    
                    # Train classifiers on synthetic data and evaluate on real test data
                    try:
                        # Call evaluate_tstr with the full synthetic data
                        metrics = evaluate_tstr(synthetic_data, X_test, y_test)
                        
                        # Store the metrics in our results structure
                        for classifier_name, accuracy in metrics.items():
                            metric_key = f"{classifier_name}_accuracy"
                            if metric_key not in model_results['metrics']:
                                model_results['metrics'][metric_key] = {}
                            model_results['metrics'][metric_key][model_name] = accuracy
                        
                        # Store training time directly with model name as key in the format MODEL-TIME
                        model_upper = model_name.upper()
                        if model_name.lower() == 'ganblr++':
                            model_upper = 'GANBLR++'
                            
                        # Check for any time information in model_cache
                        time_value = 0.0  # Default placeholder
                        for possible_key in ['train_time', 'training_time', 'time']:
                            if possible_key in model_cache:
                                time_value = model_cache[possible_key]
                                break
                                
                        # Store with the model name as the key
                        # Format the key properly to avoid CSV formatting issues
                        time_key = "training_time"
                        
                        # Ensure times dictionary is initialized properly
                        if time_key not in model_results['times']:
                            model_results['times'][time_key] = {}
                            
                        # Store the time value in the dictionary
                        model_results['times'][time_key][model_name] = time_value
                            
                        # Store BIC score if available
                        if 'bic' in model_cache:
                            if 'bic' not in model_results['bic_scores']:
                                model_results['bic_scores']['bic'] = {}
                            model_results['bic_scores']['bic'][model_name] = model_cache['bic']
                    except Exception as e:
                        print(f"Error evaluating {model_name} for {name}: {e}")
                
                # Store results for this dataset in this round
                all_rounds_results[round_idx][name] = model_results
                
                if model_name == 'rlig' and 'model' in model_cache:
                    # Use RLiG's built-in evaluate method for consistency
                    rlig_model = model_cache['model']
                    start_time = time.time()
                    
                    if isinstance(y_test, pd.DataFrame):
                        y_test_series = y_test.iloc[:, 0] if y_test.shape[1] == 1 else y_test
                    else:
                        y_test_series = y_test
                    
                    # Built-in evaluation
                    lr_result = rlig_model.evaluate(X_test, y_test_series, model='lr')
                    mlp_result = rlig_model.evaluate(X_test, y_test_series, model='mlp')
                    rf_result = rlig_model.evaluate(X_test, y_test_series, model='rf')
                    
                    # Store results
                    rlig_results = {
                        'LR': lr_result,
                        'MLP': mlp_result,
                        'RF': rf_result,
                        'AVG': (lr_result + mlp_result + rf_result) / 3
                    }
                    
                    for classifier, acc in rlig_results.items():
                        model_results['metrics'][f'RLiG-{classifier}'] = acc
                    
                    # Ensure times dictionary is initialized properly
                    if 'training_time' not in model_results['times']:
                        model_results['times']['training_time'] = {}
                        
                    # Store the time value in the dictionary
                    model_results['times']['training_time']['rlig'] = time.time() - start_time
                    
                    # Store BIC score if available
                    if 'bic' in model_cache and model_cache['bic'] is not None:
                        model_results['bic_scores']['RLiG'] = model_cache['bic']
                else:
                    # Use model-specific test data if available
                    test_X = X_test
                    test_y = y_test
                    
                    # For models that may have model-specific test data
                    if model_name in ['great', 'tabsyn'] and 'X_test' in model_cache and 'y_test' in model_cache:
                        test_X = model_cache['X_test']
                        test_y = model_cache['y_test']
                        print(f"Using model-specific test data for {model_name}")
                    
                    # Standard TSTR evaluation
                    start_time = time.time()
                    tstr_results = evaluate_tstr(synthetic_data, test_X, test_y)
                    eval_time = time.time() - start_time
                    
                    # Store metrics
                    for classifier, acc in tstr_results.items():
                        model_results['metrics'][f'{model_name.upper()}-{classifier}'] = acc
                    
                    # Store time
                    # Ensure times dictionary is initialized properly
                    if 'training_time' not in model_results['times']:
                        model_results['times']['training_time'] = {}
                        
                    # Store the time value in the dictionary
                    model_results['times']['training_time'][model_name] = eval_time
                    
                    # Store BIC score if available
                    if 'bic' in model_cache and model_cache['bic'] is not None:
                        model_results['bic_scores'][model_name.upper()] = model_cache['bic']
            
            # Store results for this dataset and round
            round_results[name] = model_results
        
        # Store this round's results
        all_rounds_results[round_idx] = round_results
    
    # Average results across all rounds
    # Determine number of iterations based on CV or rounds mode
    iterations = n_folds if use_cv else n_rounds
    
    final_results = {}
    
    for dataset_name in datasets.keys():
        # Initialize dataset results
        final_results[dataset_name] = {
            'metrics': {},
            'times': {},
            'bic_scores': {},
            'dataset_name': dataset_name
        }
        
        # Count valid iterations for this dataset
        valid_iterations = 0
        
        # Combine metrics from all iterations (folds or rounds)
        for iter_idx in range(iterations):
            if dataset_name not in all_rounds_results[iter_idx]:
                continue
                
            iteration_data = all_rounds_results[iter_idx][dataset_name]
            valid_iterations += 1
            
            # Accumulate metrics
            for metric_key, metric_value in iteration_data['metrics'].items():
                # Skip None values
                if metric_value is None:
                    continue
                
                # Check if the metric value is a dictionary
                if isinstance(metric_value, dict):
                    # Handle nested dictionary structure
                    if metric_key not in final_results[dataset_name]['metrics']:
                        final_results[dataset_name]['metrics'][metric_key] = {}
                    
                    # Process each model's metric value
                    for model_name, model_value in metric_value.items():
                        if model_value is None:
                            continue
                            
                        if model_name not in final_results[dataset_name]['metrics'][metric_key]:
                            final_results[dataset_name]['metrics'][metric_key][model_name] = 0
                        final_results[dataset_name]['metrics'][metric_key][model_name] += model_value
                else:
                    # Handle simple value (non-dictionary)
                    if metric_key not in final_results[dataset_name]['metrics']:
                        final_results[dataset_name]['metrics'][metric_key] = 0
                    final_results[dataset_name]['metrics'][metric_key] += metric_value
            
            # Accumulate times
            for time_key, time_value in iteration_data['times'].items():
                # Skip None values
                if time_value is None:
                    continue
                
                # Check if the time value is a dictionary
                if isinstance(time_value, dict):
                    # Handle nested dictionary structure
                    if time_key not in final_results[dataset_name]['times']:
                        final_results[dataset_name]['times'][time_key] = {}
                    
                    # Process each model's time value
                    for model_name, model_value in time_value.items():
                        if model_value is None:
                            continue
                            
                        if model_name not in final_results[dataset_name]['times'][time_key]:
                            final_results[dataset_name]['times'][time_key][model_name] = 0
                        final_results[dataset_name]['times'][time_key][model_name] += model_value
                else:
                    # Handle simple value (non-dictionary)
                    if time_key not in final_results[dataset_name]['times']:
                        final_results[dataset_name]['times'][time_key] = 0
                    final_results[dataset_name]['times'][time_key] += time_value
            
            # Accumulate BIC scores
            for bic_key, bic_value in iteration_data['bic_scores'].items():
                # Skip None values
                if bic_value is None:
                    continue
                
                # Check if the BIC value is a dictionary
                if isinstance(bic_value, dict):
                    # Handle nested dictionary structure
                    if bic_key not in final_results[dataset_name]['bic_scores']:
                        final_results[dataset_name]['bic_scores'][bic_key] = {}
                    
                    # Process each model's BIC value
                    for model_name, model_value in bic_value.items():
                        if model_value is None:
                            continue
                            
                        if model_name not in final_results[dataset_name]['bic_scores'][bic_key]:
                            final_results[dataset_name]['bic_scores'][bic_key][model_name] = 0
                        final_results[dataset_name]['bic_scores'][bic_key][model_name] += model_value
                else:
                    # Handle simple value (non-dictionary)
                    if bic_key not in final_results[dataset_name]['bic_scores']:
                        final_results[dataset_name]['bic_scores'][bic_key] = 0
                    final_results[dataset_name]['bic_scores'][bic_key] += bic_value
        
        # Compute averages
        if valid_iterations > 0:
            # Average metrics
            for metric_key in final_results[dataset_name]['metrics'].keys():
                if isinstance(final_results[dataset_name]['metrics'][metric_key], dict):
                    # Handle nested dictionary
                    for model_name in final_results[dataset_name]['metrics'][metric_key].keys():
                        final_results[dataset_name]['metrics'][metric_key][model_name] /= valid_iterations
                else:
                    # Handle simple value
                    final_results[dataset_name]['metrics'][metric_key] /= valid_iterations
            
            # Average times
            for time_key in final_results[dataset_name]['times'].keys():
                if isinstance(final_results[dataset_name]['times'][time_key], dict):
                    # Handle nested dictionary
                    for model_name in final_results[dataset_name]['times'][time_key].keys():
                        final_results[dataset_name]['times'][time_key][model_name] /= valid_iterations
                else:
                    # Handle simple value
                    final_results[dataset_name]['times'][time_key] /= valid_iterations
            
            # Average BIC scores
            for bic_key in final_results[dataset_name]['bic_scores'].keys():
                if isinstance(final_results[dataset_name]['bic_scores'][bic_key], dict):
                    # Handle nested dictionary
                    for model_name in final_results[dataset_name]['bic_scores'][bic_key].keys():
                        final_results[dataset_name]['bic_scores'][bic_key][model_name] /= valid_iterations
                else:
                    # Handle simple value
                    final_results[dataset_name]['bic_scores'][bic_key] /= valid_iterations
    
    if nested_cv:
        print(f"\nAveraged results across {n_rounds} random seed rounds with {n_folds}-fold cross-validation in each")
    elif use_cv:
        print(f"\nAveraged results across {n_folds} cross-validation folds")
    else:
        print(f"\nAveraged results across {n_rounds} random seed rounds")
    return final_results


# ============= RESULTS FORMATTING FUNCTIONS =============

def format_results(results):
    """Format the results into DataFrames for easier analysis"""
    # Define expected models and metric types for consistent column ordering
    models = ['ganblr++', 'ganblr', 'ctgan', 'ctabgan', 'nb', 'rlig', 'great', 'tabsyn']
    metric_types = ['LR', 'MLP', 'RF', 'XGB', 'AVG']
    
    # Initialize results dictionaries
    accuracy_results = {}
    time_results = {}
    bic_results = {}
    
    for dataset, data in results.items():
        # Process metrics - format as MODEL-METRIC_TYPE
        metrics_dict = {}
        
        for metric_key, metric_value in data['metrics'].items():
            if isinstance(metric_value, dict):
                # Extract classifier type (LR, MLP, RF, XGB, AVG) from the metric key
                if '_accuracy' in metric_key:
                    classifier_type = metric_key.split('_')[0]
                    
                    # Process each model's results for this classifier
                    for model_name, model_value in metric_value.items():
                        # Convert model name to uppercase for consistency
                        model_upper = model_name.upper()
                        if model_name.lower() == 'ganblr++':
                            model_upper = 'GANBLR++'
                            
                        # Create column in format "MODEL-METRIC_TYPE"
                        col_name = f"{model_upper}-{classifier_type}"
                        metrics_dict[col_name] = model_value
        
        accuracy_results[dataset] = metrics_dict
        
        # Process times - directly use the times from data
        times_dict = {}
        
        # Debug: print time data to see what we're working with
        print(f"DEBUG - Time data for {dataset}: {data['times']}")
        
        for time_key, time_value in data['times'].items():
            if isinstance(time_value, dict):
                for model_name, model_value in time_value.items():
                    # Format as MODEL-TIME_TYPE
                    model_upper = model_name.upper()
                    if model_name.lower() == 'ganblr++':
                        model_upper = 'GANBLR++'
                    col_name = f"{model_upper}-{time_key}"
                    times_dict[col_name] = model_value
            else:
                # This is for models where time is stored directly (not in a nested dict)
                # Common structure in the working eval_tstr_final.py file
                times_dict[time_key] = time_value
                
        # Extract training times from nested dictionary structure
        # This ensures we extract all time values from the 'training_time' nested dictionary
        if 'training_time' in data['times'] and isinstance(data['times']['training_time'], dict):
            for model_name, time_value in data['times']['training_time'].items():
                # Format model name
                model_upper = model_name.upper()
                if model_name.lower() == 'ganblr++':
                    model_upper = 'GANBLR++'
                
                # Add direct time entry for this model
                times_dict[model_upper] = time_value
                
        time_results[dataset] = times_dict
        
        # Process BIC scores - keeping the same format
        bic_dict = {}
        for bic_key, bic_value in data['bic_scores'].items():
            if isinstance(bic_value, dict):
                for model_name, model_value in bic_value.items():
                    # Format as MODEL-BIC
                    model_upper = model_name.upper()
                    if model_name.lower() == 'ganblr++':
                        model_upper = 'GANBLR++'
                    col_name = f"{model_upper}-{bic_key}"
                    bic_dict[col_name] = model_value
            else:
                bic_dict[bic_key] = bic_value
        bic_results[dataset] = bic_dict
    
    # Create DataFrames from the dictionaries
    accuracy_df = pd.DataFrame.from_dict(accuracy_results, orient='index')
    time_df = pd.DataFrame.from_dict(time_results, orient='index')
    bic_df = pd.DataFrame.from_dict(bic_results, orient='index')
    
    # Sort columns to match the requested order
    def sort_columns(df):
        # Define a custom sorting key
        def sort_key(col_name):
            if '-' in col_name:
                model, metric = col_name.split('-', 1)
                # Get the position of the model and metric
                try:
                    model_idx = [m.upper() for m in models].index(model)
                except ValueError:
                    model_idx = len(models)  # Place unknown models at the end
                
                try:
                    metric_idx = metric_types.index(metric)
                except ValueError:
                    metric_idx = len(metric_types)  # Place unknown metrics at the end
                
                return (model_idx, metric_idx)
            else:
                return (len(models), len(metric_types))  # Other columns at the end
        
        # Sort columns based on the custom key
        if not df.empty:
            sorted_cols = sorted(df.columns, key=sort_key)
            return df[sorted_cols]
        return df
    
    # Apply sorting to all DataFrames
    accuracy_df = sort_columns(accuracy_df)
    time_df = sort_columns(time_df)
    bic_df = sort_columns(bic_df)
    
    return {
        'accuracy': accuracy_df,
        'time': time_df,
        'bic': bic_df
    }


def save_results_to_csv(results_dict, prefix="tstr"):
    """Save results to CSV files"""
    # Extract directory part from prefix if it has one
    if '/' in prefix:
        directory = os.path.dirname(prefix)
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        # Use just the basename part as the prefix
        basename = os.path.basename(prefix)
    else:
        # Default to results directory
        directory = "results"
        basename = prefix
        os.makedirs(directory, exist_ok=True)
    
    for result_type, df in results_dict.items():
        filename = f"{directory}/{basename}_{result_type}_results.csv"
        df.to_csv(filename)
        if result_type == 'time':
            print(f"Saved training time results to {filename} - values are in seconds")
        else:
            print(f"Saved {result_type} results to {filename}")


# ============= ARGUMENT PARSING =============

def parse_args():
    """Parse command line arguments for the TSTR evaluation script"""
    parser = argparse.ArgumentParser(
        description="TSTR (Train on Synthetic, Test on Real) Evaluation Framework for generative models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    print("\nTSTR Evaluation Framework for Generative Models")
    print("===============================================")
    print("Models supported: RLiG, GANBLR, GANBLR++, CTGAN, Naive Bayes, GReaT, TabSyn")
    print("Classifiers: LogisticRegression, MLP, RandomForest, XGBoost (if installed)")
    print("Running with command line arguments enables customization of datasets, models, and parameters.")
    print("\nDiscretization control:")
    print("  --discretize       Apply discretization to continuous features (default)")
    print("  --no-discretize    Do not apply discretization to continuous features")
    
    # Model selection arguments
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        default=['rlig', 'ganblr', 'ganblr++', 'ctgan', 'ctabgan', 'nb', 'great', 'tabsyn'],
        help="List of models to evaluate. Options: rlig, ganblr, ganblr++, ctgan, ctabgan, nb, great, tabsyn"
    )

    """PokerHand: 158
    NSL-KDD: data/nsl-kdd/KDDTrain+_20Percent.arff
    Connect-4: 26
    Credit: 27
    Adult: 2
    Chess: 22
    letter_rocog: 59
    Magic: 159
    Nursery: 76
    Room Occupancy: 864
    Car: 19
    Maternal Health: 863
    Loan & Credit: from local directory
    """
    
    # Dataset selection arguments
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs="+", 
        default=['Rice', 'TicTacToe', 'PokerHand', 'Connect-4', 'Credit',
                 'Adult', 'Chess', 'letter_rocog', 'Magic', 'Nursery', 'Room Occupancy',
                 'Car', 'Maternal Health', 'Loan'],
        help="List of dataset names to evaluate"
    )
    
    # Add UCI dataset IDs
    parser.add_argument(
        "--uci_ids", 
        type=int, 
        nargs="+", 
        default=[545, 101, 158, 26, 27, 2, 22, 59, 159, 76, 864, 19, 863],  # Default: Rice and TicTacToe
        help="List of UCI dataset IDs to use"
    )
    
    # Add local dataset paths
    parser.add_argument(
        "--local_datasets", 
        type=str, 
        nargs="+", 
        default=['data/loan_approval_dataset.csv', 'data/UCI_Credit_Card.csv'],
        help="List of paths to local dataset files (.arff or .csv)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--n_rounds", 
        type=int, 
        default=3,
        help="Number of evaluation rounds for robust results"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Output options
    parser.add_argument(
        "--output_prefix", 
        type=str, 
        default="disc_tstr",
        help="Prefix for output CSV files (default: disc_tstr for discretized results)"
    )
    
    # Discretization control
    parser.add_argument(
        "--discretize",
        action="store_true",
        default=True,
        help="Apply discretization to continuous features"
    )
    
    parser.add_argument(
        "--no-discretize",
        action="store_false",
        dest="discretize",
        help="Do not apply discretization to continuous features"
    )
    
    # Cross-validation control
    parser.add_argument(
        "--use-cv",
        action="store_true",
        default=False,
        help="Use k-fold cross-validation instead of random seed rounds"
    )
    
    parser.add_argument(
        "--nested-cv",
        action="store_true",
        default=False,
        help="Use k-fold cross-validation within each random seed round"
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=2,
        help="Number of folds for cross-validation (default: 2)"
    )
    
    # Additional model parameters
    parser.add_argument(
        "--ctgan_epochs", 
        type=int, 
        default=50,
        help="Number of epochs for CTGAN training"
    )
    
    parser.add_argument(
        "--small_ctgan", 
        action="store_true",
        help="Use fewer epochs (10) for CTGAN and CTABGAN to speed up training"
    )
    
    parser.add_argument(
        "--rlig_episodes", 
        type=int, 
        default=2,
        help="Number of episodes for RLiG training"
    )
    
    parser.add_argument(
        "--rlig_epochs", 
        type=int, 
        default=5,
        help="Number of epochs for RLiG training"
    )

    parser.add_argument(
        "--great_bs",
        type=int,
        default=1,
        help="Number of batch size for GReaT training"
    )

    parser.add_argument(
        "--great_epochs",
        type=int,
        default=1,
        help="Number of epochs for GReaT training"
    )
    
    parser.add_argument(
        "--tabsyn_epochs",
        type=int,
        default=50,
        help="Number of epochs for TabSyn training"
    )
    
    # Verbose mode
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Display discretization status
    print(f"\nRunning with discretization: {'ENABLED' if args.discretize else 'DISABLED'}")
    
    # Set up datasets dictionary based on provided arguments
    datasets = {}
    
    # Add UCI datasets using IDs
    if UCI_AVAILABLE:
        for i, dataset_id in enumerate(args.uci_ids):
            if i < len(args.datasets):
                datasets[args.datasets[i]] = dataset_id
            else:
                # Create a generic name if not enough names provided
                datasets[f"UCI_{dataset_id}"] = dataset_id
    
    # Add local datasets
    for i, dataset_path in enumerate(args.local_datasets):
        # Calculate the appropriate index for naming
        idx = len(datasets)
        print(idx, len(args.datasets))
        if idx < len(args.datasets):
            datasets[args.datasets[idx]] = dataset_path
        else:
            # Extract filename as dataset name if not enough names provided
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            datasets[dataset_name] = dataset_path
    
    # Limit to requested datasets if both lists were provided
    if len(args.datasets) < len(datasets):
        datasets = {k: datasets[k] for k in args.datasets if k in datasets}

    # Apply small_ctgan option if specified
    if args.small_ctgan:
        args.ctgan_epochs = 10
        print(f"Note: Using reduced CTGAN/CTABGAN epochs ({args.ctgan_epochs}) for faster training")
    
    # Run the TSTR comparison with specified models and parameters
    results = compare_models_tstr(
        datasets,
        models=args.models,
        n_rounds=args.n_rounds,
        seed=args.seed,
        rlig_episodes=args.rlig_episodes,
        rlig_epochs=args.rlig_epochs,
        ctgan_epochs=args.ctgan_epochs,
        great_bs=args.great_bs,
        great_epochs=args.great_epochs,
        tabsyn_epochs=args.tabsyn_epochs,
        verbose=args.verbose,
        discretize=args.discretize,
        use_cv=args.use_cv,
        n_folds=args.n_folds,
        nested_cv=args.nested_cv
    )
    
    # Format and display results
    formatted_results = format_results(results)
    
    print("\n\n=== TSTR ACCURACY RESULTS ===")
    print(formatted_results['accuracy'])
    print("\n\n=== TIME RESULTS (seconds) ===")
    print(formatted_results['time'])
    print("\n\n=== BIC SCORE RESULTS ===")
    print(formatted_results['bic'])
    
    # Save results to CSV
    try:
        # Simple prefix, just distinguish between discretized and raw
        output_prefix = "disc_tstr" if args.discretize else "raw_tstr"
        
        # Create timestamp-based directory for results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = f"results/{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        # Save only the essential results files (accuracy, time, bic) without redundancy
        save_results_to_csv(formatted_results, prefix=f"{result_dir}/{output_prefix}")
        print(f"\nResults saved to CSV files in directory '{result_dir}' with prefix '{output_prefix}'.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")