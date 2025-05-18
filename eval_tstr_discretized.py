"""
TSTR (Train on Synthetic, Test on Real) Evaluation Framework with Discretization

This script implements a proper TSTR evaluation for generative models with explicit discretization
of continuous variables using quantile-based binning (7 bins).

This version is specifically designed to improve performance for models that work well
with discretized data, particularly GReaT and Distribution Sampling, while maintaining a fair comparison
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
- Distribution Sampling: Tabular data synthesis with statistical modeling

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

from torch.utils.data import DataLoader

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
    # Add distribution sampling directory to Python path so imports work
    import sys
    import os

    dist_sampl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distsampl')
    if dist_sampl_path not in sys.path:
        sys.path.append(dist_sampl_path)

    from distsampl.dist_sampling import DistSampling

    DIST_SAMPL_AVAILABLE = True
except ImportError as e:
    print(f"Distribution Sampling is not available. Will be skipped. Error: {e}")
    DIST_SAMPL_AVAILABLE = False

# TabDiff
try:
    tabdiff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TabDiff')
    if tabdiff_path not in sys.path:
        sys.path.append(tabdiff_path)

    from TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from TabDiff.tabdiff.trainer import Trainer
    from TabDiff.utils_train import TabDiffDataset

    from TabDiff.tabdiff.modules.main_modules import UniModMLP
    from TabDiff.tabdiff.modules.main_modules import Model

    from TabDiff.tabdiff.metrics import TabMetrics
    from TabDiff.utils_train import TabDiffDataset

    TABDIFF_AVAILABLE = True
except ImportError as e:
    print(f"TabDiff is not available. Will be skipped. Error: {e}")
    TABDIFF_AVAILABLE = False


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


def preprocess_data(X, y, name, discretize=True, model_name=None, cv_fold=None, n_folds=None):
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
    name: Name of dataset
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

    # Clean the target values for 'adult' dataset
    if name == 'adult':
        y = y.transform(lambda col: col.astype(str).str.replace('.', '', regex=False))

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
        print(f"Using {n_folds}-fold cross-validation (fold {cv_fold + 1}/{n_folds})")

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

                if name == "letter_recog":
                    # or y = df.iloc[:, [0]]
                    y = df[['lettr']]
                else:
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
                                        test_size=remaining_size / len(X_remaining),
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
                                test_size=min(remaining_slots / len(common_indices), 1.0),
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
                        test_size=min(sample_size / len(X_train), 0.8),
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
                    test_size=min(sample_size / len(X_train), 0.8),
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

        # otherwise for Mac, use this
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


def train_dist_sampl(X_train, y_train, epochs=50, random_seed=42):
    """Train the Distribution Sampling synthesizer
    
    Distribution Sampling is a tabular data synthesis method that uses statistical
    distributions to model and generate synthetic data.
    
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
    if not DIST_SAMPL_AVAILABLE:
        return None

    try:
        # Prepare data for Distribution Sampling (we need to combine X and y)
        combined_data = pd.concat([X_train, y_train], axis=1)

        # Identify categorical columns
        categorical_cols = []
        for col in combined_data.columns:
            if len(np.unique(combined_data[col])) < 10:  # Heuristic for categorical columns
                categorical_cols.append(col)

        # Initialize DistSampling with conservative settings and consistent random seed
        print(f"Training Distribution Sampling with {epochs} epochs and random_seed={random_seed}")
        dist_sampl_model = DistSampling(
            train_data=combined_data,
            categorical_columns=categorical_cols,
            epochs=epochs,
            verbose=True,
            random_seed=random_seed
        )

        # Train the model
        dist_sampl_model.fit()
        return dist_sampl_model
    except Exception as e:
        print(f"Error training Distribution Sampling model: {e}")
        return None


# Train TabDiff
def train_tabdiff(train_data, train_loader, name, epochs=50, random_seed=42):
    # run_tabdiff(config_path="configs/tabdiff_config.yaml")

    import json
    info_path = f'data/{name}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(curr_dir, "ckpt")):
        os.makedirs(os.path.join(curr_dir, "ckpt"), exist_ok=True)

    if not os.path.exists(os.path.join(curr_dir, "result")):
        os.makedirs(os.path.join(curr_dir, "result"), exist_ok=True)

    model_save_path = f'{curr_dir}/ckpt/{name}/'
    result_save_path = model_save_path.replace('ckpt', 'result')  # i.e., f'{curr_dir}/results/{dataname}/'

    if model_save_path is not None:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    if result_save_path is not None:
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)

    metric_list = ["density"]
    ## Load Metrics
    real_data_path = f'synthetic/{name}/real.csv'
    test_data_path = f'synthetic/{name}/test.csv'
    val_data_path = f'synthetic/{name}/val.csv'
    if not os.path.exists(val_data_path):
        print(
            f"{name} does not have its validation set. During MLE evaluation, a validation set will be splitted from the training set!")
        val_data_path = None

    metrics = TabMetrics(real_data_path, test_data_path, val_data_path, info, device, metric_list=metric_list)

    # from TabDiff.utils_train import TabDiffDataset
    # data_dir = f'data/{name}'
    # train_data = TabDiffDataset(name, data_dir, info)
    d_numerical, categories = train_data.d_numerical, train_data.categories

    backbone = UniModMLP(
        d_numerical=d_numerical,
        categories=(categories + 1).tolist(),  # add one for the mask category,
        num_layers=2,
        d_token=4,
        n_head=1,
        factor=32,
        bias=True,
        dim_t=1024,
        use_mlp=True

    )
    model = Model(backbone, precond=True, sigma_data=1.0, net_conditioning="sigma")
    model.to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        device=device,
        num_timesteps=50,
        scheduler='power_mean',  # 'power_mean', 'power_mean_unified', 'power_mean_per_column'
        cat_scheduler='log_linear_per_column',  # 'log_linear', 'log_linear_unified', 'log_linear_per_column'
        noise_dist='uniform_t',  # 'uniform_t' or 'log_norm'
        noise_dist_params={'P_mean': -1.2, 'P_std': 1.2},
        sigma_min=0.002,
        noise_schedule_params={"sigma_max": 80,
                               "rho": 7,
                               "eps_max": 1e-3,
                               "eps_min": 1e-5,
                               "rho_init": 7.0,
                               "rho_offset": 5.0,
                               "k_init": -6.0,
                               "k_offset": 1.0},
        edm_params={"precond": True,
                    "sigma_data": 1.0,
                    "net_conditioning": "sigma"},
        sampler_params={"stochastic_sampler": True,
                        "second_order_correction": True}
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    diffusion.to(device)
    diffusion.train()

    data_dir = f'data/{name}'
    val_data = TabDiffDataset(name, data_dir, info, isTrain=False)

    ## Disable Wandb
    import wandb
    project_name = f"tabdiff_{name}"
    logger = wandb.init(
        project=project_name,
        name=None,
        config=None,
        mode='disabled',
    )
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        steps=2,
        lr=0.001,
        weight_decay=0,
        ema_decay=0.997,
        batch_size=4096,
        check_val_every=2000,
        lr_scheduler="reduce_lr_on_plateau",
        factor=0.90,  # hyperparam for reduce_lr_on_plateau
        reduce_lr_patience=50,  # hyperparam for reduce_lr_on_plateau
        closs_weight_schedule="anneal",
        c_lambda=1.0,
        d_lambda=1.0,
        sample_batch_size=10000,
        num_samples_to_generate=None,
        model_save_path=model_save_path,
        result_save_path=result_save_path,
        device=device,
        ckpt_path=None,
        y_only=False
    )
    trainer.run_loop()
    return trainer


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
                test_size=min(25000 / len(synthetic_data), 0.5),  # Take at most 50% or 25,000
                stratify=synthetic_data[target_col],
                random_state=42
            )
        else:
            # Simple random sample if no target column or only one class
            sample_size = min(25000, len(synthetic_data))
            sampled_data = synthetic_data.sample(sample_size, random_state=42)

        sampled_data.to_csv(file_path, index=False)
        print(
            f"Saved representative sample of {model_name.upper()} synthetic data ({len(sampled_data)} of {len(synthetic_data)} samples) to {file_path}")

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
                print(f"Generating batch {i + 1}/{num_batches}")
                this_batch_size = min(batch_size, n_samples - i * batch_size)
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
                print(f"Generating batch {i + 1}/{num_batches}")
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


def generate_dist_sampl_synthetic_data(dist_sampl_model, train_data, n_samples=None):
    """Generate synthetic data from Distribution Sampling model"""
    if not DIST_SAMPL_AVAILABLE or dist_sampl_model is None:
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
                print(f"Generating batch {i + 1}/{num_batches}")
                this_batch_size = min(batch_size, n_samples - i * batch_size)
                batch = dist_sampl_model.sample(this_batch_size)
                batches.append(batch)

            synthetic_data = pd.concat(batches, ignore_index=True)
        else:
            # Regular generation for other platforms
            synthetic_data = dist_sampl_model.sample(n_samples)

        print(f"Generated {len(synthetic_data)} synthetic samples from Distribution Sampling")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from Distribution Sampling: {e}")

        # Fallback: if sampling fails, try to sample a smaller number
        try:
            fallback_samples = min(n_samples, 500)
            print(f"Trying fallback with {fallback_samples} samples")
            synthetic_data = dist_sampl_model.sample(fallback_samples)
            print(f"Generated {len(synthetic_data)} synthetic samples as fallback")
            return synthetic_data
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return None


# Generate synthetic data
def generate_tabdiff_synthetic_data(tabdiff_model, train_data, n_samples=None):
    """Generate synthetic data from TABDIFF model"""
    if not TABDIFF_AVAILABLE or tabdiff_model is None:
        return None

    if n_samples is None:
        n_samples = len(train_data)

    # try:
    # For M1/M2 Macs, generate in smaller batches
    import os
    if hasattr(os, 'uname') and os.uname().machine == 'arm64' and n_samples > 500:
        print(f"Generating {n_samples} samples in smaller batches for Apple Silicon compatibility")
        batch_size = 500
        # num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        num_batches = 2

        # Generate in batches and concatenate
        batches = []
        for i in range(num_batches):
            print(f"Generating batch {i + 1}/{num_batches}")
            this_batch_size = min(batch_size, n_samples - i * batch_size)
            tabdiff_model.diffusion.eval()
            batch = tabdiff_model.sample_synthetic(this_batch_size)
            batches.append(batch)

        synthetic_data = pd.concat(batches, ignore_index=True)
    else:
        # Regular generation for other platforms
        tabdiff_model.diffusion.eval()
        synthetic_data = tabdiff_model.sample_synthetic(n_samples, ema=True)
        print("Synthetic data generated", synthetic_data.head())
    print(f"Generated {len(synthetic_data)} synthetic samples from TABDIFF")
    return synthetic_data
    # except Exception as e:
    #     print(f"Error generating synthetic data from TABDIFF: {e}")

    # # Fallback: if sampling fails, try to sample a smaller number
    # try:
    #     fallback_samples = min(n_samples, 500)
    #     print(f"Trying fallback with {fallback_samples} samples")
    #     tabdiff_model.diffusion.eval()
    #     synthetic_data = tabdiff_model.sample_synthetic(fallback_samples)
    #     print(f"Generated {len(synthetic_data)} synthetic samples as fallback")
    #     return synthetic_data
    # except Exception as fallback_error:
    #     print(f"Fallback also failed: {fallback_error}")
    #     return None


# ============= EVALUATION FUNCTIONS =============
import pandas as pd  # Assuming pandas is used by evaluate_tstr or for y_test handling
import numpy as np  # Assuming numpy is used by evaluate_tstr


# Placeholder for your actual evaluate_tstr function:
# def evaluate_tstr(synthetic_data, X_test, y_test, target_col='target'):
#     # ... (your implementation here)
#     # Expected to return a dictionary like:
#     # {'LR': 0.75, 'MLP': 0.78, 'RF': 0.80, 'XGB': 0.82, 'AVG': 0.7875}
#     # or {'LR': None, ...} if errors occur or synthetic_data is None.
#     pass

def evaluate_models_on_fold(dataset_name, synthetic_data_cache, X_test, y_test, generative_models_to_evaluate):
    """
    Evaluates specified generative models on a specific fold using their synthetic data.

    This function trains downstream classifiers (LR, MLP, RF, XGBoost) on the
    synthetic data generated by each generative model and evaluates their performance
    on the provided real test data (X_test, y_test). It also collects the
    training time and BIC score (if available) for each generative model.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset being evaluated.
    synthetic_data_cache : dict
        A cache containing pre-generated synthetic data and metadata for various models.
        Expected structure:
        synthetic_data_cache[dataset_name]['models'][model_name]['data'] : DataFrame (synthetic data)
        synthetic_data_cache[dataset_name]['models'][model_name]['train_time'] : float (training time in seconds)
        synthetic_data_cache[dataset_name]['models'][model_name]['bic'] : float (BIC score, optional)
    X_test : pd.DataFrame
        Real test features for the current fold.
    y_test : pd.DataFrame or pd.Series
        Real test target for the current fold.
    generative_models_to_evaluate : list
        A list of strings, where each string is the name of a generative model
        (e.g., ['ctgan', 'rlig', 'nb']) to be evaluated from the cache.

    Returns:
    --------
    dict
        A dictionary containing the evaluation results for the specified models on this fold.
        Structure:
        {
            'metrics': {
                'LR_accuracy': {'model1': acc1, 'model2': acc2, ...},
                'MLP_accuracy': {'model1': acc1, ...},
                ... (RF, XGB, AVG)
            },
            'times': {
                'training_time': {'model1': time1, 'model2': time2, ...}
            },
            'bic_scores': {
                'bic': {'model1': bic1, 'model2': bic2, ...}
            },
            'dataset_name': str (name of the dataset)
        }
    """
    fold_results = {
        'metrics': {},
        'times': {'training_time': {}},
        'bic_scores': {'bic': {}},
        'dataset_name': dataset_name
    }

    # Define expected classifier accuracy metric names for consistent reporting structure
    # These correspond to the keys expected from evaluate_tstr (e.g., 'LR', 'MLP')
    # which will be appended with '_accuracy'.
    classifier_abbreviations = ['LR', 'MLP', 'RF', 'XGB', 'AVG']
    for abbr in classifier_abbreviations:
        fold_results['metrics'][f"{abbr}_accuracy"] = {}

    if dataset_name not in synthetic_data_cache or 'models' not in synthetic_data_cache[dataset_name]:
        print(
            f"Warning: Synthetic data cache or 'models' key not found for dataset '{dataset_name}' in evaluate_models_on_fold.")
        # Populate all expected models with None if cache is missing for the dataset
        for model_name_gen in generative_models_to_evaluate:
            for abbr in classifier_abbreviations:
                fold_results['metrics'][f"{abbr}_accuracy"][model_name_gen] = None
            fold_results['times']['training_time'][model_name_gen] = None
            fold_results['bic_scores']['bic'][model_name_gen] = None
        return fold_results

    all_cached_generative_models = synthetic_data_cache[dataset_name]['models']

    for model_name_gen in generative_models_to_evaluate:
        print(f"\n-- Evaluating {model_name_gen.upper()} synthetic data on fold for dataset '{dataset_name}' --")

        if model_name_gen not in all_cached_generative_models:
            print(
                f"Warning: No cached data found for generative model '{model_name_gen}' on dataset '{dataset_name}'. Results for this model will be None.")
            for abbr in classifier_abbreviations:
                fold_results['metrics'][f"{abbr}_accuracy"][model_name_gen] = None
            fold_results['times']['training_time'][model_name_gen] = None
            fold_results['bic_scores']['bic'][model_name_gen] = None
            continue

        model_cache_item = all_cached_generative_models[model_name_gen]
        synthetic_data = model_cache_item.get('data')

        # 1. Get and store training time for the generative model
        actual_train_time = model_cache_item.get('train_time')
        if actual_train_time is None:
            print(f"Warning: Actual training time not found for '{model_name_gen}' on '{dataset_name}'. Using 0.0s.")
            actual_train_time = 0.0
        fold_results['times']['training_time'][model_name_gen] = actual_train_time

        # 2. Get and store BIC score if available
        bic_score = model_cache_item.get('bic')
        fold_results['bic_scores']['bic'][model_name_gen] = bic_score  # Stores None if bic_score is None or key missing

        # 3. Evaluate TSTR if synthetic data is available
        if synthetic_data is None:
            print(
                f"No synthetic data available for '{model_name_gen}' on '{dataset_name}'. TSTR accuracies will be None.")
            for abbr in classifier_abbreviations:
                fold_results['metrics'][f"{abbr}_accuracy"][model_name_gen] = None
            continue

        try:
            # Call evaluate_tstr to get accuracies from downstream classifiers
            tstr_accuracies = evaluate_tstr(synthetic_data, X_test, y_test)

            if tstr_accuracies:  # Ensure tstr_accuracies is not None and not empty
                for classifier_abbr, accuracy_value in tstr_accuracies.items():
                    metric_key = f"{classifier_abbr}_accuracy"  # e.g., 'LR_accuracy'
                    if metric_key in fold_results['metrics']:  # Check if it's an expected metric
                        fold_results['metrics'][metric_key][model_name_gen] = accuracy_value
                    else:
                        print(
                            f"Warning: Unexpected metric key '{metric_key}' from evaluate_tstr for model '{model_name_gen}'.")
                # Ensure all expected classifier metrics are populated, even if not returned by evaluate_tstr for some reason
                for abbr in classifier_abbreviations:
                    expected_metric_key = f"{abbr}_accuracy"
                    if model_name_gen not in fold_results['metrics'][expected_metric_key]:
                        fold_results['metrics'][expected_metric_key][model_name_gen] = tstr_accuracies.get(abbr, None)

            else:  # tstr_accuracies was None or empty
                print(
                    f"evaluate_tstr returned no results for '{model_name_gen}' on '{dataset_name}'. Accuracies set to None.")
                for abbr in classifier_abbreviations:
                    fold_results['metrics'][f"{abbr}_accuracy"][model_name_gen] = None

        except Exception as e:
            print(f"Error during TSTR evaluation for '{model_name_gen}' on dataset '{dataset_name}' (fold): {e}")
            for abbr in classifier_abbreviations:
                fold_results['metrics'][f"{abbr}_accuracy"][model_name_gen] = None

    return fold_results


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


"""Helper functions"""


# Helper function to get unique values regardless of type
def get_unique_values(data):
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0].unique()
    else:
        return data.unique()


# Helper function to get data for encoding
def get_data_for_encoding(data):
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    else:
        return data


def evaluate_tstr(synthetic_data, X_test, y_test,
                  # dataname,
                  target_col='target'):
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

    # try:
    # Split synthetic data into features and target
    if target_col in synthetic_data.columns:
        syn_X = synthetic_data.drop(target_col, axis=1)
        syn_y = synthetic_data[[target_col]].rename(columns={target_col: 'target'})
    else:
        # If target column isn't found, assume last column is target
        syn_X = synthetic_data.iloc[:, :-1]
        syn_y = synthetic_data.iloc[:, -1]

    # Debug target variables
    print(f"Synthetic target type: {type(syn_y)}, values: {syn_y.head()}")
    print(f"Test target type: {type(y_test)}, values: {y_test.head()}")

    # Initialize label encoder
    label_encoder = LabelEncoder()

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
                # If conversion fails, just use label encoder
                print(f"Numeric conversion failed for {name}, using label encoder")

            # If it's still a DataFrame with 1 column, convert to Series
            if isinstance(y_var, pd.DataFrame) and y_var.shape[1] == 1:
                if name == "synthetic":
                    syn_y = y_var.iloc[:, 0]
                else:
                    y_test = y_var.iloc[:, 0]

    # Use Label Encoding
    # Check if unique values are different and apply encoding if needed
    syn_unique = get_unique_values(syn_y)
    test_unique = get_unique_values(y_test)

    print(f"Synthetic unique values: {syn_unique}")
    print(f"Test unique values: {test_unique}")

    if not np.array_equal(sorted(syn_unique), sorted(test_unique)):
        print("Unique values are different, applying label encoding...")

        # Apply label encoding
        syn_y_data = get_data_for_encoding(syn_y)
        test_y_data = get_data_for_encoding(y_test)

        syn_y_encoded = label_encoder.fit_transform(syn_y_data)

        # Convert syn_y back to DataFrame format
        if isinstance(syn_y, pd.DataFrame):
            # Keep the same column name and index
            syn_y = pd.DataFrame(syn_y_encoded,
                                 columns=syn_y.columns,
                                 index=syn_y.index)
        else:
            # Create a new DataFrame with appropriate column name
            syn_y = pd.DataFrame(syn_y_encoded,
                                 columns=['target'],
                                 index=syn_y.index if hasattr(syn_y, 'index') else None)

        print(f"Synthetic labels encoded: {syn_y}")
        print(f"Test labels encoded: {y_test}")
    else:
        print("Unique values are the same, no additional encoding needed")

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

    # For TabDiff, convert period (.) to underscore (_) in column names
    syn_X = syn_X.rename(columns={col: col.replace('.', '_') for col in syn_X.columns})
    syn_X = syn_X.rename(columns={col: col.replace('-', '_') for col in syn_X.columns})

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

    # CRITICAL: Preprocess both datasets to ensure consistent data types
    print("Preprocessing data to ensure consistent types...")

    # Make copies to avoid modifying original data
    syn_X_processed = syn_X.copy()
    X_test_processed = X_test.copy()

    # For each column, determine if it should be categorical or numerical
    categorical_columns = []
    numerical_columns = []

    for col in X_test.columns:
        # Get all unique values from both datasets
        syn_vals = syn_X_processed[col].dropna().unique()
        test_vals = X_test_processed[col].dropna().unique()
        all_vals = np.concatenate([syn_vals, test_vals])

        # Try to determine if this is categorical or numeric
        is_categorical = False

        # Check for string values that can't be converted to numbers
        for val in all_vals:
            if isinstance(val, str):
                # Try to convert to float
                try:
                    float(val)
                except (ValueError, TypeError):
                    is_categorical = True
                    break
            elif not isinstance(val, (int, float, np.integer, np.floating)):
                is_categorical = True
                break

        # Also check if it has very few unique values (likely categorical)
        if not is_categorical:
            unique_ratio = len(np.unique(all_vals)) / len(all_vals)
            if unique_ratio < 0.1:  # Less than 10% unique values
                is_categorical = True

        print(f"Column '{col}' detected as {'categorical' if is_categorical else 'numerical'}")

        if is_categorical:
            categorical_columns.append(col)
            # Convert to string to ensure consistency
            syn_X_processed[col] = syn_X_processed[col].astype(str)
            X_test_processed[col] = X_test_processed[col].astype(str)
        else:
            numerical_columns.append(col)
            # Convert to float, handling any conversion errors
            syn_X_processed[col] = pd.to_numeric(syn_X_processed[col], errors='coerce')
            X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')

    print(f"Categorical columns: {categorical_columns}")
    print(f"Numerical columns: {numerical_columns}")

    # Define classification models
    # Check if we have enough data for early stopping
    if len(syn_y) >= 20:  # Need at least 10 samples for reasonable validation split
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'MLP': MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1),
            'RF': RandomForestClassifier(n_estimators=100)
        }
    else:
        # For small datasets, disable early stopping
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'MLP': MLPClassifier(max_iter=500, early_stopping=False),
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
                    use_label_encoder=False  # Compatibility for older versions
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

    # Create preprocessing pipeline
    from sklearn.compose import ColumnTransformer

    # Build transformers list
    transformers = []
    if numerical_columns:
        transformers.append(('num', StandardScaler(), numerical_columns))
    if categorical_columns:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype='float64'), categorical_columns))

    if transformers:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
    else:
        # If no specific preprocessing needed, use identity
        preprocessor = None

    # Get feature categories for one-hot encoding
    # categories = [np.unique(np.concatenate([syn_X[col].unique(), X_test[col].unique()])) for col in X_test.columns]
    # categories = [
    #     np.unique(np.concatenate([
    #         syn_X[col].astype(str).unique(),
    #         X_test[col].astype(str).unique()
    #     ]))
    #     for col in X_test.columns
    # ]

    for name, model in models.items():
        # try:
        print(f"Training {name} on synthetic data...")

        if preprocessor is not None:
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
        else:
            # No preprocessing needed
            pipeline = Pipeline([
                ('model', model)
            ])

        # # # Sort categories for numerical columns
        # # for i, col in enumerate(X_test.columns):
        # #     if pd.api.types.is_numeric_dtype(X_test[col]):
        # #         categories[i] = np.sort(categories[i])
        #
        # # # Convert categories properly with explicit sorting for numerical columns
        # # corrected_categories = []
        # # for i, col in enumerate(X_test.columns):
        # #     cat_values = np.concatenate([syn_X[col].unique(), X_test[col].unique()])
        # #
        # #     # Check if this is a numeric column
        # #     is_numeric = pd.api.types.is_numeric_dtype(X_test[col])
        # #     print(col, is_numeric)
        # #
        # #     if is_numeric:
        # #         # Convert to float and sort
        # #         numeric_cats = np.unique(cat_values.astype(float))
        # #         corrected_categories.append(np.sort(numeric_cats))
        # #     else:
        # #         # Keep as categorical
        # #         corrected_categories.append(np.unique(cat_values))
        #
        # # # Explicitly check if column contains non-numeric strings before conversion
        # # corrected_categories = []
        # # for i, col in enumerate(X_test.columns):
        # #     cat_values = np.concatenate([syn_X[col].unique(), X_test[col].unique()])
        # #
        # #     # Check if ALL values can be converted to float
        # #     try:
        # #         # Try converting to float - if it works for all values, it's numeric
        # #         _ = [float(x) for x in cat_values]
        # #         is_numeric = True
        # #     except (ValueError, TypeError):
        # #         # If ANY conversion fails, treat as categorical
        # #         is_numeric = False
        # #
        # #     if is_numeric:
        # #         # Convert to float and sort
        # #         numeric_cats = np.unique(np.array([float(x) for x in cat_values]))
        # #         corrected_categories.append(np.sort(numeric_cats))
        # #     else:
        # #         # Keep as categorical
        # #         corrected_categories.append(np.unique(cat_values))
        # #
        # #     print(f"Column '{col}' is {'numeric' if is_numeric else 'categorical'}")
        #
        # # Detect categorical columns by examining data types and values
        # corrected_categories = []
        # for i, col in enumerate(X_test.columns):
        #     # Get unique values from both datasets
        #     syn_vals = syn_X[col].unique().tolist()
        #     test_vals = X_test[col].unique().tolist()
        #     all_vals = syn_vals + test_vals
        #
        #     # Try to determine if this is categorical or numeric
        #     is_categorical = False
        #
        #     # Check for non-numeric values
        #     for val in all_vals:
        #         # Skip NaN values in check
        #         if pd.isna(val):
        #             continue
        #         # If we find any string or non-numeric value, mark as categorical
        #         if isinstance(val, str) and not val.strip().replace('.', '', 1).replace('-', '', 1).isdigit():
        #             is_categorical = True
        #             break
        #         # Check other non-numeric types
        #         if not isinstance(val, (int, float, np.integer, np.floating)):
        #             is_categorical = True
        #             break
        #
        #     print(f"Column '{col}' detected as {'categorical' if is_categorical else 'numeric'}")
        #
        #     # Process based on detected type
        #     if is_categorical:
        #         # Convert all to strings for categorical column
        #         all_vals_str = [str(x) for x in all_vals if not pd.isna(x)]
        #         unique_vals = list(set(all_vals_str))
        #         corrected_categories.append(np.array(unique_vals))
        #     else:
        #         # Process as numeric column
        #         numeric_vals = []
        #         for val in all_vals:
        #             if pd.isna(val):
        #                 continue
        #             try:
        #                 numeric_vals.append(float(val))
        #             except (ValueError, TypeError):
        #                 # If any conversion fails, we need to treat as strings
        #                 is_categorical = True
        #                 break
        #
        #         if is_categorical:
        #             # Handle the fallback case
        #             print(f"Fallback: Column '{col}' contains mixed types, treating as categorical")
        #             all_vals_str = [str(x) for x in all_vals if not pd.isna(x)]
        #             unique_vals = list(set(all_vals_str))
        #             corrected_categories.append(np.array(unique_vals))
        #         else:
        #             # All numeric values
        #             unique_vals = sorted(set(numeric_vals))
        #             corrected_categories.append(np.array(unique_vals))
        #
        # # Let scikit-learn handle categories automatically
        # from sklearn.compose import ColumnTransformer
        #
        # # Create a more robust pipeline
        # pipeline = Pipeline([
        #     ('preprocessor', ColumnTransformer(
        #         transformers=[
        #             ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        #              list(range(len(X_test.columns))))
        #         ],
        #         remainder='passthrough'
        #     )),
        #     ('model', model)
        # ])

        # pipeline = Pipeline([
        #     ('encoder', OneHotEncoder(categories=corrected_categories, handle_unknown='ignore', sparse_output=False)),
        #     ('model', model)
        # ])

        # Train on synthetic data (preprocessed)
        pipeline.fit(syn_X_processed, syn_y)

        # Test on real data (preprocessed)
        y_pred = pipeline.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} TSTR accuracy: {acc:.4f}")
        # except Exception as e:
        #     print(f"Error evaluating {name}: {e}")
        #     results[name] = None

    # Calculate average accuracy across all models (as done in the paper)
    valid_accs = [acc for acc in results.values() if acc is not None]
    if valid_accs:
        results['AVG'] = sum(valid_accs) / len(valid_accs)
        print(f"Average TSTR accuracy: {results['AVG']:.4f}")
    else:
        results['AVG'] = None

    return results
    # except Exception as e:
    #     print(f"Error in TSTR evaluation: {e}")
    #     return {'LR': None, 'MLP': None, 'RF': None, 'XGB': None, 'AVG': None}


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
        # Note: The train_and_evaluate_* functions (originally located here) have been removed
        # as they are not used in the current implementation. The actual model evaluation
        # is done in the evaluate_models_on_fold() function.

        print(f"Error with Naive Bayes: {e}")


# ============= MAIN COMPARISON FUNCTION =============


def compare_models_tstr(datasets, models=None, n_rounds=3, seed=42, rlig_episodes=2, rlig_epochs=5,
                        ctgan_epochs=50, great_bs=1, great_epochs=5, dist_sampl_epochs=50, verbose=False,
                        discretize=True,
                        use_cv=False, n_folds=2, nested_cv=False, tabdiff_epochs=5):
    """
    Compare generative models using TSTR methodology as described in the paper
    with multiple rounds of cross-validation for robustness

    Parameters:
    -----------
    datasets : dict
        Dictionary mapping dataset names to dataset sources
    models : list or None
        List of models to evaluate. If None, evaluate all available models.
        Options: 'rlig', 'ganblr', 'ganblr++', 'ctgan', 'nb', 'great', 'dist_sampl'
    n_rounds : int
        Number of rounds of cross-validation to run (default: 3)
    # ... (other parameters)
    """
    if models is None:
        models = ['rlig', 'ganblr', 'ganblr++', 'ctgan', 'ctabgan', 'nb', 'great', 'dist_sampl', 'tabdiff']

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
        print(f"  - GReaT epochs: {great_epochs}")  # Corrected from great_bs to great_epochs for print
        print(f"  - DistSampl epochs: {dist_sampl_epochs}")
        print(f"  - TabDiff epochs: {tabdiff_epochs}")

    synthetic_data_cache = {}
    print("\n\n== GENERATING SYNTHETIC DATA FOR ALL MODELS ==\n")

    for name, dataset_info in datasets.items():
        print(f"\n{'=' * 50}\nProcessing dataset: {name}\n{'=' * 50}")
        X, y = load_dataset(name, dataset_info)
        if X is None or y is None:
            continue

        try:
            if use_cv:
                synthetic_data_cache[name] = {'folds': [], 'models': {}}
                for fold_idx in range(n_folds):
                    X_train, X_test, y_train, y_test = preprocess_data(
                        X, y, name=name, discretize=discretize, cv_fold=fold_idx, n_folds=n_folds
                    )
                    train_data = pd.concat([X_train, y_train], axis=1)
                    print(f"Fold {fold_idx + 1}/{n_folds} data loaded and preprocessed. "
                          f"Training data shape: {train_data.shape}")
                    synthetic_data_cache[name]['folds'].append({
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        'train_data': train_data
                    })
                # For CV, model training happens per fold if logic demands,
                # but typically generative models are trained once on the full training set or equivalent.
                # The current structure trains generative models once *before* CV folds for evaluation.
                # We will use the first fold's X_train, y_train for generative model training if use_cv.
                # This assumes generative model training isn't part of the CV folds themselves.
                # If generative models should be retrained per fold, this section needs restructuring.
                # For now, let's assume training on data derived from the first available split.
                if synthetic_data_cache[name]['folds']:
                    X_train_for_gen_model = synthetic_data_cache[name]['folds'][0]['X_train']
                    y_train_for_gen_model = synthetic_data_cache[name]['folds'][0]['y_train']
                    train_data_for_gen_model = synthetic_data_cache[name]['folds'][0]['train_data']
                else:
                    print(f"Warning: No folds available for {name} to train generative models. Skipping.")
                    continue

            else:  # Traditional train-test split
                X_train, X_test, y_train, y_test = preprocess_data(X, y, name=name, discretize=discretize)
                train_data = pd.concat([X_train, y_train], axis=1)
                print(f"Data loaded and preprocessed with discretize={discretize}. "
                      f"Training data shape: {train_data.shape}")
                synthetic_data_cache[name] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'train_data': train_data, 'models': {}
                }
                X_train_for_gen_model = X_train
                y_train_for_gen_model = y_train
                train_data_for_gen_model = train_data

        except Exception as e:
            print(f"Error preprocessing data for {name}: {e}")
            continue

        n_samples = len(train_data_for_gen_model)  # Use the determined training data

        # --- GANBLR++ ---
        if 'ganblr++' in models and PGMPY_AVAILABLE:
            print("\n-- Generating synthetic data for GANBLR++ --")
            model_train_time = 0.0
            try:
                start_model_time = time.time()
                hc = HillClimbSearch(train_data_for_gen_model)
                best_model_hc = hc.estimate(scoring_method=BIC(train_data_for_gen_model))
                bn_hc = train_bn(best_model_hc, train_data_for_gen_model)
                model_train_time = time.time() - start_model_time
                if bn_hc:
                    hc_bic = structure_score(bn_hc, train_data_for_gen_model, scoring_method="bic-cg")
                    hc_synthetic = generate_bn_synthetic_data(bn_hc, train_data_for_gen_model, n_samples=n_samples)
                    if hc_synthetic is not None:
                        synthetic_data_cache[name]['models']['ganblr++'] = {
                            'data': hc_synthetic, 'bic': hc_bic, 'train_time': model_train_time
                        }
                        save_synthetic_data(hc_synthetic, "ganblrpp", name)  # Assuming save_synthetic_data exists
                else:
                    synthetic_data_cache[name]['models']['ganblr++'] = {'data': None, 'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating GANBLR++ synthetic data: {e}")
                synthetic_data_cache[name]['models']['ganblr++'] = {'data': None, 'train_time': model_train_time}

        # --- GANBLR ---
        if 'ganblr' in models and PGMPY_AVAILABLE:
            print("\n-- Generating synthetic data for GANBLR --")
            model_train_time = 0.0
            try:
                start_model_time = time.time()
                ts = TreeSearch(train_data_for_gen_model)
                best_model_ts = ts.estimate()
                bn_ts = train_bn(best_model_ts, train_data_for_gen_model)
                model_train_time = time.time() - start_model_time
                if bn_ts:
                    ts_bic = structure_score(bn_ts, train_data_for_gen_model, scoring_method="bic-cg")
                    ts_synthetic = generate_bn_synthetic_data(bn_ts, train_data_for_gen_model, n_samples=n_samples)
                    if ts_synthetic is not None:
                        synthetic_data_cache[name]['models']['ganblr'] = {
                            'data': ts_synthetic, 'bic': ts_bic, 'train_time': model_train_time
                        }
                        save_synthetic_data(ts_synthetic, "ganblr", name)
                else:
                    synthetic_data_cache[name]['models']['ganblr'] = {'data': None, 'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating GANBLR synthetic data: {e}")
                synthetic_data_cache[name]['models']['ganblr'] = {'data': None, 'train_time': model_train_time}

        # --- CTGAN ---
        if 'ctgan' in models and CTGAN_AVAILABLE:
            print("\n-- Generating synthetic data for CTGAN --")
            model_train_time = 0.0
            ctgan_model = None
            try:
                # MODIFICATION START: Use X_train_for_gen_model, y_train_for_gen_model
                ctgan_train_data_feed = pd.concat([X_train_for_gen_model, y_train_for_gen_model], axis=1)
                discrete_columns = [col for col in ctgan_train_data_feed.columns if
                                    len(np.unique(ctgan_train_data_feed[col])) < 10]

                start_model_time = time.time()
                ctgan_model = train_ctgan(
                    ctgan_train_data_feed,
                    discrete_columns=discrete_columns,
                    epochs=ctgan_epochs,
                    batch_size=min(500, len(ctgan_train_data_feed))
                )
                model_train_time = time.time() - start_model_time
                # MODIFICATION END

                if ctgan_model:
                    ctgan_synthetic = generate_ctgan_synthetic_data(ctgan_model, ctgan_train_data_feed,
                                                                    n_samples=n_samples)
                    if ctgan_synthetic is not None:
                        synthetic_data_cache[name]['models']['ctgan'] = {
                            'data': ctgan_synthetic, 'train_time': model_train_time  # MODIFICATION: Store actual time
                        }
                        save_synthetic_data(ctgan_synthetic, "ctgan", name)
                else:  # Handle case where model training failed
                    synthetic_data_cache[name]['models']['ctgan'] = {'data': None, 'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating CTGAN synthetic data: {e}")
                synthetic_data_cache[name]['models']['ctgan'] = {'data': None, 'train_time': model_train_time}

        # --- CTABGAN ---
        if 'ctabgan' in models and CTABGAN_AVAILABLE:
            print("\n-- Generating synthetic data for CTABGAN --")
            model_train_time = 0.0
            ctabgan_model = None
            try:
                # MODIFICATION START: Use X_train_for_gen_model, y_train_for_gen_model
                categorical_columns = [col for col in X_train_for_gen_model.columns if
                                       X_train_for_gen_model[col].dtype == 'object' or len(
                                           np.unique(X_train_for_gen_model[col])) < 10]

                start_model_time = time.time()
                ctabgan_model = train_ctabgan(
                    X_train_for_gen_model,
                    y_train_for_gen_model,
                    categorical_columns=categorical_columns,
                    epochs=ctgan_epochs
                )
                model_train_time = time.time() - start_model_time
                # MODIFICATION END

                if ctabgan_model:
                    # Prepare train data for synthetic data generation
                    ctabgan_train_data_feed = pd.concat([X_train_for_gen_model, y_train_for_gen_model], axis=1)
                    ctabgan_synthetic = generate_ctabgan_synthetic_data(ctabgan_model, ctabgan_train_data_feed,
                                                                        n_samples=n_samples)
                    if ctabgan_synthetic is not None:
                        synthetic_data_cache[name]['models']['ctabgan'] = {
                            'data': ctabgan_synthetic, 'train_time': model_train_time  # MODIFICATION: Store actual time
                        }
                        save_synthetic_data(ctabgan_synthetic, "ctabgan", name)
                else:  # Handle case where model training failed
                    synthetic_data_cache[name]['models']['ctabgan'] = {'data': None, 'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating CTABGAN synthetic data: {e}")
                synthetic_data_cache[name]['models']['ctabgan'] = {'data': None, 'train_time': model_train_time}

        # --- Naive Bayes (NB) ---
        if 'nb' in models:
            print("\n-- Generating synthetic data for Naive Bayes --")
            model_train_time = 0.0
            nb_model = None
            try:
                # MODIFICATION START: Use X_train_for_gen_model, y_train_for_gen_model
                start_model_time = time.time()
                nb_model = train_naive_bayes(X_train_for_gen_model, y_train_for_gen_model)
                model_train_time = time.time() - start_model_time
                # MODIFICATION END

                if nb_model:
                    nb_bic = get_gaussianNB_bic_score(nb_model,
                                                      train_data_for_gen_model)  # Use train_data_for_gen_model for BIC
                    nb_synthetic = generate_nb_synthetic_data(nb_model, X_train_for_gen_model, y_train_for_gen_model,
                                                              n_samples=n_samples)
                    if nb_synthetic is not None:
                        synthetic_data_cache[name]['models']['nb'] = {
                            'data': nb_synthetic, 'bic': nb_bic, 'train_time': model_train_time
                            # MODIFICATION: Store actual time
                        }
                        save_synthetic_data(nb_synthetic, "nb", name)
                else:  # Handle case where model training failed
                    synthetic_data_cache[name]['models']['nb'] = {'data': None, 'bic': None,
                                                                  'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating Naive Bayes synthetic data: {e}")
                synthetic_data_cache[name]['models']['nb'] = {'data': None, 'bic': None, 'train_time': model_train_time}

        # --- RLiG ---
        if 'rlig' in models and RLIG_AVAILABLE:
            print("\n-- Generating synthetic data for RLiG --")
            model_train_time = 0.0
            rlig_model = None
            try:
                # MODIFICATION START: Use X_train_for_gen_model, y_train_for_gen_model
                start_model_time = time.time()
                rlig_model = train_rlig(X_train_for_gen_model, y_train_for_gen_model, episodes=rlig_episodes,
                                        epochs=rlig_epochs)
                model_train_time = time.time() - start_model_time
                # MODIFICATION END

                if rlig_model:
                    rlig_bic = rlig_model.best_score if hasattr(rlig_model, 'best_score') else None
                    # MODIFICATION: generate_rlig_synthetic_data (if exists) or use rlig_model.sample
                    # For consistency, let's assume rlig_model.sample is the generation step
                    rlig_synthetic = rlig_model.sample(n_samples)  # Generate n_samples
                    if isinstance(rlig_synthetic, np.ndarray):  # Ensure it's a DataFrame
                        columns = list(X_train_for_gen_model.columns) + [
                            y_train_for_gen_model.columns[0] if isinstance(y_train_for_gen_model,
                                                                           pd.DataFrame) else y_train_for_gen_model.name or 'target']
                        rlig_synthetic = pd.DataFrame(rlig_synthetic, columns=columns)

                    if rlig_synthetic is not None:
                        synthetic_data_cache[name]['models']['rlig'] = {
                            'data': rlig_synthetic, 'bic': rlig_bic, 'model': rlig_model, 'train_time': model_train_time
                            # MODIFICATION: Store actual time
                        }
                        save_synthetic_data(rlig_synthetic, "rlig", name)
                else:  # Handle case where model training failed
                    synthetic_data_cache[name]['models']['rlig'] = {'data': None, 'bic': None, 'model': None,
                                                                    'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating RLiG synthetic data: {e}")
                synthetic_data_cache[name]['models']['rlig'] = {'data': None, 'bic': None, 'model': None,
                                                                'train_time': model_train_time}

        # --- GReaT ---
        if 'great' in models and GREAT_AVAILABLE:
            print("\n-- Generating synthetic data for GReaT --")
            model_train_time = 0.0
            great_model = None
            try:
                # GReaT might have its own preprocessing needs, ensure X_train_for_gen_model, y_train_for_gen_model are suitable
                # or re-preprocess if GReaT needs specific format (e.g. non-discretized for its internal handling)
                # For this example, we use the already preprocessed X_train_for_gen_model, y_train_for_gen_model.
                # If GReaT needs specific preprocessing, it should be done here.
                # X_train_great, _, y_train_great, _ = preprocess_data(X, y, discretize=DISCRETIZE_FOR_GREAT_TRAINING) # Example

                start_model_time = time.time()
                great_model = train_great(X_train_for_gen_model, y_train_for_gen_model, batch_size=great_bs,
                                          epochs=great_epochs)
                model_train_time = time.time() - start_model_time

                if great_model:
                    great_train_data_feed = pd.concat([X_train_for_gen_model, y_train_for_gen_model], axis=1)
                    great_synthetic = generate_great_synthetic_data(great_model, great_train_data_feed,
                                                                    n_samples=n_samples)
                    if great_synthetic is not None:
                        synthetic_data_cache[name]['models']['great'] = {
                            'data': great_synthetic,
                            'train_time': model_train_time,  # Store actual time
                            # 'X_test' and 'y_test' specific to GReaT if its preprocessing differs significantly for test.
                            # For now, assume common X_test, y_test from the split.
                        }
                        save_synthetic_data(great_synthetic, "great", name)
                else:
                    synthetic_data_cache[name]['models']['great'] = {'data': None, 'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating GReaT synthetic data: {e}")
                synthetic_data_cache[name]['models']['great'] = {'data': None, 'train_time': model_train_time}

        # --- DistSampl ---
        if 'dist_sampl' in models and DIST_SAMPL_AVAILABLE:
            print("\n-- Generating synthetic data for Distribution Sampling --")
            model_train_time = 0.0
            dist_sampl_model = None
            try:
                # Similar to GReaT, ensure X_train_for_gen_model, y_train_for_gen_model are suitable.
                start_model_time = time.time()
                dist_sampl_model = train_dist_sampl(X_train_for_gen_model, y_train_for_gen_model,
                                                    epochs=dist_sampl_epochs, random_seed=seed)
                model_train_time = time.time() - start_model_time

                if dist_sampl_model:
                    dist_sampl_train_data_feed = pd.concat([X_train_for_gen_model, y_train_for_gen_model],
                                                           axis=1)  # For context
                    dist_sampl_synthetic = generate_dist_sampl_synthetic_data(dist_sampl_model,
                                                                              dist_sampl_train_data_feed,
                                                                              n_samples=n_samples)
                    if dist_sampl_synthetic is not None:
                        synthetic_data_cache[name]['models']['dist_sampl'] = {
                            'data': dist_sampl_synthetic,
                            'train_time': model_train_time  # Store actual time
                        }
                        save_synthetic_data(dist_sampl_synthetic, "dist_sampl", name)
                else:
                    synthetic_data_cache[name]['models']['dist_sampl'] = {'data': None, 'train_time': model_train_time}
            except Exception as e:
                print(f"Error generating Distribution Sampling synthetic data: {e}")
                synthetic_data_cache[name]['models']['dist_sampl'] = {'data': None, 'train_time': model_train_time}

        # --- TabDiff ---
        if 'tabdiff' in models and TABDIFF_AVAILABLE:
            print("\n-- Generating synthetic data for TabDiff --")
            # try:
            # Get data and preprocess based on discretization flag for TabDiff
            X_train_tabdiff, X_test_tabdiff, y_train_tabdiff, y_test_tabdiff = (
                preprocess_data(X, y, name=name, discretize=discretize, model_name='tabdiff')
            )
            print("Shape of X_train_tabdiff: ", X_train_tabdiff.head(), X_train_tabdiff.shape)
            print("Shape of X_test_tabdiff: ", X_test_tabdiff.head(), X_test_tabdiff.shape)

            # from _utils import save_to_csv
            # # Save train data
            # save_to_csv(X_train_tabdiff, y_train_tabdiff, f"data/{name}", "train.csv")
            #
            # # Save test data
            # save_to_csv(X_test_tabdiff, y_test_tabdiff, f"data/{name}", "test.csv")
            #
            # # from TabDiff.tabdiff_data_loader import create_tabdiff_dataloader
            # # batch_size = 256
            # # train_loader = create_tabdiff_dataloader(X_train_tabdiff, y_train_tabdiff,
            # #                                          batch_size=batch_size)

            import json
            info_path = f'data/{name}/info.json'
            with open(info_path, 'r') as f:
                info = json.load(f)

            data_dir = f'data/{name}'
            print("Info loaded", info, name, data_dir)
            train_data = TabDiffDataset(name, data_dir, info)
            d_numerical, categories = train_data.d_numerical, train_data.categories
            print("categories", categories)

            batch_size = 256

            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )

            # Train TabDiff model with consistent random seed
            tabdiff_model = train_tabdiff(train_data=train_data, train_loader=train_loader, name=name,
                                          epochs=tabdiff_epochs, random_seed=seed)

            if tabdiff_model:
                # Generate synthetic data
                tabdiff_synthetic = generate_tabdiff_synthetic_data(tabdiff_model, train_data, n_samples=10)

                if tabdiff_synthetic is not None:
                    # Store in cache
                    synthetic_data_cache[name]['models']['tabdiff'] = {
                        'data': tabdiff_synthetic,
                        'X_test': X_test_tabdiff,
                        'y_test': y_test_tabdiff
                    }

                    # Save synthetic data
                    os.makedirs("train_data", exist_ok=True)
                    tabdiff_synthetic.head(1000).to_csv(f"train_data/tabdiff_{name}_synthetic.csv", index=False)
                    print(f"TabDiff synthetic data saved to train_data/tabdiff_{name}_synthetic.csv")
            # except Exception as e:
            #     print(f"Error generating TabDiff synthetic data: {e}")

    # Configure approach based on CV and nested CV options
    if nested_cv:
        print(f"\n\n{'=' * 20} USING {n_folds}-FOLD CROSS-VALIDATION WITHIN {n_rounds} RANDOM SEED ROUNDS {'=' * 20}\n")
    elif use_cv:
        print(f"\n\n{'=' * 20} USING {n_folds}-FOLD CROSS-VALIDATION {'=' * 20}\n")
    else:
        print(f"\n\n{'=' * 20} USING {n_rounds} ROUNDS WITH DIFFERENT RANDOM SEEDS {'=' * 20}\n")

    all_rounds_results = {}

    if nested_cv or use_cv:  # Combined logic for CV paths
        num_outer_loops = n_rounds if nested_cv else 1
        for round_idx in range(num_outer_loops):
            all_rounds_results[round_idx] = {}
            if nested_cv:
                print(f"\n\n{'=' * 20} RANDOM SEED ROUND {round_idx + 1}/{n_rounds} {'=' * 20}\n")
                round_seed = seed + round_idx
                np.random.seed(round_seed)
                # Potentially re-trigger synthetic data generation here if it depends on round_seed for models
                # However, current structure generates synthetic data once.

            for dataset_name_eval in synthetic_data_cache.keys():
                X_orig, y_orig = load_dataset(dataset_name_eval,
                                              datasets[dataset_name_eval])  # Load original data for splitting

                if X_orig is None or y_orig is None:
                    continue

                fold_results_list = []
                for fold_idx_eval in range(n_folds):
                    print(
                        f"\n{'=' * 15} FOLD {fold_idx_eval + 1}/{n_folds} FOR {dataset_name_eval} (Round {round_idx + 1 if nested_cv else 1}) {'=' * 15}\n")
                    # Get specific X_test, y_test for this fold
                    _, X_test_fold, _, y_test_fold = preprocess_data(
                        X_orig, y_orig, name=dataset_name_eval, discretize=discretize, cv_fold=fold_idx_eval,
                        n_folds=n_folds
                    )
                    # evaluate_models_on_fold uses synthetic_data_cache which has models trained once.
                    # It will use X_test_fold, y_test_fold for evaluation.
                    fold_model_results = evaluate_models_on_fold(
                        dataset_name_eval, synthetic_data_cache, X_test_fold, y_test_fold, models
                    )
                    fold_results_list.append(fold_model_results)
                all_rounds_results[round_idx][dataset_name_eval] = average_fold_results(fold_results_list)

    else:  # Traditional approach: Multiple rounds with different random seeds
        for round_idx in range(n_rounds):
            all_rounds_results[round_idx] = {}
            # round_results = {} # This was defined locally, use all_rounds_results[round_idx]

            print(f"\n\n{'=' * 20} RANDOM SEED ROUND {round_idx + 1}/{n_rounds} {'=' * 20}\n")
            round_seed = seed + round_idx
            np.random.seed(round_seed)
            # Note: Synthetic data is already generated. Seed change here affects downstream classifier randomness if any.

            for name_eval in synthetic_data_cache.keys():  # Changed 'name' to 'name_eval' to avoid conflict
                print(f"\n{'=' * 50}\nEvaluating dataset: {name_eval} (Round {round_idx + 1})\n{'=' * 50}")

                cached_data = synthetic_data_cache[name_eval]
                X_test_trad = cached_data['X_test']  # From the initial split
                y_test_trad = cached_data['y_test']  # From the initial split

                model_results_trad = {
                    'metrics': {}, 'times': {}, 'bic_scores': {}, 'dataset_name': name_eval
                }
                models_cache_trad = cached_data['models']

                for model_name_trad, model_cache_trad in models_cache_trad.items():
                    if model_name_trad not in models:
                        continue
                    print(f"\n-- Evaluating {model_name_trad.upper()} synthetic data --")
                    synthetic_data_trad = model_cache_trad.get('data')
                    if synthetic_data_trad is None:
                        print(f"No synthetic data for {model_name_trad}, skipping evaluation.")
                        # Ensure metrics/times are still populated with None or defaults
                        for classifier_name_default in ['LR', 'MLP', 'RF', 'XGB', 'AVG']:
                            metric_key_default = f"{classifier_name_default}_accuracy"
                            if metric_key_default not in model_results_trad['metrics']:
                                model_results_trad['metrics'][metric_key_default] = {}
                            model_results_trad['metrics'][metric_key_default][model_name_trad] = None
                        if 'training_time' not in model_results_trad['times']:
                            model_results_trad['times']['training_time'] = {}
                        model_results_trad['times']['training_time'][model_name_trad] = model_cache_trad.get(
                            'train_time', 0.0)  # Still record train time
                        if 'bic' not in model_results_trad['bic_scores']:
                            model_results_trad['bic_scores']['bic'] = {}
                        model_results_trad['bic_scores']['bic'][model_name_trad] = model_cache_trad.get('bic')
                        continue

                    # MODIFICATION START: Correctly handle time for non-CV path
                    # The 'eval_time' from evaluate_tstr is for downstream classifiers.
                    # We need the generative model's training time.
                    actual_generative_model_train_time = model_cache_trad.get('train_time', 0.0)

                    if 'training_time' not in model_results_trad['times']:
                        model_results_trad['times']['training_time'] = {}
                    model_results_trad['times']['training_time'][model_name_trad] = actual_generative_model_train_time
                    # MODIFICATION END

                    # Handle BIC score
                    if 'bic' in model_cache_trad:
                        if 'bic' not in model_results_trad['bic_scores']:
                            model_results_trad['bic_scores']['bic'] = {}
                        model_results_trad['bic_scores']['bic'][model_name_trad] = model_cache_trad.get('bic')

                    # Special handling for RLiG's built-in evaluate if needed, or standard TSTR
                    if model_name_trad == 'rlig' and 'model' in model_cache_trad and RLIG_AVAILABLE:
                        rlig_model_eval = model_cache_trad['model']
                        if rlig_model_eval:  # Check if model exists
                            y_test_series_trad = y_test_trad.iloc[:, 0] if isinstance(y_test_trad, pd.DataFrame) and \
                                                                           y_test_trad.shape[1] == 1 else y_test_trad
                            try:
                                lr_result = rlig_model_eval.evaluate(X_test_trad, y_test_series_trad, model='lr')
                                mlp_result = rlig_model_eval.evaluate(X_test_trad, y_test_series_trad, model='mlp')
                                rf_result = rlig_model_eval.evaluate(X_test_trad, y_test_series_trad, model='rf')
                                rlig_eval_results = {'LR': lr_result, 'MLP': mlp_result, 'RF': rf_result,
                                                     'AVG': (lr_result + mlp_result + rf_result) / 3}
                                for classifier, acc in rlig_eval_results.items():
                                    metric_key = f"{classifier}_accuracy"  # Ensure key matches evaluate_tstr
                                    if metric_key not in model_results_trad['metrics']:
                                        model_results_trad['metrics'][metric_key] = {}
                                    model_results_trad['metrics'][metric_key][model_name_trad] = acc
                            except Exception as rlig_eval_e:
                                print(f"Error during RLiG built-in evaluation: {rlig_eval_e}")
                                # Populate with None if RLiG eval fails
                                for classifier_name_rl in ['LR', 'MLP', 'RF', 'AVG']:
                                    metric_key_rl = f"{classifier_name_rl}_accuracy"
                                    if metric_key_rl not in model_results_trad['metrics']:
                                        model_results_trad['metrics'][metric_key_rl] = {}
                                    model_results_trad['metrics'][metric_key_rl][model_name_trad] = None
                        else:  # RLiG model was None
                            for classifier_name_rl in ['LR', 'MLP', 'RF', 'AVG']:
                                metric_key_rl = f"{classifier_name_rl}_accuracy"
                                if metric_key_rl not in model_results_trad['metrics']:
                                    model_results_trad['metrics'][metric_key_rl] = {}
                                model_results_trad['metrics'][metric_key_rl][model_name_trad] = None
                    else:
                        # Standard TSTR evaluation for other models
                        test_X_current = X_test_trad
                        test_y_current = y_test_trad
                        if model_name_trad in ['great',
                                               'dist_sampl'] and 'X_test' in model_cache_trad and 'y_test' in model_cache_trad:
                            # This logic for model-specific test data seems less common if preprocessing is unified.
                            # test_X_current = model_cache_trad['X_test']
                            # test_y_current = model_cache_trad['y_test']
                            pass  # Keep using the common X_test_trad, y_test_trad for now

                        print("Before: ", synthetic_data_trad)
                        if name_eval == "letter_recog":
                            print("calling")
                            tstr_eval_results = evaluate_tstr(synthetic_data_trad,
                                                              test_X_current,
                                                              test_y_current, target_col="lettr")
                        else:
                            tstr_eval_results = evaluate_tstr(synthetic_data_trad, test_X_current, test_y_current)
                        for classifier, acc in tstr_eval_results.items():  # evaluate_tstr returns dict like {'LR': acc_lr, ...}
                            metric_key = f"{classifier}_accuracy"  # Construct the key like 'LR_accuracy'
                            if metric_key not in model_results_trad['metrics']:
                                model_results_trad['metrics'][metric_key] = {}
                            model_results_trad['metrics'][metric_key][model_name_trad] = acc
                all_rounds_results[round_idx][name_eval] = model_results_trad

    # Average results across all rounds/folds
    final_results = {}
    iterations_count = n_rounds if (
                nested_cv or not use_cv) else 1  # if use_cv and not nested_cv, it's 1 outer loop for results collection
    if use_cv and not nested_cv:  # For plain CV, results are already averaged by average_fold_results
        final_results = all_rounds_results[0]  # all_rounds_results[0] contains per-dataset averaged fold results
    else:  # For nested_cv or traditional rounds, average across rounds
        for dataset_name_avg in datasets.keys():
            if not any(dataset_name_avg in all_rounds_results[r] for r in all_rounds_results):
                continue  # Skip if dataset wasn't processed in any round

            final_results[dataset_name_avg] = {
                'metrics': {}, 'times': {}, 'bic_scores': {}, 'dataset_name': dataset_name_avg
            }
            metric_accumulators = {}
            time_accumulators = {}
            bic_accumulators = {}
            valid_round_counts = {}  # Per metric/time/bic type

            for round_idx_avg in range(iterations_count):
                if dataset_name_avg not in all_rounds_results.get(round_idx_avg, {}):
                    continue
                round_data = all_rounds_results[round_idx_avg][dataset_name_avg]

                # Accumulate Metrics
                for metric_key, model_metrics in round_data.get('metrics', {}).items():
                    if metric_key not in metric_accumulators: metric_accumulators[metric_key] = {}
                    if metric_key not in valid_round_counts: valid_round_counts[metric_key] = {}
                    for model_n, val in model_metrics.items():
                        if val is not None:
                            metric_accumulators[metric_key][model_n] = metric_accumulators[metric_key].get(model_n,
                                                                                                           0) + val
                            valid_round_counts[metric_key][model_n] = valid_round_counts[metric_key].get(model_n, 0) + 1
                # Accumulate Times
                for time_category, model_times in round_data.get('times',
                                                                 {}).items():  # e.g., time_category is 'training_time'
                    if time_category not in time_accumulators: time_accumulators[time_category] = {}
                    if time_category not in valid_round_counts: valid_round_counts[time_category] = {}
                    for model_n, val in model_times.items():
                        if val is not None:
                            time_accumulators[time_category][model_n] = time_accumulators[time_category].get(model_n,
                                                                                                             0) + val
                            valid_round_counts[time_category][model_n] = valid_round_counts[time_category].get(model_n,
                                                                                                               0) + 1
                # Accumulate BIC
                for bic_category, model_bics in round_data.get('bic_scores', {}).items():  # e.g., bic_category is 'bic'
                    if bic_category not in bic_accumulators: bic_accumulators[bic_category] = {}
                    if bic_category not in valid_round_counts: valid_round_counts[bic_category] = {}
                    for model_n, val in model_bics.items():
                        if val is not None:
                            bic_accumulators[bic_category][model_n] = bic_accumulators[bic_category].get(model_n,
                                                                                                         0) + val
                            valid_round_counts[bic_category][model_n] = valid_round_counts[bic_category].get(model_n,
                                                                                                             0) + 1

            # Calculate Averages
            for m_key, models_data in metric_accumulators.items():
                if m_key not in final_results[dataset_name_avg]['metrics']: final_results[dataset_name_avg]['metrics'][
                    m_key] = {}
                for mdl, total_val in models_data.items():
                    count = valid_round_counts.get(m_key, {}).get(mdl, 1)
                    final_results[dataset_name_avg]['metrics'][m_key][mdl] = total_val / count if count > 0 else None
            for t_cat, models_data in time_accumulators.items():
                if t_cat not in final_results[dataset_name_avg]['times']: final_results[dataset_name_avg]['times'][
                    t_cat] = {}
                for mdl, total_val in models_data.items():
                    count = valid_round_counts.get(t_cat, {}).get(mdl, 1)
                    final_results[dataset_name_avg]['times'][t_cat][mdl] = total_val / count if count > 0 else 0.0
            for b_cat, models_data in bic_accumulators.items():
                if b_cat not in final_results[dataset_name_avg]['bic_scores']:
                    final_results[dataset_name_avg]['bic_scores'][b_cat] = {}
                for mdl, total_val in models_data.items():
                    count = valid_round_counts.get(b_cat, {}).get(mdl, 1)
                    final_results[dataset_name_avg]['bic_scores'][b_cat][mdl] = total_val / count if count > 0 else None

    if nested_cv:
        print(f"\nAveraged results across {n_rounds} random seed rounds with {n_folds}-fold cross-validation in each")
    elif use_cv:
        print(f"\nAveraged results across {n_folds} cross-validation folds")
    else:  # Traditional rounds
        print(f"\nAveraged results across {n_rounds} random seed rounds")
    return final_results


# ============= RESULTS FORMATTING FUNCTIONS =============

def format_results(results):
    """Format the results into DataFrames for easier analysis"""
    # Define expected models and metric types for consistent column ordering
    models = ['ganblr++', 'ganblr', 'ctgan', 'ctabgan', 'nb', 'rlig', 'great', 'dist_sampl']
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

        # Time data extraction from the results structure

        # First, look for directly stored time values (non-nested, like RLIG, NB, etc.)
        # These are stored with uppercase model names as keys, like in eval_tstr_final.py
        for time_key, time_value in data['times'].items():
            if not isinstance(time_value, dict):
                # This is for models where time is stored directly with model name as key
                times_dict[time_key] = time_value

        # Now check for any values in the nested 'training_time' structure
        if 'training_time' in data['times'] and isinstance(data['times']['training_time'], dict):
            for model_name, time_value in data['times']['training_time'].items():
                # Format model name
                model_upper = model_name.upper()
                if model_name.lower() == 'ganblr++':
                    model_upper = 'GANBLR++'

                # If the model doesn't already have a time entry, add it
                if model_upper not in times_dict:
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
    print("Models supported: RLiG, GANBLR, GANBLR++, CTGAN, Naive Bayes, GReaT, Distribution Sampling")
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
        default=['rlig', 'ganblr', 'ganblr++', 'ctgan', 'ctabgan', 'nb', 'great', 'dist_sampl', 'tabdiff'],
        help="List of models to evaluate. Options: rlig, ganblr, ganblr++, ctgan, ctabgan, nb, great, dist_sampl, tab_diff"
    )

    """PokerHand: 158
    NSL-KDD: data/nsl-kdd/KDDTrain+_20Percent.arff
    Connect-4: 26
    Credit: 27 --> default/credit: 350
    Adult: 2
    Chess: 22
    letter_recog: 59
    Magic: 159
    Nursery: 76
    Room_Occupancy: 864
    Car: 19
    Maternal_Health: 863
    Loan & Credit: from local directory
    """

    # Dataset selection arguments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=['Adult', 'Car', 'Chess', 'Connect-4', 'Default',
                 'letter_recog', 'Magic', 'Maternal_Health', 'Nursery', 'Rice',
                 'Room_Occupancy'],
        help="List of dataset names to evaluate"
    )

    # Add UCI dataset IDs
    parser.add_argument(
        "--uci_ids",
        type=int,
        nargs="+",
        default=[
            2,                # Adult
            19,               # Car   --- generate nan in tabdiff --> skip
            22,               # Chess
            26,               # Connect-4
            350,              # Default/Credit
            59,               # Letter Recognition
            159,              # Magic
            863,              # Maternal Health
            76,               # Nursery
            545,              # Rice
            864               # Room Occupancy
        ],
        # default=[545, 101, 158, 26, 27, 2, 22, 59, 159, 76, 864, 19, 863],  # Default: Rice and TicTacToe
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
        "--dist_sampl_epochs",
        type=int,
        default=50,
        help="Number of epochs for Distribution Sampling training"
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
        dist_sampl_epochs=args.dist_sampl_epochs,
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
