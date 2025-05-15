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


def preprocess_data(X, y, discretize=True, model_name=None):
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
                ('discretizer', KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='quantile'))
            ])
            print("Using discretization with quantile binning (7 bins)")
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
    
    # Split data
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
        
        # Workaround for memory issues: use smaller dataframe
        if len(X_train) > 1000:
            print(f"Large dataset detected ({len(X_train)} rows). Using 1000-row sample for CTGAN training.")
            X_train = X_train.sample(1000, random_state=42)
        
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
        # For small datasets only to avoid timeouts
        if len(X_train) > 1000:
            print(f"Dataset too large for CTABGAN in this evaluation framework. Sampling 1000 rows with stratification.")
            # Sample the data to avoid memory issues, use stratified sampling to ensure all classes have enough samples
            
            # Check class distribution
            if isinstance(y_train, pd.DataFrame):
                y_values = y_train.iloc[:, 0]
            else:
                y_values = y_train
                
            class_counts = y_values.value_counts()
            min_class_count = class_counts.min()
            
            # If the minimum class count is very low, we need to ensure we have enough samples of each class
            if min_class_count < 5:
                print(f"Warning: Minimum class count is very low ({min_class_count}). Using stratified sampling.")
                # Use stratified sampling
                try:
                    from sklearn.model_selection import train_test_split
                    # Keep all instances of rare classes and sample from common classes
                    rare_classes = class_counts[class_counts < 5].index.tolist()
                    
                    # If there are too many rare classes, sample at least 2 from each class
                    if len(rare_classes) > 20:  # Arbitrary threshold
                        # Sample with replacement to ensure at least 2 samples per class
                        sample_size = min(1000, len(X_train))
                        indices = []
                        
                        # First, ensure at least 2 samples from each class
                        for class_val in class_counts.index:
                            class_indices = y_values[y_values == class_val].index.tolist()
                            # Sample with replacement if needed
                            if len(class_indices) == 1:
                                indices.extend([class_indices[0], class_indices[0]])  # Duplicate the single instance
                            else:
                                # Take at least 2 samples
                                sample_indices = np.random.choice(class_indices, size=min(2, len(class_indices)), replace=False)
                                indices.extend(sample_indices)
                        
                        # Fill the rest with random samples
                        if len(indices) < sample_size:
                            remaining = sample_size - len(indices)
                            # Exclude indices already selected
                            available_indices = [i for i in range(len(X_train)) if i not in indices]
                            if available_indices:
                                extra_indices = np.random.choice(available_indices, 
                                                                size=min(remaining, len(available_indices)), 
                                                                replace=False)
                                indices.extend(extra_indices)
                        
                        # Trim if we have too many
                        indices = indices[:sample_size]
                        
                    else:
                        # Keep all rare classes and sample from others
                        rare_indices = y_values[y_values.isin(rare_classes)].index.tolist()
                        
                        # Determine how many common class samples we can include
                        remaining_slots = 1000 - len(rare_indices)
                        
                        if remaining_slots > 0:
                            # Get indices of common classes
                            common_indices = y_values[~y_values.isin(rare_classes)].index.tolist()
                            
                            # Sample from common classes
                            sampled_common = np.random.choice(common_indices, 
                                                            size=min(remaining_slots, len(common_indices)), 
                                                            replace=False)
                            
                            # Combine rare and sampled common indices
                            indices = rare_indices + sampled_common.tolist()
                        else:
                            # If we have too many rare classes, sample from them
                            indices = np.random.choice(rare_indices, size=1000, replace=False)
                    
                    # Get the sampled data
                    X_train = X_train.iloc[indices]
                    if isinstance(y_train, pd.DataFrame):
                        y_train = y_train.iloc[indices]
                    else:
                        y_train = y_train.iloc[indices]
                        
                    # Verify we have at least 2 samples of each class in the result
                    if isinstance(y_train, pd.DataFrame):
                        result_counts = y_train.iloc[:, 0].value_counts()
                    else:
                        result_counts = y_train.value_counts()
                        
                    min_result_count = result_counts.min()
                    if min_result_count < 2:
                        print(f"Warning: After sampling, minimum class count is still {min_result_count}.")
                        print("Some classes may be excluded during training.")
                        
                except Exception as e:
                    print(f"Error during stratified sampling: {e}")
                    # Fall back to simple random sampling
                    indices = np.random.choice(len(X_train), 1000, replace=False)
                    X_train = X_train.iloc[indices]
                    if isinstance(y_train, pd.DataFrame):
                        y_train = y_train.iloc[indices]
                    else:
                        y_train = y_train.iloc[indices]
            else:
                # If class distribution is ok, use simple random sampling
                indices = np.random.choice(len(X_train), 1000, replace=False)
                X_train = X_train.iloc[indices]
                if isinstance(y_train, pd.DataFrame):
                    y_train = y_train.iloc[indices]
                else:
                    y_train = y_train.iloc[indices]
        
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
                pipeline.fit(syn_X, syn_y)
                
                # Test on real data
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results[name] = acc
                print(f"{name} TSTR accuracy: {acc:.4f}")
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
            
            model_results['times']['RLiG'] = rlig_time
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
        model_results['times']['TS'] = ts_time
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
            model_results['times']['GANBLR'] = ts_time
            
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
                
                # Save synthetic data sample
                ts_synthetic.head(1000).to_csv(f"train_data/ganblr_{dataset_name}_synthetic.csv", index=False)
                print(f"GANBLR synthetic data sample saved to train_data/ganblr_{dataset_name}_synthetic.csv")
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
        model_results['times']['HC'] = hc_time
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
            model_results['times']['GANBLR++'] = hc_time
            
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
                
                # Save synthetic data sample
                hc_synthetic.head(1000).to_csv(f"train_data/ganblrpp_{dataset_name}_synthetic.csv", index=False)
                print(f"GANBLR++ synthetic data sample saved to train_data/ganblrpp_{dataset_name}_synthetic.csv")
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
        model_results['times']['CTGAN'] = ctgan_time
        
        print(f"CTGAN - Time: {ctgan_time:.2f}s")
        
        # Save synthetic data sample
        dataset_name = model_results.get('dataset_name', 'unknown')
        try:
            # Create directory if it doesn't exist
            os.makedirs("train_data", exist_ok=True)
            
            # Save synthetic data sample
            ctgan_synthetic.head(1000).to_csv(f"train_data/ctgan_{dataset_name}_synthetic.csv", index=False)
            print(f"CTGAN synthetic data sample saved to train_data/ctgan_{dataset_name}_synthetic.csv")
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
        model_results['times']['CTABGAN'] = ctabgan_time
        
        print(f"CTABGAN - Time: {ctabgan_time:.2f}s")
        
        # Save synthetic data sample
        dataset_name = model_results.get('dataset_name', 'unknown')
        try:
            # Create directory if it doesn't exist
            os.makedirs("train_data", exist_ok=True)
            
            # Save synthetic data sample
            ctabgan_synthetic.head(1000).to_csv(f"train_data/ctabgan_{dataset_name}_synthetic.csv", index=False)
            print(f"CTABGAN synthetic data sample saved to train_data/ctabgan_{dataset_name}_synthetic.csv")
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
        model_results['times']['NB'] = nb_time
        model_results['bic_scores']['NB'] = get_gaussianNB_bic_score(nb, train_data) if nb else None
        
        print(f"Naive Bayes - Time: {nb_time:.2f}s")
        
        # Save synthetic data sample
        dataset_name = model_results.get('dataset_name', 'unknown')
        try:
            # Create directory if it doesn't exist
            os.makedirs("train_data", exist_ok=True)
            
            # Save synthetic data sample
            nb_synthetic.head(1000).to_csv(f"train_data/nb_{dataset_name}_synthetic.csv", index=False)
            print(f"Naive Bayes synthetic data sample saved to train_data/nb_{dataset_name}_synthetic.csv")
        except Exception as e:
            print(f"Error saving Naive Bayes synthetic data: {e}")
    except Exception as e:
        print(f"Error with Naive Bayes: {e}")


# ============= MAIN COMPARISON FUNCTION =============

def compare_models_tstr(datasets, models=None, n_rounds=3, seed=42, rlig_episodes=2, rlig_epochs=5, 
                  ctgan_epochs=50, great_bs=1, great_epochs=5, tabsyn_epochs=50, verbose=False, discretize=True):
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
            X_train, X_test, y_train, y_test = preprocess_data(X, y, discretize=args.discretize)
            train_data = pd.concat([X_train, y_train], axis=1)
            print(f"Data loaded and preprocessed with discretize={args.discretize}. Training data shape: {train_data.shape}")
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            continue
            
        # Initialize synthetic data cache for this dataset
        synthetic_data_cache[name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train, 
            'y_test': y_test,
            'train_data': train_data,
            'models': {}
        }
        
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
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        hc_synthetic.head(1000).to_csv(f"train_data/ganblrpp_{name}_synthetic.csv", index=False)
                        print(f"GANBLR++ synthetic data saved to train_data/ganblrpp_{name}_synthetic.csv")
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
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        ts_synthetic.head(1000).to_csv(f"train_data/ganblr_{name}_synthetic.csv", index=False)
                        print(f"GANBLR synthetic data saved to train_data/ganblr_{name}_synthetic.csv")
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
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        ctgan_synthetic.head(1000).to_csv(f"train_data/ctgan_{name}_synthetic.csv", index=False)
                        print(f"CTGAN synthetic data saved to train_data/ctgan_{name}_synthetic.csv")
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
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        ctabgan_synthetic.head(1000).to_csv(f"train_data/ctabgan_{name}_synthetic.csv", index=False)
                        print(f"CTABGAN synthetic data saved to train_data/ctabgan_{name}_synthetic.csv")
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
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        nb_synthetic.head(1000).to_csv(f"train_data/nb_{name}_synthetic.csv", index=False)
                        print(f"Naive Bayes synthetic data saved to train_data/nb_{name}_synthetic.csv")
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

                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        great_synthetic.head(1000).to_csv(f"train_data/great_{name}_synthetic.csv", index=False)
                        print(f"GReaT synthetic data saved to train_data/great_{name}_synthetic.csv")
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
                        
                        # Save synthetic data
                        os.makedirs("train_data", exist_ok=True)
                        tabsyn_synthetic.head(1000).to_csv(f"train_data/tabsyn_{name}_synthetic.csv", index=False)
                        print(f"TabSyn synthetic data saved to train_data/tabsyn_{name}_synthetic.csv")
            except Exception as e:
                print(f"Error generating TabSyn synthetic data: {e}")

    # Now run multiple rounds of cross-validation on the generated synthetic data
    for round_idx in range(n_rounds):
        print(f"\n\n{'='*20} CROSS-VALIDATION ROUND {round_idx+1}/{n_rounds} {'='*20}\n")
        
        # Set a different seed for each round but in a deterministic way
        round_seed = seed + round_idx
        np.random.seed(round_seed)
        
        round_results = {}
        
        # For each dataset, evaluate models using the pre-generated synthetic data
        for name in synthetic_data_cache.keys():
            print(f"\n{'='*50}\nEvaluating dataset: {name} (Round {round_idx+1})\n{'='*50}")
            
            # Get cached data
            cached_data = synthetic_data_cache[name]
            X_test = cached_data['X_test'] 
            y_test = cached_data['y_test']
            
            # Initialize results for this dataset and round
            model_results = {
                'metrics': {},
                'times': {},
                'bic_scores': {},
                'dataset_name': name
            }
            
            # Evaluate each model's synthetic data
            for model_name, model_cache in cached_data['models'].items():
                print(f"\n-- Evaluating {model_name.upper()} synthetic data --")
                
                # Get synthetic data and BIC score
                synthetic_data = model_cache['data']
                
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
                    
                    model_results['times']['RLiG'] = time.time() - start_time
                    
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
                    model_results['times'][model_name.upper()] = eval_time
                    
                    # Store BIC score if available
                    if 'bic' in model_cache and model_cache['bic'] is not None:
                        model_results['bic_scores'][model_name.upper()] = model_cache['bic']
            
            # Store results for this dataset and round
            round_results[name] = model_results
        
        # Store this round's results
        all_rounds_results[round_idx] = round_results
    
    # Average results across all rounds
    final_results = {}
    
    for dataset_name in datasets.keys():
        # Initialize dataset results
        final_results[dataset_name] = {
            'metrics': {},
            'times': {},
            'bic_scores': {},
            'dataset_name': dataset_name
        }
        
        # Count valid rounds for this dataset
        valid_rounds = 0
        
        # Combine metrics from all rounds
        for round_idx in range(n_rounds):
            if dataset_name not in all_rounds_results[round_idx]:
                continue
                
            round_data = all_rounds_results[round_idx][dataset_name]
            valid_rounds += 1
            
            # Accumulate metrics
            for metric_key, metric_value in round_data['metrics'].items():
                # Skip None values
                if metric_value is None:
                    continue
                    
                if metric_key not in final_results[dataset_name]['metrics']:
                    final_results[dataset_name]['metrics'][metric_key] = 0
                final_results[dataset_name]['metrics'][metric_key] += metric_value
            
            # Accumulate times
            for time_key, time_value in round_data['times'].items():
                # Skip None values
                if time_value is None:
                    continue
                    
                if time_key not in final_results[dataset_name]['times']:
                    final_results[dataset_name]['times'][time_key] = 0
                final_results[dataset_name]['times'][time_key] += time_value
            
            # Accumulate BIC scores
            for bic_key, bic_value in round_data['bic_scores'].items():
                # Skip None values
                if bic_value is None:
                    continue
                    
                if bic_key not in final_results[dataset_name]['bic_scores']:
                    final_results[dataset_name]['bic_scores'][bic_key] = 0
                final_results[dataset_name]['bic_scores'][bic_key] += bic_value
        
        # Compute averages
        if valid_rounds > 0:
            # Average metrics
            for metric_key in final_results[dataset_name]['metrics'].keys():
                final_results[dataset_name]['metrics'][metric_key] /= valid_rounds
            
            # Average times
            for time_key in final_results[dataset_name]['times'].keys():
                final_results[dataset_name]['times'][time_key] /= valid_rounds
            
            # Average BIC scores
            for bic_key in final_results[dataset_name]['bic_scores'].keys():
                final_results[dataset_name]['bic_scores'][bic_key] /= valid_rounds
    
    print(f"\nAveraged results across {n_rounds} rounds")
    return final_results


# ============= RESULTS FORMATTING FUNCTIONS =============

def format_results(results):
    """Format the results into DataFrames for easier analysis"""
    accuracy_results = {}
    time_results = {}
    bic_results = {}
    
    for dataset, data in results.items():
        accuracy_results[dataset] = data['metrics']
        time_results[dataset] = data['times']
        bic_results[dataset] = data['bic_scores']
    
    # Convert to DataFrames
    accuracy_df = pd.DataFrame.from_dict(accuracy_results, orient='index')
    time_df = pd.DataFrame.from_dict(time_results, orient='index')
    bic_df = pd.DataFrame.from_dict(bic_results, orient='index')
    
    return {
        'accuracy': accuracy_df,
        'time': time_df,
        'bic': bic_df
    }


def save_results_to_csv(results_dict, prefix="tstr"):
    """Save results to CSV files"""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    for result_type, df in results_dict.items():
        filename = f"results/{prefix}_{result_type}_results.csv"
        df.to_csv(filename)
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
        discretize=args.discretize
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
        # Set the prefix based on discretization, if not already customized
        output_prefix = args.output_prefix
        if output_prefix == "disc_tstr" and not args.discretize:
            output_prefix = "raw_tstr"  # Change default prefix if no discretization
        
        save_results_to_csv(formatted_results, prefix=output_prefix)
        print(f"\nResults saved to CSV files in results directory with prefix '{output_prefix}'.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")