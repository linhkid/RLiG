"""
TSTR (Train on Synthetic, Test on Real) Evaluation Framework

This script implements a proper TSTR evaluation for generative models as described in the RLiG paper.
Models evaluated:
- RLiG (primary model from the paper)
- GANBLR with Structure Learning (referenced as baseline)
- CTGAN (Conditional Tabular GAN)
- Basic Bayesian Network approaches for comparison

The TSTR methodology:
1. Train a generative model on real data
2. Generate synthetic data from the trained model 
3. Train classification models (LR, MLP, RF) on the synthetic data
4. Test these classification models on real test data
5. Measure accuracy (how well models trained on synthetic data perform on real data)
"""

import time
import gc
import os
import warnings
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pgmpy').setLevel(logging.ERROR)

# scipy imports
from scipy.io.arff import loadarff

# scikit-learn imports
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# pgmpy imports
try:
    from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.inference import VariableElimination
    from pgmpy.metrics import structure_score
    PGMPY_AVAILABLE = True
except ImportError:
    print("pgmpy not available. Bayesian Network models will be skipped.")
    PGMPY_AVAILABLE = False

# causalnex import
try:
    from causalnex.structure import StructureModel
    from causalnex.structure.notears import from_pandas
    CAUSALNEX_AVAILABLE = True
except ImportError:
    print("CausalNex not available. NOTEARS will be skipped.")
    CAUSALNEX_AVAILABLE = False

# RLiG import
try:
    from ganblr.models import RLiG
    RLIG_AVAILABLE = True
except ImportError:
    print("RLiG not available. Will be skipped.")
    RLIG_AVAILABLE = False

# We're using RLiG only from ganblr.models
# For GANBLR-style models, we'll use pgmpy with different structure learning methods
# just like in the original tstr_evaluation.py

# CTGAN import
try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except ImportError:
    print("CTGAN not available. CTGAN model will be skipped.")
    CTGAN_AVAILABLE = False

# Dataset handling
try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    print("ucimlrepo not available. Will use local datasets.")
    UCI_AVAILABLE = False


# Read the ARFF file using scipy's loadarff function
def read_arff_file(file_path):
    data, meta = loadarff(file_path)
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    # Convert byte strings to regular strings (if needed)
    for col in df.columns:
        if df[col].dtype == object:  # Object type typically indicates byte strings from ARFF
            df[col] = df[col].str.decode('utf-8')

    return df, meta


# Data preprocessing functions
def label_encode_cols(X, cols):
    """Label encode categorical columns"""
    X_encoded = X.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le
    return X_encoded, encoders


def preprocess_data(X, y):
    """Preprocess data: discretize continuous variables and encode categoricals"""
    # Identify column types
    continuous_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    print("Continuous columns: ", continuous_cols)
    print("Categorical columns: ", categorical_cols)
    
    # Create transformation pipeline
    transformers = []
    if len(continuous_cols) > 0:
        continuous_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
        ])
        transformers.append(('num', continuous_transformer, continuous_cols))
    
    # Handle categorical columns
    if len(categorical_cols) > 0:
        X, encoders = label_encode_cols(X, categorical_cols)

    # Apply transformations
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=continuous_cols.tolist() + categorical_cols.tolist())

    # Handle target variable
    if y.dtypes[0] == 'object':
        label_encoder = LabelEncoder()
        y_transformed = pd.DataFrame(label_encoder.fit_transform(y.values.ravel()), columns=y.columns)
    else:
        y_transformed = y
    
    # Split data
    return train_test_split(X_transformed_df, y_transformed, test_size=0.2, random_state=42, stratify=y)


# Model training functions
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


# CTGAN Model
def train_ctgan(X_train, discrete_columns=None, epochs=100, batch_size=500):
    """Train a CTGAN model"""
    if not CTGAN_AVAILABLE:
        return None
        
    try:
        # Convert X_train to DataFrame if it's not already
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        
        # Identify categorical columns if not provided
        if discrete_columns is None:
            discrete_columns = []
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or len(np.unique(X_train[col])) < 10:
                    discrete_columns.append(col)
        
        # Initialize and train CTGAN model
        ctgan_model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        print(f"Training CTGAN with {epochs} epochs, categorical columns: {discrete_columns}")
        ctgan_model.fit(X_train, discrete_columns)
        return ctgan_model
    except Exception as e:
        print(f"Error training CTGAN model: {e}")
        return None


def generate_ctgan_synthetic_data(ctgan_model, train_data, n_samples=None):
    """Generate synthetic data from CTGAN model"""
    if not CTGAN_AVAILABLE or ctgan_model is None:
        return None
        
    if n_samples is None:
        n_samples = len(train_data)
    
    try:
        # Sample from the CTGAN model
        synthetic_data = ctgan_model.sample(n_samples)
        print(f"Generated {len(synthetic_data)} synthetic samples from CTGAN")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from CTGAN: {e}")
        return None


# Synthetic data generation functions
def generate_bn_synthetic_data(bn_model, train_data, n_samples=None):
    """
    Generate synthetic data from a Bayesian Network model
    
    By default, generates the same number of samples as the training data,
    following the methodology in the paper
    """
    if not PGMPY_AVAILABLE or bn_model is None:
        return None
    
    if n_samples is None:
        n_samples = len(train_data)
    
    try:
        # Sample from the Bayesian Network
        sampler = BayesianModelSampling(bn_model)
        synthetic_data = sampler.forward_sample(size=n_samples)
        
        # Ensure synthetic data has the same column order as train_data
        # This is crucial for consistent evaluation
        col_order = list(train_data.columns)
        synthetic_data = synthetic_data[col_order]
        
        print(f"Generated {len(synthetic_data)} synthetic samples from Bayesian Network")
        return synthetic_data
    except Exception as e:
        print(f"Error generating synthetic data from BN: {e}")
        return None


def generate_nb_synthetic_data(nb_model, X_train, y_train, n_samples=None):
    """
    Generate synthetic data from a Naive Bayes model
    
    By default, generates the same number of samples as the training data,
    following the methodology in the paper
    """
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


# Note: We're not using the GANBLR class directly
# Instead, we're implementing GANBLR/GANBLR++ functionality using structure learning methods
# For GANBLR: use Tree Search (KDB-like structure)
# For GANBLR++: use Hill Climbing (more flexible structure)


# Train RLiG model
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


# TSTR evaluation function
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
        return {'LR': None, 'MLP': None, 'RF': None, 'AVG': None}
    
    try:
        # Split synthetic data into features and target
        if target_col in synthetic_data.columns:
            syn_X = synthetic_data.drop(target_col, axis=1)
            syn_y = synthetic_data[target_col]
        else:
            # If target column isn't found, assume last column is target
            syn_X = synthetic_data.iloc[:, :-1]
            syn_y = synthetic_data.iloc[:, -1]
            
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
        
        results = {}
        
        # Get feature categories for one-hot encoding
        categories = [np.unique(np.concatenate([syn_X[col].unique(), X_test[col].unique()])) for col in X_test.columns]
        
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
        return {'LR': None, 'MLP': None, 'RF': None, 'AVG': None}


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


# Main TSTR comparison function
def compare_models_tstr(datasets, models=None):
    """
    Compare generative models using TSTR methodology as described in the paper
    
    Parameters:
    -----------
    datasets : dict
        Dictionary mapping dataset names to dataset sources
    models : list or None
        List of models to evaluate. If None, evaluate all available models.
        Options: 'rlig', 'ganblr', 'ganblr++', 'ctgan', 'nb'
    """
    # Default models to evaluate
    if models is None:
        models = ['rlig', 'ganblr', 'ganblr++', 'ctgan', 'nb']
    
    results = {}
    
    for name, dataset_info in datasets.items():
        print(f"\n{'='*50}\nProcessing dataset: {name}\n{'='*50}")
        
        # Load dataset
        if isinstance(dataset_info, int) and UCI_AVAILABLE:
            try:
                data = fetch_ucirepo(id=dataset_info)
                X = data.data.features
                # Change the name of columns to avoid "-" to parsing error
                X.columns = [col.replace('-', '_') for col in X.columns]
                y = data.data.targets
                # Change the name of y dataframe to avoid duplicate "class" keyword
                y.columns = ["target"]
            except Exception as e:
                print(f"Error loading UCI dataset {name} (id={dataset_info}): {e}")
                continue
        elif isinstance(dataset_info, str):
            try:
                if dataset_info.endswith(".csv"):
                    df = pd.read_csv(dataset_info)
                    X = df.iloc[:, :-1]
                    # Change the name of columns to avoid "-" to parsing error
                    X.columns = [col.replace('-', '_') for col in X.columns]
                    y = df.iloc[:, -1:]
                    # Change the name of y dataframe to avoid duplicate "class" keyword
                    y.columns = ["target"]
                else:
                    # Read arff file
                    df, meta = read_arff_file(dataset_info)
                    # Encode categorical variables
                    X = df.drop('class', axis=1)
                    # Change the name of columns to avoid "-" to parsing error
                    X.columns = [col.replace('-', '_') for col in X.columns]
                    y = df.iloc[:, -1:]
                    # Change the name of y dataframe to avoid duplicate "class" keyword
                    y.columns = ["target"]
            except Exception as e:
                print(f"Error loading dataset from file {dataset_info}: {e}")
                continue
        else:
            print(f"Invalid dataset specification for {name}")
            continue
            
        # Preprocess data
        try:
            X_train, X_test, y_train, y_test = preprocess_data(X, y)
            train_data = pd.concat([X_train, y_train], axis=1)
            print(f"Data loaded and preprocessed. Training data shape: {train_data.shape}")
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            continue
            
        # Initialize results dictionary for this dataset
        model_results = {
            'metrics': {},
            'times': {},
            'bic_scores': {}
        }
        
        # Set number of synthetic samples to match training data size
        n_samples = len(train_data)
        
        # Structure learning models for GANBLR implementations
        if (('ganblr' in models or 'ganblr++' in models) and PGMPY_AVAILABLE):
            # Hill Climbing Search for structure learning (GANBLR++)
            if 'ganblr++' in models:
                print("\n--------------------------------------------------")
                print("RUNNING HILL CLIMBING FOR GANBLR++ STRUCTURE")
                print("--------------------------------------------------")
                start_time = time.time()
                try:
                    hc = HillClimbSearch(train_data)
                    best_model_hc = hc.estimate(scoring_method=BIC(train_data))
                    bn_hc = train_bn(best_model_hc, train_data)
                    hc_time = time.time() - start_time
                    
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
                    except Exception as e:
                        print(f"Error evaluating GANBLR++ model: {e}")
                    
                except Exception as e:
                    print(f"Error with Hill Climbing: {e}")
            
            # Tree Search for structure learning (GANBLR)
            if 'ganblr' in models:
                print("\n--------------------------------------------------")
                print("RUNNING TREE SEARCH FOR GANBLR STRUCTURE")
                print("--------------------------------------------------")
                start_time = time.time()
                try:
                    ts = TreeSearch(train_data)
                    best_model_ts = ts.estimate()
                    bn_ts = train_bn(best_model_ts, train_data)
                    ts_time = time.time() - start_time
                    
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
                    except Exception as e:
                        print(f"Error evaluating GANBLR model: {e}")
                    
                except Exception as e:
                    print(f"Error with Tree Search: {e}")
                
        # CTGAN
        if 'ctgan' in models and CTGAN_AVAILABLE:
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
                    epochs=50,  # Reduced for testing
                    batch_size=min(500, len(ctgan_train_data))  # Adjust batch size for small datasets
                )
                ctgan_time = time.time() - start_time
                
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
            except Exception as e:
                print(f"Error evaluating CTGAN model: {e}")
                
        
        # 4. Naive Bayes as a simple baseline
        if 'nb' in models:
            print("\n--------------------------------------------------")
            print("EVALUATING NAIVE BAYES")
            print("--------------------------------------------------")
            start_time = time.time()
            try:
                nb = train_naive_bayes(X_train, y_train)
                nb_time = time.time() - start_time
                
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
            except Exception as e:
                print(f"Error with Naive Bayes: {e}")
        
        # 5. RLiG (primary model from the paper)
        if 'rlig' in models and RLIG_AVAILABLE:
            print("\n--------------------------------------------------")
            print("EVALUATING RLiG")
            print("--------------------------------------------------")
            start_time = time.time()
            try:
                # Initialize and train RLiG model
                rlig_model = train_rlig(X_train, y_train, episodes=2, epochs=5)
                rlig_time = time.time() - start_time
                
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
                    
                    # Store individual results
                    rlig_results = {
                        'LR': lr_result,
                        'MLP': mlp_result,
                        'RF': rf_result,
                        'AVG': (lr_result + mlp_result + rf_result) / 3
                    }
                    
                    for model_name, acc in rlig_results.items():
                        model_results['metrics'][f'RLiG-{model_name}'] = acc
                    
                    model_results['times']['RLiG'] = rlig_time
                    model_results['bic_scores']['RLiG'] = rlig_model.best_score if hasattr(rlig_model, 'best_score') else None
                    
                    print(f"RLiG TSTR results: {rlig_results}")
                    print(f"RLiG - Time: {rlig_time:.2f}s")
                    
                    # Save the network structure visualization
                    try:
                        model_graphviz = rlig_model.bayesian_network.to_graphviz()
                        model_graphviz.draw(f"rlig_{name}_network.png", prog="dot")
                        print(f"RLiG network visualization saved to rlig_{name}_network.png")
                    except Exception as e:
                        print(f"Error saving RLiG network visualization: {e}")
                except Exception as e:
                    print(f"Error evaluating RLiG model: {e}")
            except Exception as e:
                print(f"Error with RLiG: {e}")
            
            # Clean up to prevent memory issues
            gc.collect()
        
        # Store results for this dataset
        results[name] = model_results
    
    return results


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


if __name__ == "__main__":
    # Define datasets to evaluate
    datasets = {
        'Rice': 545,  # UCI ID for Rice dataset
        'TicTacToe': 101,  # UCI ID for Tic-tac-toe dataset
        'car': 'data/car.arff',
        'letter': 'data/letter-recog.arff',
        'adult': 'data/adult.arff',
        'magic': 'data/magic.arff',
        'poker-hand': 'data/poker-hand.arff',
        'chess': 'data/chess.arff',
        'nursery': 'data/nursery.arff',
    }
    
    # Run the TSTR comparison with all datasets and models
    # Include all models in the evaluation
    models_to_run = ['rlig', 'ctgan', 'nb', 'ganblr', 'ganblr++']
    results = compare_models_tstr(datasets, models=models_to_run)
    
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
        save_results_to_csv(formatted_results, prefix="tstr")
        print("\nResults saved to CSV files in results directory.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")