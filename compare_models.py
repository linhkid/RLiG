"""
Model Comparison Framework for RLiG

This script compares RLiG with baseline models:
- HillClimbSearch (pgmpy)
- TreeSearch (pgmpy)
- GaussianNB (scikit-learn)
- NOTEARS (causalnex)

The comparison focuses on:
1. Classification accuracy
2. BIC score
3. Training time
"""

import time
import gc
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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# pgmpy imports
from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.metrics import structure_score

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
    print("continuous columns: ", continuous_cols)
    print("categorical columns: ", categorical_cols)
    
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

# Model training and evaluation functions
def train_bn(model, data):
    """Train a Bayesian Network model"""
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(model.nodes())
    bn.add_edges_from(model.edges())
    print("Bayesian Network: ", bn)
    
    # Fit model using Maximum Likelihood Estimation
    try:
        bn.fit(data, estimator=MaximumLikelihoodEstimator)
        return bn
    except Exception as e:
        print(f"Error fitting Bayesian Network: {e}")
        return None

def evaluate_bn_model(model, X_test, y_test):
    """Evaluate Bayesian Network classification performance"""
    if model is None:
        return None
    
    infer = VariableElimination(model)
    target_var = y_test.columns[0]  # Assumes only one target variable
    model_nodes = set(model.nodes())
    y_pred = []
    
    # Make predictions for each test instance
    for index, row in X_test.iterrows():
        evidence = {k: v for k, v in row.to_dict().items() if k in model_nodes}
        try:
            q = infer.map_query(variables=[target_var], evidence=evidence, show_progress=False)
            y_pred.append(q[target_var])
        except Exception as e:
            y_pred.append(None)
    
    # Filter out None predictions
    y_test_classes = y_test[target_var].unique()
    y_pred = [pred if pred in y_test_classes else None for pred in y_pred]
    
    valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
    if not valid_indices:
        return 0.0  # No valid predictions
    
    y_pred = [y_pred[i] for i in valid_indices]
    y_test_filtered = y_test.iloc[valid_indices].values.ravel()
    
    # Calculate accuracy
    return accuracy_score(y_test_filtered, y_pred)

def evaluate_naive_bayes(model, X_test, y_test):
    """Evaluate Naive Bayes model"""
    try:
        y_pred = model.predict(X_test)
        y_test_values = y_test.values.ravel()
        return accuracy_score(y_test_values, y_pred)
    except Exception as e:
        print(f"Error evaluating Naive Bayes: {e}")
        return None

def evaluate_rlig(model, X_test, y_test):
    """Evaluate RLiG model using its built-in evaluate method"""
    try:
        lr_result = model.evaluate(X_test, y_test, model='lr')
        mlp_result = model.evaluate(X_test, y_test, model='mlp')
        rf_result = model.evaluate(X_test, y_test, model='rf')
        return {
            'LR': lr_result,
            'MLP': mlp_result,
            'RF': rf_result
        }
    except Exception as e:
        print(f"Error evaluating RLiG: {e}")
        return None

def get_bic_score(model, data):
    """Calculate the BIC score for a model"""
    try:
        return structure_score(model, data, scoring_method="bic-cg")
    except Exception as e:
        print(f"Error calculating BIC score: {e}")
        return None

def get_gaussianNB_bic_score(model, data):
    """Calculate the BIC score for a model"""
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

# Main comparison function
def compare_models(datasets):
    """Compare all models on the given datasets"""
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
            train_data.to_csv(f"{name}_train_data.csv")
            print(f"Data loaded and preprocessed. "
                  f"Total number of data: {X.shape}. "
                  f"Training data shape: {train_data.shape}")
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            continue
            
        # Initialize results dictionary for this dataset
        model_results = {
            'metrics': {},
            'times': {},
            'bic_scores': {}
        }
        
        # Hill Climbing Search
        print("\nTraining Hill Climbing model...")
        start_time = time.time()

        hc = HillClimbSearch(train_data)
        best_model_hc = hc.estimate(scoring_method=BIC(train_data))
        bn_hc = train_bn(best_model_hc, train_data)
        hc_time = time.time() - start_time

        # Evaluate
        hc_acc = evaluate_bn_model(bn_hc, X_test, y_test)
        hc_bic = get_bic_score(bn_hc, train_data) if bn_hc else None

        model_results['metrics']['Hill Climbing'] = hc_acc
        model_results['times']['Hill Climbing'] = hc_time
        model_results['bic_scores']['Hill Climbing'] = hc_bic

        print(f"Hill Climbing - Accuracy: {hc_acc:.4f}, Time: {hc_time:.2f}s, BIC: {hc_bic}")

        try:
            hc = HillClimbSearch(train_data)
            best_model_hc = hc.estimate(scoring_method=BIC(train_data))
            bn_hc = train_bn(best_model_hc, train_data)
            hc_time = time.time() - start_time
            
            # Evaluate
            hc_acc = evaluate_bn_model(bn_hc, X_test, y_test)
            hc_bic = get_bic_score(bn_hc, train_data) if bn_hc else None
            
            model_results['metrics']['Hill Climbing'] = hc_acc
            model_results['times']['Hill Climbing'] = hc_time
            model_results['bic_scores']['Hill Climbing'] = hc_bic
            
            print(f"Hill Climbing - Accuracy: {hc_acc:.4f}, Time: {hc_time:.2f}s, BIC: {hc_bic}")
        except Exception as e:
            print(f"Error with Hill Climbing: {e}")
        
        # Tree Search
        print("\nTraining Tree Search model...")
        start_time = time.time()
        try:
            ts = TreeSearch(train_data)
            best_model_ts = ts.estimate()
            bn_ts = train_bn(best_model_ts, train_data)
            ts_time = time.time() - start_time
            
            # Evaluate
            ts_acc = evaluate_bn_model(bn_ts, X_test, y_test)
            ts_bic = get_bic_score(bn_ts, train_data) if bn_ts else None
            
            model_results['metrics']['Tree Search'] = ts_acc
            model_results['times']['Tree Search'] = ts_time
            model_results['bic_scores']['Tree Search'] = ts_bic
            
            print(f"Tree Search - Accuracy: {ts_acc:.4f}, Time: {ts_time:.2f}s, BIC: {ts_bic}")
        except Exception as e:
            print(f"Error with Tree Search: {e}")
        
        # Naive Bayes
        print("\nTraining Naive Bayes model...")
        start_time = time.time()
        try:
            nb = GaussianNB()
            nb.fit(X_train, y_train.values.ravel())
            nb_time = time.time() - start_time
            
            # Evaluate
            nb_acc = evaluate_naive_bayes(nb, X_test, y_test)
            nb_bic = get_gaussianNB_bic_score(nb, train_data) if nb else None
            
            model_results['metrics']['Naive Bayes'] = nb_acc
            model_results['times']['Naive Bayes'] = nb_time
            model_results['bic_scores']['Naive'] = nb_bic
            
            print(f"Naive Bayes - Accuracy: {nb_acc:.4f}, Time: {nb_time:.2f}s, BIC: {nb_bic}")
        except Exception as e:
            print(f"Error with Naive Bayes: {e}")
        
        # NOTEARS
        if CAUSALNEX_AVAILABLE:
            print("\nTraining NOTEARS model...")
            start_time = time.time()
            try:
                sm = from_pandas(train_data, w_threshold=0.8)
                bn_nt = train_bn(sm, train_data)
                nt_time = time.time() - start_time
                
                # Evaluate
                nt_acc = evaluate_bn_model(bn_nt, X_test, y_test)
                nt_bic = get_bic_score(bn_nt, train_data) if bn_nt else None
                
                model_results['metrics']['NOTEARS'] = nt_acc
                model_results['times']['NOTEARS'] = nt_time
                model_results['bic_scores']['NOTEARS'] = nt_bic
                
                print(f"NOTEARS - Accuracy: {nt_acc:.4f}, Time: {nt_time:.2f}s, BIC: {nt_bic}")
            except Exception as e:
                print(f"Error with NOTEARS: {e}")
        
        # RLiG
        if RLIG_AVAILABLE:
            print("\nTraining RLiG model...")
            start_time = time.time()

            try:
                # Make a copy of the original data for RLiG
                # RLiG requires the original format data
                rlig_model = RLiG()

                # Use reduced parameters for testing to speed things up
                episodes = 2  # Reduced from 10
                epochs = 5  # Reduced from 10

                # Ensure the data is properly formatted
                if isinstance(y_test, pd.DataFrame):
                    y_series = y_test.iloc[:, 0] if y_test.shape[1] == 1 else y
                else:
                    y_series = y_test

                # Add safeguard for CPD normalization issues
                print(f"Starting RLiG fit with {episodes} episodes and {epochs} epochs...")
                rlig_model.fit(X_train, y_train, episodes=episodes, gan=1, k=0, epochs=epochs, n=1)
                rlig_time = time.time() - start_time

                # Evaluate using RLiG's own evaluation with error handling
                try:
                    print("Evaluating RLiG model...")
                    rlig_results = evaluate_rlig(rlig_model, X_test, y_series)
                    rlig_bic = rlig_model.best_score if hasattr(rlig_model, 'best_score') else None

                    if rlig_results:
                        for clf_name, acc in rlig_results.items():
                            model_results['metrics'][f'RLiG-{clf_name}'] = acc

                    model_results['times']['RLiG'] = rlig_time
                    model_results['bic_scores']['RLiG'] = rlig_bic

                    print(f"RLiG - Results: {rlig_results}, Time: {rlig_time:.2f}s, BIC: {rlig_bic}")
                except Exception as e:
                    print(f"Error evaluating RLiG model: {e}")

                # Save the network structure visualization
                try:
                    # if hasattr(rlig_model, 'bayesian_network'):
                    model_graphviz = rlig_model.bayesian_network.to_graphviz()
                    model_graphviz.draw(f"rlig_{name}_network.png", prog="dot")
                    print(f"RLiG network visualization saved to rlig_{name}_network.png")
                except Exception as e:
                    print(f"Error saving RLiG network visualization: {e}")

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

if __name__ == "__main__":
    # Define datasets to evaluate
    # Format: {name: dataset_id_or_path}
    datasets = {
        'Rice': 545,        # UCI ID for Rice dataset
        'TicTacToe': 101,   # UCI ID for Tic-tac-toe dataset
        # Add more datasets as needed
        'car': 'data/car.arff',
        'adult': 'data/adult.arff'
    }
    
    # Run the comparison
    results = compare_models(datasets)
    
    # Format and display results
    formatted_results = format_results(results)
    
    print("\n\n=== ACCURACY RESULTS ===")
    print(formatted_results['accuracy'])
    print("\n\n=== TIME RESULTS (seconds) ===")
    print(formatted_results['time'])
    print("\n\n=== BIC SCORE RESULTS ===")
    print(formatted_results['bic'])
    
    # Save results to CSV
    try:
        formatted_results['accuracy'].to_csv('accuracy_results.csv')
        formatted_results['time'].to_csv('time_results.csv')
        formatted_results['bic'].to_csv('bic_results.csv')
        print("\nResults saved to CSV files.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")