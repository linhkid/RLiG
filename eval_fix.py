"""
TSTR (Train on Synthetic, Test on Real) Evaluation Framework
Simplified Dataset Loading and Integrated Model Training from User's Script.
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
from scipy.io.arff import loadarff as scipy_loadarff # Renamed
import urllib.request
import json # For saving arguments

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss # log_loss for BIC with GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Used in user's evaluate_tstr
from sklearn.exceptions import ConvergenceWarning

# --- UCI ML Repo Fetcher ---
try:
    from ucimlrepo import fetch_ucirepo
    UCIMLREPO_AVAILABLE = True
    print("ucimlrepo library found.")
except ImportError:
    print("ucimlrepo library not found. Please install (`pip install ucimlrepo`). UCI datasets via ID will be skipped.")
    UCIMLREPO_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.preprocessing._discretization')
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="BayesianModel has been renamed to BayesianNetwork.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Set node_states as a dict or CPDs for all nodes", category=UserWarning)
logging.getLogger('pgmpy').setLevel(logging.ERROR) # pgmpy can be verbose

# --- Model Specific Imports (from user's script) ---
# These availability flags are set based on imports from the user's script content.
# Ensure these libraries are actually installed in your environment.

PGMPY_AVAILABLE = False
try:
    from pgmpy.estimators import HillClimbSearch, BIC, TreeSearch, MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork # User script used this
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.metrics import structure_score
    PGMPY_AVAILABLE = True
    print("PGMpy available.")
except ImportError:
    print("pgmpy not available. Bayesian Network models (GANBLR, GANBLR++, NB-pgmpy) will be skipped.")

RLIG_AVAILABLE = False
try:
    from ganblr.models import RLiG # From user's script structure
    RLIG_AVAILABLE = True
    print("RLiG available.")
except ImportError:
    print("RLiG not available. Will be skipped.")

CTGAN_AVAILABLE = False
try:
    from ctgan import CTGAN # User script uses this simple form
    CTGAN_AVAILABLE = True
    print("CTGAN available.")
except ImportError:
    print("CTGAN not available. CTGAN model will be skipped.")

CTABGAN_AVAILABLE = False
try:
    # Assuming ctabgan is in a specific path relative to the script, as in user's original script
    # This might need adjustment based on your actual project structure.
    # For a cleaner setup, install ctabgan as a package.
    sys_path_modified_ctabgan = False
    try:
        from ctabgan.model.ctabgan import CTABGAN # User script specific path
        CTABGAN_AVAILABLE = True
    except ImportError:
        # Try adding local path if needed, similar to user's original attempt
        # This is generally not recommended for robust scripts.
        pass # Keeping it simple for now.
    if CTABGAN_AVAILABLE: print("CTABGAN available.")
    else: print("CTABGAN not available. Will be skipped.")
except Exception as e_ctab_imp:
     print(f"CTABGAN import issue: {e_ctab_imp}")


GREAT_AVAILABLE = False
try:
    from be_great import GReaT # User script uses this
    GREAT_AVAILABLE = True
    print("GReaT available.")
except ImportError:
    print("GReaT is not available. Will be skipped.")

DIST_SAMPL_AVAILABLE = False
try:
    # Assuming distsampl is in a specific path as per user's script
    import sys
    dist_sampl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distsampl')
    if dist_sampl_path not in sys.path: sys.path.append(dist_sampl_path)
    from dist_sampling import DistSampling # from distsampl.dist_sampling
    DIST_SAMPL_AVAILABLE = True
    print("Distribution Sampling available.")
except ImportError as e:
    print(f"Distribution Sampling not available. Will be skipped. Error: {e}")

TABDIFF_AVAILABLE = False
try:
    # Assuming TabDiff is in a specific path as per user's script
    import sys
    tabdiff_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TabDiff')
    if tabdiff_path not in sys.path: sys.path.append(tabdiff_path)
    from tabdiff_module import run_tabdiff # User's script has this
    # Note: `run_tabdiff` seems like a main script runner, not a model training function.
    # Using it directly as `train_tabdiff` might be incorrect. This needs to be a proper
    # training function that returns a model, and a corresponding generate function.
    # For now, it will be called as is, but likely needs refactoring for TabDiff.
    TABDIFF_AVAILABLE = True
    print("TabDiff available (integration might need review).")
except ImportError as e:
    print(f"TabDiff is not available. Will be skipped. Error: {e}")


XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available.")
except ImportError:
    print("XGBoost not available. XGBoost classifier will be skipped in TSTR.")


# --- Simplified Dataset Configurations ---
DEFAULT_UCI_CONFIG = {
    'pokerhand':    {'id': 158, 'name': 'Poker Hand',            'target_column': 'CLASS'},
    'connect4':     {'id': 26,  'name': 'Connect-4 Opening',     'target_column': 'class'},
    'german_credit':{'id': 144, 'name': 'Statlog German Credit', 'target_column': 'class'},
    'adult':        {'id': 2,   'name': 'Adult Census Income',   'target_column': 'income'},
    'chess_krkp':   {'id': 22,  'name': 'Chess KRvKP',           'target_column': 'class'},
    'letter':       {'id': 59,  'name': 'Letter Recognition',    'target_column': 'lettr'},
    'magic':        {'id': 159, 'name': 'MAGIC Gamma Telescope', 'target_column': 'class'},
    'nursery':      {'id': 76,  'name': 'Nursery',               'target_column': 'class'},
    'occupancy':    {'id': 325, 'name': 'Occupancy Detection',   'target_column': 'Occupancy'}, # Changed ID to 325
    'car':          {'id': 19,  'name': 'Car Evaluation',        'target_column': 'class'},
    'maternal_health':{'id':863,'name': 'Maternal Health Risk',  'target_column': 'RiskLevel'},
}

DEFAULT_LOCAL_CONFIG = {
    'loan': {
        'name': 'Loan Approval (Local)',
        'path': 'data/loan_approval_dataset.csv',
        'data_type': 'csv_header',
        'target_column': 'loan_status',
        'id_columns_to_drop': ['loan_id'],
    },
    'kdd': {
        'name': 'NSL-KDD Train+ (Local)',
        'path': 'data/nsl-kdd/Full Data/KDDTrain+.arff',
        'data_type': 'arff',
        'target_column': 'class', # May need 'target_column_inferred' logic during load
    }
}

# Global verbose (can be set by args)
verbose_global = False

# --- Helper Functions (Adapted from user script and previous versions) ---
def read_arff_custom(file_path, dataset_name_for_error="Unknown"): # From user script
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data, meta = scipy_loadarff(f)
        df = pd.DataFrame(data)
        for col in df.select_dtypes([object]):
            try: df[col] = df[col].str.decode('utf-8').str.strip()
            except AttributeError: pass
            df[col] = df[col].replace('', pd.NA)
        df.attrs['meta'] = meta
        return df
    except Exception as e:
        print(f"Error reading ARFF file '{file_path}' for '{dataset_name_for_error}': {e}")
        return None

def load_local_dataset(dataset_key, config): # Adapted
    print(f"Loading local dataset: {config['name']} (Key: {dataset_key})")
    data_type = config['data_type']; target_col_cfg = config['target_column']
    id_cols_drop = config.get('id_columns_to_drop', []); missing_sym = config.get('missing_values_symbol', None)
    try:
        file_path = config['path']
        if not os.path.exists(file_path): print(f"ERR: Local file not found: {file_path}"); return None, None, config
        df = None
        if data_type == 'arff':
            df = read_arff_custom(file_path, config['name'])
            if df is not None and target_col_cfg not in df.columns:
                inferred_target = None
                if 'kdd' in dataset_key.lower() and len(df.columns) > 1: inferred_target = df.columns[-2]
                elif df.shape[1] > 0: inferred_target = df.columns[-1]
                if inferred_target and inferred_target in df.columns:
                    print(f"WARN: Target '{target_col_cfg}' not by name. Using '{inferred_target}' for ARFF '{config['name']}'.")
                    config['target_column_inferred'] = inferred_target
                else: print(f"ERR: Cannot ID target for ARFF {config['name']}."); return None, None, config
        elif data_type == 'csv_header':
            params = {'skipinitialspace': True};_ = params.update({'na_values': missing_sym}) if missing_sym else None
            df = pd.read_csv(file_path, **params)
        else: print(f"ERR: Unknown local type '{data_type}' for '{config['name']}'."); return None, None, config

        if df is None: print(f"ERR: Failed to load DF for '{config['name']}'."); return None, None, config
        if id_cols_drop: df = df.drop(columns=[c for c in id_cols_drop if c in df.columns], errors='ignore')
        for col in df.select_dtypes(['object', 'string']):
            df[col] = df[col].str.strip()
            if missing_sym: df[col] = df[col].replace(str(missing_sym).strip(), pd.NA)

        final_target = config.get('target_column_inferred', target_col_cfg)
        if final_target not in df.columns: print(f"ERR: Target '{final_target}' not in '{config['name']}'. Cols: {df.columns.tolist()}"); return None,None,config
        X = df.drop(columns=[final_target]); y = df[[final_target]]
        print(f"Local '{config['name']}' loaded. X:{X.shape}, y:{y.shape}, Target:'{final_target}'")
        return X, y, config
    except Exception as e: print(f"ERR loading local '{config['name']}': {e}"); return None, None, config

def fetch_and_prepare_uci_dataset(dataset_key, config): # Adapted
    if not UCIMLREPO_AVAILABLE: print(f"SKIP: ucimlrepo NA for '{config['name']}'."); return None, None, config
    print(f"Fetching UCI: {config['name']} (ID: {config['id']})")
    try:
        repo = fetch_ucirepo(id=config['id']); X = repo.data.features; y_df = repo.data.targets
        if y_df is None or y_df.empty: print(f"ERR: Target NA for UCI '{config['name']}'."); return None, None, config

        cfg_target = config['target_column']
        if cfg_target in y_df.columns: y = y_df[[cfg_target]]
        elif y_df.shape[1] == 1:
            orig_name = y_df.columns[0]; y = y_df.rename(columns={orig_name: cfg_target})
            print(f"Renamed fetched target '{orig_name}' to '{cfg_target}' for '{config['name']}'.")
            config['target_column_inferred'] = cfg_target
        else: print(f"ERR: Target '{cfg_target}' not in fetched {y_df.columns.tolist()} for {config['name']}."); return None,None,config

        for col in X.select_dtypes(['object', 'string']): X[col] = X[col].str.strip()
        for col in y.select_dtypes(['object', 'string']): y[col] = y[col].str.strip()
        print(f"UCI '{config['name']}' fetched. X:{X.shape}, y:{y.shape}, Target:'{y.columns[0]}'")
        return X, y, config
    except Exception as e: print(f"ERR fetching UCI '{config['name']}': {e}"); return None, None, config

def preprocess_data(X_orig, y_orig_df, target_column_name_actual, discretize=True, n_bins=7,
                    bin_strategy='quantile', test_size=0.2, random_state=42,
                    cv_fold=None, n_folds=None, model_name_for_disc=None): # model_name_for_disc from user script
    """Uses preprocessing logic similar to the user's provided script."""
    # This function adapts the user's preprocess_data logic.
    # It returns: X_train_eval, X_test_eval, y_train_eval_series, y_test_eval_series, df_gen_model_train_full
    # where _eval versions are OHEd/scaled for downstream classifiers,
    # and df_gen_model_train_full is for generative models (features label-encoded/discretized, target label-encoded).

    X = X_orig.copy()
    if not isinstance(y_orig_df, pd.DataFrame) or target_column_name_actual not in y_orig_df.columns:
        raise ValueError(f"y_orig_df expects DataFrame with column '{target_column_name_actual}'. Has: {y_orig_df.columns if isinstance(y_orig_df,pd.DataFrame) else 'NotDF'}")
    y_series_original = y_orig_df[target_column_name_actual].copy()

    # --- Imputation (from user script) ---
    if X.isnull().any().any():
        if verbose_global: print("Imputing missing values in features...")
        for col in X.select_dtypes(include=['object', 'category']).columns: X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Missing")
        for col in X.select_dtypes(include=['number']).columns: X[col] = X[col].fillna(X[col].median())
    if y_series_original.isnull().any():
        if verbose_global: print("Imputing missing values in target...")
        if pd.api.types.is_object_dtype(y_series_original) or pd.api.types.is_categorical_dtype(y_series_original):
            y_series_original = y_series_original.fillna(y_series_original.mode()[0] if not y_series_original.mode().empty else "Missing_T")
        else: y_series_original = y_series_original.fillna(y_series_original.median())

    # --- Target Encoding ---
    le_target = LabelEncoder()
    y_encoded_series = pd.Series(le_target.fit_transform(y_series_original), name=target_column_name_actual, index=X.index)

    # --- Feature Processing for Generative Model Training Data (df_for_gen_model_training) ---
    X_for_gen = X.copy()
    gen_continuous_cols = X_for_gen.select_dtypes(include=['number']).columns.tolist()
    gen_categorical_cols = X_for_gen.select_dtypes(include=['object', 'category']).columns.tolist()

    # Discretization (quantile as per simplified request, user script had uniform)
    # User script's preprocess_data had `apply_discretization = discretize` and `model_name` logic.
    # Sticking to global `discretize` flag for now.
    if discretize and gen_continuous_cols:
        if verbose_global: print(f"Discretizing for GEN models (Quantile, {n_bins} bins): {gen_continuous_cols}")
        discretizer_gen = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=bin_strategy, subsample=200000, random_state=random_state)
        try: X_for_gen[gen_continuous_cols] = discretizer_gen.fit_transform(X_for_gen[gen_continuous_cols])
        except ValueError as e_d_gen: print(f"WARN: Gen Discretization failed: {e_d_gen}. Numerics as is.")

    # Label Encoding for categorical features in X_for_gen (user script does this)
    # This is important for pgmpy-based models and some others.
    gen_encoders = {}
    if gen_categorical_cols:
        if verbose_global: print(f"Label Encoding for GEN models: {gen_categorical_cols}")
        for col in gen_categorical_cols:
            le = LabelEncoder(); X_for_gen[col] = le.fit_transform(X_for_gen[col].astype(str)); gen_encoders[col] = le # Ensure string type before LE

    df_for_gen_model_training = X_for_gen.copy()
    df_for_gen_model_training[target_column_name_actual] = y_encoded_series # Add label encoded target


    # --- Feature Processing for Downstream Evaluation Models (X_train_eval, X_test_eval) ---
    # These need scaling for numerical and OHE for categorical.
    X_for_eval_ohe = X.copy() # Start from X before gen model discretization/LE
    eval_num_cols = X_for_eval_ohe.select_dtypes(include=['number']).columns.tolist()
    eval_cat_cols = X_for_eval_ohe.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers_eval = []
    if eval_num_cols: transformers_eval.append(('num', StandardScaler(), eval_num_cols))
    if eval_cat_cols:
        valid_eval_cat = [c for c in eval_cat_cols if X_for_eval_ohe[c].nunique() > 1] # Avoid OHE for single-value cols
        if valid_eval_cat: transformers_eval.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_eval_cat))

    X_processed_for_eval_split = X_for_eval_ohe
    if transformers_eval:
        col_transformer_eval = ColumnTransformer(transformers=transformers_eval, remainder='passthrough', n_jobs=-1)
        try:
            X_processed_np_eval = col_transformer_eval.fit_transform(X_for_eval_ohe)
            X_processed_for_eval_split = pd.DataFrame(X_processed_np_eval, columns=col_transformer_eval.get_feature_names_out(), index=X.index)
        except Exception as e_ohe: # Fallback if get_feature_names_out fails
            print(f"WARN: OHE get_feature_names_out failed: {e_ohe}. Using default num names.")
            X_processed_for_eval_split = pd.DataFrame(X_processed_np_eval, index=X.index)

    # --- Data Splitting ---
    if cv_fold is not None and n_folds is not None: # CV fold split
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        train_idx, test_idx = list(skf.split(X_processed_for_eval_split, y_encoded_series))[cv_fold]
        X_train_eval, X_test_eval = X_processed_for_eval_split.iloc[train_idx], X_processed_for_eval_split.iloc[test_idx]
        y_train_eval_series, y_test_eval_series = y_encoded_series.iloc[train_idx], y_encoded_series.iloc[test_idx]
    else: # Standard train/test split
        X_train_eval, X_test_eval, y_train_eval_series, y_test_eval_series = train_test_split(
            X_processed_for_eval_split, y_encoded_series, test_size=test_size, random_state=random_state, stratify=y_encoded_series
        )
    if verbose_global: print(f"Split shapes: X_train_eval={X_train_eval.shape}, X_test_eval={X_test_eval.shape}")
    return X_train_eval, X_test_eval, y_train_eval_series, y_test_eval_series, df_for_gen_model_training


def get_classifiers(random_state=42): # From user script
    models = {
        'LR': LogisticRegression(max_iter=1000, random_state=random_state), # Increased max_iter
        'MLP': MLPClassifier(max_iter=500, early_stopping=True, random_state=random_state, n_iter_no_change=10),
        'RF': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    }
    if XGBOOST_AVAILABLE:
        try: models['XGB'] = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)
        except TypeError: models['XGB'] = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state, n_jobs=-1) # Older XGB
    return models

def evaluate_tstr(synthetic_data_df_raw, X_real_test_ohe, y_real_test_series,
                  original_X_for_column_info, # Used to guide synthetic data OHE
                  random_state_eval=42, verbose_eval_tstr=False): # Renamed random_state
    """Trains classifiers on synthetic data (after OHE) and evaluates on real OHE'd test data."""
    # This is a critical function. Adapting the user's version carefully.
    # The user's `evaluate_tstr` did OHE internally. This version assumes
    # X_real_test_ohe is already OHE'd by preprocess_data.
    # We need to OHE synthetic_data_df_raw to match X_real_test_ohe.

    results = {}; accuracies = []
    default_clf_keys = ['LR', 'MLP', 'RF', 'AVG']
    if XGBOOST_AVAILABLE: default_clf_keys.insert(3, 'XGB')
    for clf_name_init in default_clf_keys: results[clf_name_init] = None

    if synthetic_data_df_raw is None or synthetic_data_df_raw.empty:
        print("WARN evaluate_tstr: Synthetic data is None or empty."); return results

    target_col_name_syn = y_real_test_series.name # Assumed target name from real data
    if target_col_name_syn not in synthetic_data_df_raw.columns:
        print(f"WARN evaluate_tstr: Target '{target_col_name_syn}' not in synthetic columns {synthetic_data_df_raw.columns.tolist()}. Trying last.")
        if synthetic_data_df_raw.shape[1]>0: target_col_name_syn = synthetic_data_df_raw.columns[-1]
        else: print("ERR evaluate_tstr: Synthetic data has no columns."); return results

    X_synthetic_raw_feat = synthetic_data_df_raw.drop(columns=[target_col_name_syn], errors='ignore')
    y_synthetic_series_target = synthetic_data_df_raw[target_col_name_syn]

    # Preprocess X_synthetic_raw_feat (OHE, Scale) to match X_real_test_ohe
    # Use original_X_for_column_info (which is pre-discretization, pre-OHE real X) to define transformers
    X_syn_eval_pp = X_synthetic_raw_feat.copy()
    # Impute any NaNs in synthetic data (should be rare if models are good)
    for col in X_syn_eval_pp.columns:
        if X_syn_eval_pp[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_syn_eval_pp[col]): X_syn_eval_pp[col] = X_syn_eval_pp[col].fillna(X_syn_eval_pp[col].median())
            else: X_syn_eval_pp[col] = X_syn_eval_pp[col].fillna(X_syn_eval_pp[col].mode()[0] if not X_syn_eval_pp[col].mode().empty else "MissingSyn")

    num_cols_orig = original_X_for_column_info.select_dtypes(include=np.number).columns.tolist()
    cat_cols_orig = original_X_for_column_info.select_dtypes(include=['object','category']).columns.tolist()

    # Ensure the synthetic data only has columns that were in original_X_for_column_info
    valid_cols_in_syn = [c for c in X_syn_eval_pp.columns if c in original_X_for_column_info.columns]
    X_syn_eval_pp = X_syn_eval_pp[valid_cols_in_syn]

    # Filter num_cols_orig and cat_cols_orig to only those present in X_syn_eval_pp
    num_cols_syn_pp = [c for c in num_cols_orig if c in X_syn_eval_pp.columns]
    cat_cols_syn_pp = [c for c in cat_cols_orig if c in X_syn_eval_pp.columns]

    transformers_syn_pp = []
    if num_cols_syn_pp: transformers_syn_pp.append(('num', StandardScaler(), num_cols_syn_pp))
    if cat_cols_syn_pp:
        valid_cat_syn_pp = [c for c in cat_cols_syn_pp if X_syn_eval_pp[c].nunique() > 1]
        if valid_cat_syn_pp: transformers_syn_pp.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_cat_syn_pp))

    X_synthetic_ohe = X_syn_eval_pp # Fallback
    if transformers_syn_pp:
        col_trans_syn = ColumnTransformer(transformers=transformers_syn_pp, remainder='passthrough', n_jobs=-1)
        try:
            # Fit transformer ON THE SYNTHETIC DATA to get its OHE version
            X_synthetic_ohe_np = col_trans_syn.fit_transform(X_syn_eval_pp)
            X_synthetic_ohe = pd.DataFrame(X_synthetic_ohe_np, columns=col_trans_syn.get_feature_names_out(), index=X_syn_eval_pp.index)
        except Exception as e_syn_pp: print(f"WARN evaluate_tstr: Preprocessing synthetic X failed: {e_syn_pp}. Using raw numeric/label encoded if possible.")

    # Align columns of X_synthetic_ohe with X_real_test_ohe
    missing_in_syn = set(X_real_test_ohe.columns) - set(X_synthetic_ohe.columns)
    for c_add_syn in missing_in_syn: X_synthetic_ohe[c_add_syn] = 0
    extra_in_syn = set(X_synthetic_ohe.columns) - set(X_real_test_ohe.columns)
    X_synthetic_ohe = X_synthetic_ohe.drop(columns=list(extra_in_syn), errors='ignore')
    try:
      X_synthetic_ohe = X_synthetic_ohe[X_real_test_ohe.columns] # Ensure order
    except KeyError as e_align_final:
        print(f"CRIT_ERR evaluate_tstr: Final column alignment failed. Syn:{X_synthetic_ohe.shape}, Real:{X_real_test_ohe.shape}. Err:{e_align_final}"); return results


    classifiers_eval = get_classifiers(random_state_eval)
    for name_clf, clf_obj in classifiers_eval.items():
        try:
            clf_obj.fit(X_synthetic_ohe, y_synthetic_series_target)
            y_pred_clf = clf_obj.predict(X_real_test_ohe)
            acc_clf = accuracy_score(y_real_test_series, y_pred_clf)
            results[name_clf] = acc_clf
            if acc_clf is not None: accuracies.append(acc_clf)
        except Exception as e_clf: print(f"ERR evaluate_tstr: Classifier {name_clf} failed: {e_clf}"); results[name_clf] = None

    results['AVG'] = np.mean([a for a in accuracies if a is not None]) if accuracies else None
    if verbose_eval_tstr: print(f"  TSTR Accuracies for this model: {results}")
    return results


# --- Model Training Wrappers (adapted from the original script) ---

# === Bayesian Network Functions ===
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


# === Naive Bayes Functions ===
def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes model"""
    nb = GaussianNB()
    try:
        nb.fit(X_train, y_train.values.ravel())
        return nb
    except Exception as e:
        print(f"Error training Naive Bayes: {e}")
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


# === CTGAN Functions ===
def train_ctgan(X_train, discrete_columns=None, epochs=100, batch_size=500):
    """Train a CTGAN model with compatibility fixes"""
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
        
        # For large datasets, use stratified sampling
        if len(X_train) >= 50000:  # Only sub-sample for large datasets
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
                print(f"Using {len(X_train)} stratified samples for CTGAN training.")
            else:
                # If no target column, use simple random sampling
                sample_size = max(25000, int(0.3 * len(X_train)))
                X_train = X_train.sample(min(sample_size, len(X_train)), random_state=42)
                print(f"Using {len(X_train)} random samples for CTGAN training.")
        
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
        
        # Train CTGAN
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


# === CTABGAN Functions ===
def train_ctabgan(X_train, y_train, categorical_columns=None, epochs=50):
    """Train a CTABGAN model"""
    if not CTABGAN_AVAILABLE:
        return None
        
    try:
        # For large datasets, use stratified sampling
        if len(X_train) >= 50000:  # Only subsample for large datasets
            print(f"Large dataset detected ({len(X_train)} rows). Using stratified sampling for CTABGAN training.")
            
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
        
        ctabgan_model = CTABGAN(
            raw_csv_path=temp_csv_path,
            test_ratio=0.2,  # Keep consistent with the evaluation framework's test split
            categorical_columns=categorical_columns,
            integer_columns=integer_columns,
            problem_type=problem_type,
            epochs=epochs
        )
        
        # Train the model
        ctabgan_model.fit()
        return ctabgan_model
    except Exception as e:
        print(f"Error training CTABGAN model: {e}")
        return None


def generate_ctabgan_synthetic_data(ctabgan_model, train_data, n_samples=None):
    """Generate synthetic data from CTABGAN model"""
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
        return None


# === RLiG Functions ===
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


def generate_rlig_synthetic_data(rlig_model, train_data, n_samples=None):
    """Generate synthetic data from a RLiG model"""
    if not RLIG_AVAILABLE or rlig_model is None:
        return None
        
    if n_samples is None:
        n_samples = len(train_data)
    
    try:
        # Get target column name from train_data
        # Assume the last column is the target
        target_col = train_data.columns[-1]
        feature_cols = train_data.columns[:-1]
        
        # Sample from the RLiG model
        synthetic_data = rlig_model.sample(n_samples)
        
        # Check if synthetic_data is a tuple (X, y) or a DataFrame
        if isinstance(synthetic_data, tuple) and len(synthetic_data) == 2:
            X_synthetic, y_synthetic = synthetic_data
            
            # Convert to DataFrame if they are not already
            if not isinstance(X_synthetic, pd.DataFrame):
                X_synthetic = pd.DataFrame(X_synthetic, columns=feature_cols)
            
            if not isinstance(y_synthetic, pd.DataFrame) and not isinstance(y_synthetic, pd.Series):
                if isinstance(train_data[target_col], pd.Series):
                    y_synthetic = pd.Series(y_synthetic, name=target_col)
                else:
                    y_synthetic = pd.DataFrame(y_synthetic, columns=[target_col])
            
            # Combine X and y
            synthetic_df = pd.concat([X_synthetic, y_synthetic], axis=1)
        else:
            # Assume synthetic_data is already a DataFrame
            synthetic_df = synthetic_data
            
        print(f"Generated {len(synthetic_df)} synthetic samples from RLiG")
        return synthetic_df
    except Exception as e:
        print(f"Error generating synthetic data from RLiG: {e}")
        return None


# === GReaT Functions ===
def train_great(X_train, y_train, batch_size=1, epochs=1):
    """Train a Generation of Realistic Tabular data with transformers model"""
    if not GREAT_AVAILABLE:
        return None

    try:
        # Initialize and train GReaT model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available: {torch.cuda.is_available()}. Using device: {device}")
        
        # Configure GReaT with appropriate parameters and suppress warnings
        great_model = GReaT(
            llm='distilgpt2', 
            batch_size=batch_size, 
            epochs=epochs, 
            fp16=True,
            gradient_accumulation_steps=8,
            metric_for_best_model="accuracy"
        )
        
        # Set pad token explicitly to address the attention mask warning
        if hasattr(great_model, 'tokenizer') and great_model.tokenizer is not None:
            if great_model.tokenizer.pad_token is None:
                great_model.tokenizer.pad_token = great_model.tokenizer.eos_token
                print("Set pad_token to eos_token to fix attention mask warning")

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
        return None


# === Distribution Sampling Functions ===
def train_dist_sampl(X_train, y_train, epochs=50, random_seed=42):
    """Train the Distribution Sampling synthesizer"""
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
        
        # Initialize DistSampling
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
                print(f"Generating batch {i+1}/{num_batches}")
                this_batch_size = min(batch_size, n_samples - i*batch_size)
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
        return None


# === TabDiff Functions (placeholder implementation) ===
def train_tabdiff(train_data, train_loader, name, epochs=5, random_seed=42):
    """
    Train a TabDiff model
    
    Note: This is a placeholder implementation. The actual implementation would require
    a more complex setup with TabDiff's specific data format and training procedure.
    """
    if not TABDIFF_AVAILABLE:
        return None
    
    print("TabDiff training functionality is not yet fully implemented")
    return None


def generate_tabdiff_synthetic_data(tabdiff_model, train_data, n_samples=None):
    """
    Generate synthetic data from TabDiff model
    
    Note: This is a placeholder implementation.
    """
    if not TABDIFF_AVAILABLE or tabdiff_model is None:
        return None
    
    print("TabDiff sampling functionality is not yet fully implemented")
    return None

# --- BIC Score ---
def get_gaussianNB_bic_score(model, data_df_bic): # from user script
    try:
        X_bic = data_df_bic.iloc[:, :-1]; y_bic = data_df_bic.iloc[:, -1] # Assume target is last
        n_samples, n_features = X_bic.shape
        n_classes = len(np.unique(y_bic))
        probs = model.predict_proba(X_bic)
        log_likelihood = -log_loss(y_bic, probs, labels=model.classes_, normalize=False)
        k = n_classes * 2 * n_features + (n_classes - 1) # Params for GaussianNB
        bic = -2 * log_likelihood + k * np.log(n_samples)
        return bic
    except Exception as e: print(f"Error calc BIC for GaussianNB: {e}"); return None

def calculate_bic_score(model_object, train_data_for_bic, model_type_str): # Adapted
    if model_type_str == 'nb_gaussian' and isinstance(model_object, GaussianNB):
        return get_gaussianNB_bic_score(model_object, train_data_for_bic)
    elif model_type_str in ['ganblr', 'ganblr++', 'nb_pgmpy'] and PGMPY_AVAILABLE and isinstance(model_object, DiscreteBayesianNetwork):
        try:
            data_for_scoring = train_data_for_bic[list(model_object.nodes())].copy()
            # pgmpy's BIC needs discrete data, often integer encoded.
            # Assume df_for_gen_model_training (passed as train_data_for_bic) has features
            # already label encoded or discretized ordinally.
            for col in data_for_scoring.columns: # Ensure int type for pgmpy scorer
                 if not pd.api.types.is_integer_dtype(data_for_scoring[col]):
                    try: data_for_scoring[col] = LabelEncoder().fit_transform(data_for_scoring[col].astype(str))
                    except: pass # If conversion fails, pgmpy might handle or error out

            bic_estimator = BicScore(data_for_scoring)
            return bic_estimator.score(model_object)
        except Exception as e: print(f"Could not calculate BIC for pgmpy {model_type_str}: {e}"); return None
    elif hasattr(model_object, 'best_score') and model_type_str == 'rlig': # RLiG stores its score
        return model_object.best_score
    return None


# --- Main Comparison Framework (Adapted from previous full versions) ---
def compare_models_tstr(loaded_datasets_map, models_to_run, n_rounds, current_seed,
                        # Model specific args from main args
                        arg_rlig_episodes, arg_rlig_epochs, arg_ctgan_epochs,
                        arg_great_bs, arg_great_epochs, arg_dist_sampl_epochs, arg_tabdiff_epochs,
                        main_verbose, main_discretize, main_n_bins, main_bin_strategy,
                        main_use_cv, main_n_folds, main_nested_cv, main_device_str):

    synthetic_data_cache = {} # Key: dataset_key. Value: {'models': {...}, 'dataset_config': cfg, 'X_orig_cols_info': X_orig}

    print("\n\n== PHASE 1: GENERATING SYNTHETIC DATA ==")
    for ds_key, ds_data_info in loaded_datasets_map.items():
        X_original_dataset = ds_data_info['X']
        y_original_dataset_df = ds_data_info['y'] # This is a DataFrame
        config_dataset = ds_data_info['config']
        display_name_dataset = config_dataset.get('name', ds_key)
        target_col_dataset_actual = config_dataset.get('target_column_inferred', config_dataset['target_column'])

        print(f"\n--- Preparing data for Gen Models on: {display_name_dataset} (Key: {ds_key}) ---")
        synthetic_data_cache[ds_key] = {
            'models': {}, 'dataset_config': config_dataset.copy(),
            'X_orig_cols_info': X_original_dataset.copy() # Store original X for TSTR OHE guidance
        }

        # This call prepares df_for_gen_model_training (X features potentially discretized, target LabelEncoded)
        # AND ALSO returns OHE'd test splits if not using CV (X_test_eval_non_cv, y_test_eval_non_cv_series)
        _, X_test_eval_non_cv_main, _, y_test_eval_non_cv_series_main, df_for_gen_model_training_main = preprocess_data(
            X_original_dataset, y_original_dataset_df, target_col_dataset_actual,
            main_discretize, main_n_bins, main_bin_strategy,
            test_size=0.25, random_state=current_seed, cv_fold=None, n_folds=None,
            model_name_for_disc=None # Pass model name if user's preprocess_data uses it for specific discretization
        )
        if not (main_use_cv or main_nested_cv):
            synthetic_data_cache[ds_key]['X_test_eval_non_cv'] = X_test_eval_non_cv_main
            synthetic_data_cache[ds_key]['y_test_eval_non_cv_series'] = y_test_eval_non_cv_series_main

        num_samples_to_generate = len(df_for_gen_model_training_main)
        if num_samples_to_generate == 0:
            print(f"ERR: df_for_gen_model_training is empty for {display_name_dataset}. Skipping generation."); continue

        for model_name_ds_loop in models_to_run: # Iterate over models selected in args
            print(f"\n-- Training {model_name_ds_loop.upper()} on {display_name_dataset} --")
            model_obj_trained = None; time_train_model = 0.0; data_synthetic_model = None; bic_val_model = None

            # Calls to user's specific train/generate functions from their script
            # These need to be wrapped or adapted if they don't fit the expected pattern
            # or if time capture needs to be external.

            # For functions like train_ctgan(X_train, discrete_columns, epochs, batch_size)
            # X_train should be df_for_gen_model_training_main.
            # discrete_columns needs to be inferred from df_for_gen_model_training_main.

            # Target column is target_col_dataset_actual in df_for_gen_model_training_main

            # --- CTGAN (from user's script) ---
            if model_name_ds_loop.lower() == 'ctgan' and CTGAN_AVAILABLE:
                start_t_ctgan = time.time()
                # User's train_ctgan takes combined X_train (features+target), discrete_cols
                # df_for_gen_model_training_main is already combined and target is LE'd
                # discrete_columns for CTGAN often refers to categorical feature columns (not target)
                ctgan_discrete_cols = [c for c in df_for_gen_model_training_main.columns
                                       if c != target_col_dataset_actual and
                                       (df_for_gen_model_training_main[c].dtype == 'object' or
                                        df_for_gen_model_training_main[c].nunique() < (main_n_bins + 5))] # Heuristic for discretized being cat

                # User's `train_ctgan` has internal sampling for large datasets.
                model_obj_trained = train_ctgan(df_for_gen_model_training_main, ctgan_discrete_cols, epochs=arg_ctgan_epochs, batch_size=500) # Using user's train_ctgan
                time_train_model = time.time() - start_t_ctgan
                if model_obj_trained:
                    data_synthetic_model = generate_ctgan_synthetic_data(model_obj_trained, df_for_gen_model_training_main, num_samples_to_generate)

            # --- RLiG (from user's script) ---
            elif model_name_ds_loop.lower() == 'rlig' and RLIG_AVAILABLE:
                start_t_rlig = time.time()
                X_rlig_train = df_for_gen_model_training_main.drop(columns=[target_col_dataset_actual])
                y_rlig_train_series = df_for_gen_model_training_main[target_col_dataset_actual] # Already LE'd
                model_obj_trained = train_rlig(X_rlig_train, y_rlig_train_series, episodes=arg_rlig_episodes, epochs=arg_rlig_epochs)
                time_train_model = time.time() - start_t_rlig
                if model_obj_trained:
                    # RLiG's sample might be direct: model_obj_trained.sample(num_samples_to_generate)
                    # And then needs to be combined into a DataFrame matching df_for_gen_model_training_main schema.
                    # This requires a wrapper or careful handling.
                    # Assuming user's generate_rlig_synthetic_data (if provided, or a wrapper) handles this.
                    # For now, let's assume it needs a wrapper to return a df.
                    try:
                        raw_syn_rlig = model_obj_trained.sample(num_samples_to_generate)
                        # Construct DataFrame. This is simplified. RLiG may output X and y separately or need more complex reconstruction.
                        if isinstance(raw_syn_rlig, tuple) and len(raw_syn_rlig) == 2: # if (X,y)
                             data_synthetic_model = pd.concat([pd.DataFrame(raw_syn_rlig[0], columns=X_rlig_train.columns),
                                                              pd.Series(raw_syn_rlig[1], name=target_col_dataset_actual)], axis=1)
                        elif isinstance(raw_syn_rlig, np.ndarray): # if combined Xy
                             data_synthetic_model = pd.DataFrame(raw_syn_rlig, columns=df_for_gen_model_training_main.columns)
                        else: # if just X
                             print("WARN: RLiG sample output format unexpected. Assuming features only.")
                             data_synthetic_model = pd.DataFrame(raw_syn_rlig, columns=X_rlig_train.columns)
                             # Need to add a dummy target or handle this case
                        if hasattr(model_obj_trained, 'best_score'): bic_val_model = model_obj_trained.best_score
                    except Exception as e_rlig_syn: print(f"ERR generating RLiG synthetic: {e_rlig_syn}")


            # --- NaiveBayes (GaussianNB from user script, for TSTR) ---
            # For a generative NB, we'd use pgmpy as shown in the placeholder `train_naive_bayes_generative`
            # The user's `train_naive_bayes` uses sklearn's GaussianNB, which is discriminative.
            # If 'nb' means a generative pgmpy Naive Bayes:
            elif model_name_ds_loop.lower() == 'nb' and PGMPY_AVAILABLE: # Assuming a pgmpy based generative NB
                 print("Using PGMpy based Generative Naive Bayes (Placeholder implementation needed).")
                 # TODO: Implement or call a `train_pgmpy_naive_bayes_wrapper`
                 # model_obj_trained, time_train_model = train_pgmpy_naive_bayes_wrapper(df_for_gen_model_training_main, target_col_dataset_actual)
                 # if model_obj_trained: data_synthetic_model = generate_pgmpy_naive_bayes_data_wrapper(...)
                 # bic_val_model = calculate_bic_score(model_obj_trained, df_for_gen_model_training_main, 'nb_pgmpy')

            # --- GANBLR & GANBLR++ (using pgmpy components from user script) ---
            elif model_name_ds_loop.lower() in ['ganblr', 'ganblr++'] and PGMPY_AVAILABLE:
                print(f"Training {model_name_ds_loop.upper()} on {display_name_dataset} using pgmpy.")
                start_t_bn = time.time()
                structure_learning_algo = TreeSearch if model_name_ds_loop.lower() == 'ganblr' else HillClimbSearch

                # Ensure data is suitable for pgmpy (all discrete, integer encoded)
                # df_for_gen_model_training_main's categoricals are LE'd, numericals are ordinal discretized.
                # pgmpy usually expects all columns to be int if not providing state names.
                data_for_pgmpy = df_for_gen_model_training_main.copy()
                for col_pgm in data_for_pgmpy.columns:
                    if not pd.api.types.is_integer_dtype(data_for_pgmpy[col_pgm]):
                        try: data_for_pgmpy[col_pgm] = LabelEncoder().fit_transform(data_for_pgmpy[col_pgm].astype(str))
                        except: print(f"WARN: Could not ensure int type for pgmpy col {col_pgm}")

                try:
                    est = structure_learning_algo(data_for_pgmpy)
                    # User script uses BIC for HillClimb, TreeSearch might not take scoring_method in estimate()
                    if model_name_ds_loop.lower() == 'ganblr++':
                        best_model_struct = est.estimate(scoring_method=BIC(data_for_pgmpy))
                    else: # ganblr (TreeSearch)
                        best_model_struct = est.estimate() # TreeSearch estimate() might not take scoring_method

                    # Train parameters on this structure (user's train_bn)
                    model_obj_trained = train_bn(best_model_struct, data_for_pgmpy) # train_bn from user script
                    time_train_model = time.time() - start_t_bn
                    if model_obj_trained:
                        data_synthetic_model = generate_bn_synthetic_data(model_obj_trained, data_for_pgmpy, num_samples_to_generate)
                        bic_val_model = calculate_bic_score(model_obj_trained, data_for_pgmpy, model_name_ds_loop.lower())
                except Exception as e_bn_train:
                    print(f"ERR training {model_name_ds_loop}: {e_bn_train}")
                    time_train_model = time.time() - start_t_bn # Record time even if failed


            # --- CTABGAN (from user's script) ---
            elif model_name_ds_loop.lower() == 'ctabgan' and CTABGAN_AVAILABLE:
                start_t_ctab = time.time()
                # User's train_ctabgan takes X, y, cat_cols, epochs
                X_ctab_train = df_for_gen_model_training_main.drop(columns=[target_col_dataset_actual])
                y_ctab_train_df = df_for_gen_model_training_main[[target_col_dataset_actual]] # Needs to be DataFrame for user's func

                # Infer categorical columns for CTABGAN from X_ctab_train (non-target features)
                # User's train_ctabgan expects original (pre-LE/pre-discretization) categoricals,
                # but df_for_gen_model_training_main has these already processed.
                # This might require passing column type info from original data.
                # For now, using a heuristic on df_for_gen_model_training_main's X part.
                ctabgan_cat_cols = [c for c in X_ctab_train.columns if X_ctab_train[c].nunique() < (main_n_bins+5)]

                model_obj_trained = train_ctabgan(X_ctab_train, y_ctab_train_df, categorical_columns=ctabgan_cat_cols, epochs=arg_ctgan_epochs) # user's ctgan_epochs for ctabgan too
                time_train_model = time.time() - start_t_ctab
                if model_obj_trained:
                    data_synthetic_model = generate_ctabgan_synthetic_data(model_obj_trained, df_for_gen_model_training_main, num_samples_to_generate)


            # --- GReaT (from user's script) ---
            elif model_name_ds_loop.lower() == 'great' and GREAT_AVAILABLE:
                start_t_gr = time.time()
                X_gr_train = df_for_gen_model_training_main.drop(columns=[target_col_dataset_actual])
                y_gr_train_series = df_for_gen_model_training_main[target_col_dataset_actual]
                model_obj_trained = train_great(X_gr_train, y_gr_train_series, batch_size=arg_great_bs, epochs=arg_great_epochs)
                time_train_model = time.time() - start_t_gr
                if model_obj_trained:
                    data_synthetic_model = generate_great_synthetic_data(model_obj_trained, df_for_gen_model_training_main, num_samples_to_generate)


            # --- DistSampl (from user's script) ---
            elif model_name_ds_loop.lower() == 'dist_sampl' and DIST_SAMPL_AVAILABLE:
                start_t_ds = time.time()
                X_ds_train = df_for_gen_model_training_main.drop(columns=[target_col_dataset_actual])
                y_ds_train_df = df_for_gen_model_training_main[[target_col_dataset_actual]] # User's func expects y_train DataFrame
                model_obj_trained = train_dist_sampl(X_ds_train, y_ds_train_df, epochs=arg_dist_sampl_epochs, random_seed=current_seed)
                time_train_model = time.time() - start_t_ds
                if model_obj_trained:
                    data_synthetic_model = generate_dist_sampl_synthetic_data(model_obj_trained, df_for_gen_model_training_main, num_samples_to_generate)

            # --- TabDiff (from user's script) ---
            # TabDiff integration in user's script is complex and seems to do its own data loading/handling via DataLoader.
            # This needs significant adaptation to fit the current `df_for_gen_model_training_main` flow.
            # The `train_tabdiff` in user's script takes `train_data` (a TabDiffDataset obj), `train_loader`, `name`, `epochs`, `seed`.
            elif model_name_ds_loop.lower() == 'tabdiff' and TABDIFF_AVAILABLE:
                print("TabDiff training (using user's script function, may need adaptation)...")
                start_t_td = time.time()
                # TODO: Adapt df_for_gen_model_training_main to what user's train_tabdiff expects for `train_data` and `train_loader`.
                # This is a major TODO as it requires understanding TabDiff's specific data input.
                # For now, this will likely fail or need a placeholder call.
                # model_obj_trained = train_tabdiff(df_for_gen_model_training_main, None, ds_key, arg_tabdiff_epochs, current_seed) # Placeholder train_loader=None
                print("TODO: Implement proper data adaptation and call for train_tabdiff.")
                time_train_model = time.time() - start_t_td
                if model_obj_trained:
                    # data_synthetic_model = generate_tabdiff_synthetic_data(model_obj_trained, df_for_gen_model_training_main, num_samples_to_generate)
                    print("TODO: Implement generate_tabdiff_synthetic_data.")


            else:
                if verbose_global: print(f"Model {model_name_ds_loop} not implemented in this training loop or not available. Skipping.")

            synthetic_data_cache[ds_key]['models'][model_name_ds_loop] = {
                'data': data_synthetic_model, 'train_time': time_train_model, 'bic': bic_val_model, 'model_obj': model_obj_trained
            }
            if data_synthetic_model is not None: print(f"Generated {len(data_synthetic_model)} samples for {model_name_ds_loop}. Time: {time_train_model:.2f}s")
            else: print(f"Failed to generate data for {model_name_ds_loop} on {display_name_dataset}.")
            gc.collect()

# --- Helper functions for evaluation phase ---

def evaluate_models_on_fold(dataset_key, synthetic_data_cache, X_test_ohe, y_test_series, 
                          models_to_evaluate, verbose_eval=False, random_seed_eval=42, 
                          original_X_for_column_info=None):
    """
    Evaluates all synthetic data models on a single test fold.
    
    Parameters:
    -----------
    dataset_key : str
        Key of the dataset in synthetic_data_cache
    synthetic_data_cache : dict
        Dictionary containing synthetic data for all models
    X_test_ohe : DataFrame
        One-hot encoded test features
    y_test_series : Series
        Test target values
    models_to_evaluate : list
        List of model names to evaluate
    verbose_eval : bool
        Whether to print detailed evaluation information
    random_seed_eval : int
        Random seed for evaluation classifiers
    original_X_for_column_info : DataFrame
        Original feature DataFrame for guiding OHE in evaluate_tstr
        
    Returns:
    --------
    dict : Results dictionary with metrics, times, and BIC scores
    """
    if verbose_eval: print(f"Evaluating {len(models_to_evaluate)} models on dataset {dataset_key}")
    
    # Initialize results structure
    fold_results = {
        'metrics': {}, 
        'times': {'training_time': {}}, 
        'bic_scores': {'bic': {}}
    }
    
    # Handle case where no models were generated for this dataset
    if dataset_key not in synthetic_data_cache:
        print(f"No synthetic data found for dataset {dataset_key}")
        return fold_results
    
    # Loop through all models
    for model_name in models_to_evaluate:
        if verbose_eval: print(f"Evaluating {model_name} on {dataset_key}...")
        
        # Get model's synthetic data and stats from cache
        model_data = synthetic_data_cache[dataset_key]['models'].get(model_name, {})
        synthetic_data = model_data.get('data')
        train_time = model_data.get('train_time', None)
        bic_score = model_data.get('bic', None)
        
        # Store training time
        fold_results['times']['training_time'][model_name] = train_time
        
        # Store BIC score if available
        if bic_score is not None:
            fold_results['bic_scores']['bic'][model_name] = bic_score
        
        # Skip evaluation if no synthetic data was generated
        if synthetic_data is None or len(synthetic_data) == 0:
            if verbose_eval: print(f"No synthetic data available for {model_name} on {dataset_key}")
            continue
            
        # Use the evaluate_tstr function to get classifier accuracies
        clf_accuracies = evaluate_tstr(
            synthetic_data, 
            X_test_ohe, 
            y_test_series,
            original_X_for_column_info,
            random_state_eval=random_seed_eval,
            verbose_eval_tstr=verbose_eval
        )
        
        # Store accuracy for each classifier
        for clf_name, accuracy in clf_accuracies.items():
            metric_key = f"{clf_name}_accuracy"
            if metric_key not in fold_results['metrics']:
                fold_results['metrics'][metric_key] = {}
            fold_results['metrics'][metric_key][model_name] = accuracy
            
    return fold_results


def average_fold_results(fold_results_list):
    """
    Averages results across multiple folds or rounds.
    
    Parameters:
    -----------
    fold_results_list : list
        List of result dictionaries from evaluate_models_on_fold
        
    Returns:
    --------
    dict : Averaged results
    """
    if not fold_results_list:
        return {}
        
    # Initialize the structure for averaged results
    averaged_results = {
        'metrics': {},
        'times': {'training_time': {}},
        'bic_scores': {'bic': {}}
    }
    
    # Track counts for proper averaging
    counts = {
        'metrics': {},
        'times': {'training_time': {}},
        'bic_scores': {'bic': {}}
    }
    
    # Accumulate values across all folds
    for fold_result in fold_results_list:
        # Process metrics (classifier accuracies)
        for metric_name, model_scores in fold_result.get('metrics', {}).items():
            if metric_name not in averaged_results['metrics']:
                averaged_results['metrics'][metric_name] = {}
                counts['metrics'][metric_name] = {}
                
            for model_name, score in model_scores.items():
                if score is not None:
                    if model_name not in averaged_results['metrics'][metric_name]:
                        averaged_results['metrics'][metric_name][model_name] = 0
                        counts['metrics'][metric_name][model_name] = 0
                    
                    averaged_results['metrics'][metric_name][model_name] += score
                    counts['metrics'][metric_name][model_name] += 1
        
        # Process training times
        for model_name, time_val in fold_result.get('times', {}).get('training_time', {}).items():
            if time_val is not None:
                if model_name not in averaged_results['times']['training_time']:
                    averaged_results['times']['training_time'][model_name] = 0
                    counts['times']['training_time'][model_name] = 0
                
                averaged_results['times']['training_time'][model_name] += time_val
                counts['times']['training_time'][model_name] += 1
        
        # Process BIC scores
        for model_name, bic_val in fold_result.get('bic_scores', {}).get('bic', {}).items():
            if bic_val is not None:
                if model_name not in averaged_results['bic_scores']['bic']:
                    averaged_results['bic_scores']['bic'][model_name] = 0
                    counts['bic_scores']['bic'][model_name] = 0
                
                averaged_results['bic_scores']['bic'][model_name] += bic_val
                counts['bic_scores']['bic'][model_name] += 1
    
    # Calculate averages
    # For metrics
    for metric_name, model_scores in averaged_results['metrics'].items():
        for model_name, accumulated_score in model_scores.items():
            count = counts['metrics'][metric_name].get(model_name, 0)
            if count > 0:
                averaged_results['metrics'][metric_name][model_name] = accumulated_score / count
            else:
                averaged_results['metrics'][metric_name][model_name] = None
    
    # For training times
    for model_name, accumulated_time in averaged_results['times']['training_time'].items():
        count = counts['times']['training_time'].get(model_name, 0)
        if count > 0:
            averaged_results['times']['training_time'][model_name] = accumulated_time / count
        else:
            averaged_results['times']['training_time'][model_name] = None
    
    # For BIC scores
    for model_name, accumulated_bic in averaged_results['bic_scores']['bic'].items():
        count = counts['bic_scores']['bic'].get(model_name, 0)
        if count > 0:
            averaged_results['bic_scores']['bic'][model_name] = accumulated_bic / count
        else:
            averaged_results['bic_scores']['bic'][model_name] = None
    
    return averaged_results


    # --- PHASE 2 & 3: EVALUATION & AGGREGATION ---
    # (This section is adapted from the previous full version, ensure correctness)
    print("\n\n== PHASE 2 & 3: TSTR EVALUATION & RESULT AGGREGATION ==")
    all_rounds_results_by_dataset_key = {}
    num_eval_rounds_final = n_rounds if not (main_use_cv and not main_nested_cv) else 1

    for round_idx_final in range(num_eval_rounds_final):
        current_round_seed_eval = current_seed + round_idx_final
        np.random.seed(current_round_seed_eval); torch.manual_seed(current_round_seed_eval)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_round_seed_eval)
        print(f"\n=== Evaluation Round {round_idx_final+1}/{num_eval_rounds_final} (Seed: {current_round_seed_eval}) ===")

        for dataset_key_eval, cache_entry_eval in synthetic_data_cache.items():
            if not cache_entry_eval.get('models'): continue # Skip if no models were processed for this dataset

            if dataset_key_eval not in all_rounds_results_by_dataset_key:
                all_rounds_results_by_dataset_key[dataset_key_eval] = {}

            config_eval = cache_entry_eval['dataset_config']
            display_name_eval = config_eval.get('name', dataset_key_eval)
            target_col_eval_actual = config_eval.get('target_column_inferred', config_eval['target_column'])
            # Get original X and y for creating CV folds' test sets
            X_orig_for_fold_creation = loaded_datasets_map[dataset_key_eval]['X']
            y_orig_df_for_fold_creation = loaded_datasets_map[dataset_key_eval]['y']
            original_X_for_tstr_eval = cache_entry_eval['X_orig_cols_info'] # For guiding OHE in evaluate_tstr

            if main_use_cv or main_nested_cv:
                fold_results_list_current_round_ds = []
                for fold_idx_eval_loop in range(main_n_folds):
                    print(f"\n--- Fold {fold_idx_eval_loop + 1}/{main_n_folds} for {display_name_eval} (Round {round_idx_final+1}) ---")
                    # Get OHE'd test split for this fold
                    _, X_test_ohe_current_fold, _, y_test_series_current_fold, _ = preprocess_data(
                        X_orig_for_fold_creation, y_orig_df_for_fold_creation, target_col_eval_actual,
                        main_discretize, main_n_bins, main_bin_strategy, # Use main discretize for eval consistency
                        cv_fold=fold_idx_eval_loop, n_folds=main_n_folds, random_state=current_round_seed_eval,
                        model_name_for_disc=None # Not model-specific for eval splits
                    )
                    fold_run_res_dict = evaluate_models_on_fold(
                        dataset_key_eval, synthetic_data_cache, X_test_ohe_current_fold, y_test_series_current_fold,
                        models_to_run, verbose_global, current_round_seed_eval, original_X_for_tstr_eval
                    )
                    fold_results_list_current_round_ds.append(fold_run_res_dict)
                all_rounds_results_by_dataset_key[dataset_key_eval][round_idx_final] = average_fold_results(fold_results_list_current_round_ds)
            else: # Non-CV path
                X_test_non_cv_ohe_main = cache_entry_eval['X_test_eval_non_cv']
                y_test_non_cv_series_main = cache_entry_eval['y_test_eval_non_cv_series']
                non_cv_round_results_dict = evaluate_models_on_fold(
                     dataset_key_eval, synthetic_data_cache, X_test_non_cv_ohe_main, y_test_non_cv_series_main,
                     models_to_run, verbose_global, current_round_seed_eval, original_X_for_tstr_eval
                )
                all_rounds_results_by_dataset_key[dataset_key_eval][round_idx_final] = non_cv_round_results_dict

    final_results_aggregated = {}
    for ds_key_agg, rounds_data_agg in all_rounds_results_by_dataset_key.items():
        list_round_res_ds = [res for res in rounds_data_agg.values() if res]
        if list_round_res_ds: final_results_aggregated[ds_key_agg] = average_fold_results(list_round_res_ds)

    return final_results_aggregated, synthetic_data_cache # Return cache for format_results


def format_results(results_dict_by_dataset_key, sdc_for_names=None): # From user script, adapted
    if not results_dict_by_dataset_key:
        return {'accuracy': pd.DataFrame(), 'time': pd.DataFrame(), 'bic': pd.DataFrame()}

    all_accuracy_dfs, all_time_dfs, all_bic_dfs = [], [], []
    all_model_names_found = set()
    for ds_results in results_dict_by_dataset_key.values():
        if ds_results and 'times' in ds_results and 'training_time' in ds_results['times']:
            all_model_names_found.update(ds_results['times']['training_time'].keys())
    model_order = sorted(list(all_model_names_found), key=lambda x: x.lower()) # Ensure consistent order

    for dataset_key, data in results_dict_by_dataset_key.items():
        ds_display_name = dataset_key
        if sdc_for_names and dataset_key in sdc_for_names and 'dataset_config' in sdc_for_names[dataset_key]:
            ds_display_name = sdc_for_names[dataset_key]['dataset_config'].get('name', dataset_key)

        # Accuracy: metrics are {'LR_accuracy': {'modelA': val}, ...}
        acc_data_for_df = []
        if 'metrics' in data:
            # Sort metric keys (LR_accuracy, MLP_accuracy, ...)
            metric_keys_sorted = sorted([k for k in data['metrics'].keys() if k.endswith('_accuracy')])
            for metric_name_full in metric_keys_sorted:
                clf_abbr = metric_name_full.replace('_accuracy', '')
                row_data = {'Metric': clf_abbr}
                for model_col in model_order: # Use sorted model_order for columns
                    row_data[model_col.upper()] = data['metrics'].get(metric_name_full, {}).get(model_col)
                acc_data_for_df.append(row_data)
        if acc_data_for_df:
            df_acc_ds = pd.DataFrame(acc_data_for_df).set_index('Metric')
            df_acc_ds.columns = pd.MultiIndex.from_product([[ds_display_name], df_acc_ds.columns])
            all_accuracy_dfs.append(df_acc_ds)

        # Time
        if 'times' in data and 'training_time' in data['times']:
            time_row_data = {}
            for model_col in model_order:
                time_row_data[model_col.upper()] = data['times']['training_time'].get(model_col)
            df_time_ds = pd.DataFrame([time_row_data], index=['Training Time (s)'])
            df_time_ds.columns = pd.MultiIndex.from_product([[ds_display_name], df_time_ds.columns])
            all_time_dfs.append(df_time_ds)

        # BIC
        if 'bic_scores' in data and 'bic' in data['bic_scores']:
            bic_row_data = {}
            has_any_bic = False
            for model_col in model_order:
                val = data['bic_scores']['bic'].get(model_col)
                bic_row_data[model_col.upper()] = val
                if val is not None: has_any_bic = True
            if has_any_bic:
                df_bic_ds = pd.DataFrame([bic_row_data], index=['BIC Score'])
                df_bic_ds.columns = pd.MultiIndex.from_product([[ds_display_name], df_bic_ds.columns])
                all_bic_dfs.append(df_bic_ds)

    final_accuracy_df = pd.concat(all_accuracy_dfs, axis=1).sort_index(axis=1, level=0) if all_accuracy_dfs else pd.DataFrame()
    final_time_df = pd.concat(all_time_dfs, axis=1).sort_index(axis=1, level=0) if all_time_dfs else pd.DataFrame()
    final_bic_df = pd.concat(all_bic_dfs, axis=1).sort_index(axis=1, level=0) if all_bic_dfs else pd.DataFrame()

    return {'accuracy': final_accuracy_df, 'time': final_time_df, 'bic': final_bic_df}


def save_results_to_csv(formatted_results_dict, prefix="tstr_eval", directory="results"): # from user script
    os.makedirs(directory, exist_ok=True)
    for result_type, df in formatted_results_dict.items():
        if not df.empty:
            filename = os.path.join(directory, f"{prefix}_{result_type}_results.csv")
            df.to_csv(filename)
            print(f"Saved {result_type} results to {filename}")
        else:
            print(f"No {result_type} data to save for prefix {prefix}.")


# --- Argument Parser (Simplified) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Simplified TSTR Evaluation with UCI ID and Local File Support")
    parser.add_argument("--models",type=str,nargs="+",default=['ctgan','rlig'], help="Models: ctgan rlig nb ganblr ganblr++ great dist_sampl tabdiff ctabgan") # Default to a few
    parser.add_argument("--dataset_keys", type=str, nargs="*", default=None, help="Specific dataset keys (from UCI/Local configs) to run.")
    parser.add_argument("--include_local", action='store_true', help="Include default local datasets (KDD, Loan) with UCI.")
    parser.add_argument("--include_local_only", action='store_true', help="Run ONLY default local datasets.")
    parser.add_argument("--include_all_uci", action='store_true', help="Explicitly include all default UCI (used if other flags might exclude them).")


    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_rounds", type=int, default=1, help="Eval rounds / Nested CV outer loops.") # Default 1
    parser.add_argument("--discretize", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--n_bins", type=int, default=7) # User default
    parser.add_argument("--bin_strategy", type=str, default='quantile', choices=['quantile', 'uniform', 'kmeans'])
    parser.add_argument("--use_cv", action='store_true', help="Use k-fold CV for TSTR evaluation.")
    parser.add_argument("--n_folds", type=int, default=2) # User default from their script
    parser.add_argument("--nested_cv", action='store_true', help="Use nested CV.")
    parser.add_argument("--verbose", action='store_true')

    # Model hyperparams (add all from user's script)
    parser.add_argument("--ctgan_epochs", type=int, default=50) # User default
    parser.add_argument("--rlig_episodes", type=int, default=2) # User default
    parser.add_argument("--rlig_epochs", type=int, default=5)   # User default
    parser.add_argument("--great_bs", type=int, default=1)      # User default
    parser.add_argument("--great_epochs", type=int, default=1)  # User default
    parser.add_argument("--dist_sampl_epochs", type=int, default=50) # User default
    parser.add_argument("--tabdiff_epochs", type=int, default=5) # User default (TabDiff might need more)
    # TODO: Add args for CTABGAN epochs if different from CTGAN, GANBLR params etc.

    return parser.parse_args()

# --- Main Execution ---
def main():
    global verbose_global # To allow helper functions to use it without passing
    args = parse_args()
    verbose_global = args.verbose

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Settings: Seed={args.seed}, Device={device}, Verbose={verbose_global}, Models={args.models}")
    print(f"Discretize={args.discretize}, Bins={args.n_bins}, Strategy={args.bin_strategy}")
    print(f"CV={args.use_cv}, Folds={args.n_folds}, NestedCV={args.nested_cv}, Rounds={args.n_rounds}")


    keys_to_run = set()
    if args.dataset_keys: # Specific keys take precedence
        keys_to_run.update(args.dataset_keys)
    else: # No specific keys, use flags or default
        if args.include_local_only:
            keys_to_run.update(DEFAULT_LOCAL_CONFIG.keys())
            print("Running local datasets ONLY.")
        else:
            if args.include_all_uci or not args.include_local: # Default to UCI if no flags or only --include_local (which adds to UCI)
                keys_to_run.update(DEFAULT_UCI_CONFIG.keys())
                print("Including default UCI datasets.")
            if args.include_local:
                keys_to_run.update(DEFAULT_LOCAL_CONFIG.keys())
                print("Including default Local datasets.")
        if not keys_to_run: # Fallback if logic somehow results in empty set (e.g. only --include_local_only but no local defined)
             print("Defaulting to all UCI datasets as no specific selection resolved.")
             keys_to_run.update(DEFAULT_UCI_CONFIG.keys())


    if not keys_to_run: print("No datasets selected. Exiting."); exit()
    print(f"Final dataset keys to process: {sorted(list(keys_to_run))}")

    loaded_datasets = {}
    for key_main_load in sorted(list(keys_to_run)):
        X_main, y_main, cfg_main = None, None, None
        if key_main_load in DEFAULT_UCI_CONFIG:
            cfg_main = DEFAULT_UCI_CONFIG[key_main_load].copy()
            X_main, y_main, cfg_main = fetch_and_prepare_uci_dataset(key_main_load, cfg_main)
        elif key_main_load in DEFAULT_LOCAL_CONFIG:
            cfg_main = DEFAULT_LOCAL_CONFIG[key_main_load].copy()
            X_main, y_main, cfg_main = load_local_dataset(key_main_load, cfg_main)

        if X_main is not None and y_main is not None:
            loaded_datasets[key_main_load] = {'X': X_main, 'y': y_main, 'config': cfg_main}
        else: print(f"Skipping dataset '{key_main_load}' due to loading error.")

    if not loaded_datasets: print("No datasets loaded successfully. Exiting."); exit()

    results_final_agg, sdc_final = compare_models_tstr( # sdc_final is synthetic_data_cache
        loaded_datasets, args.models, args.n_rounds, args.seed,
        args.rlig_episodes, args.rlig_epochs, args.ctgan_epochs,
        args.great_bs, args.great_epochs, args.dist_sampl_epochs, args.tabdiff_epochs, # Pass all model args
        args.verbose, args.discretize, args.n_bins, args.bin_strategy,
        args.use_cv, args.n_folds, args.nested_cv, device
    )

    if results_final_agg:
        formatted_out = format_results(results_final_agg, sdc_final) # Pass sdc for display names
        print("\n\n=== TSTR ACCURACY RESULTS ==="); print(formatted_out['accuracy'].to_string())
        print("\n\n=== MODEL TRAINING TIME (s) ==="); print(formatted_out['time'].to_string())
        if not formatted_out['bic'].empty: print("\n\n=== BIC SCORES ==="); print(formatted_out['bic'].to_string())
        else: print("\n\n=== BIC SCORES ===\n (No BIC scores)")

        ts_out = time.strftime("%Y%m%d_%H%M%S")
        res_dir_out = os.path.join("results_tstr_simplified", f"{ts_out}_run")
        save_results_to_csv(formatted_out, prefix=f"eval_{ts_out}", directory=res_dir_out)
        with open(os.path.join(res_dir_out, "run_args.json"), 'w') as f_args_out: json.dump(vars(args), f_args_out, indent=4)
        print(f"Results and args saved to: {res_dir_out}")
    else: print("Evaluation completed, but no results were aggregated.")
    print("\nScript finished.")

if __name__ == "__main__":
    # Create dummy data directories if they don't exist for robustness on first run
    os.makedirs("data/nsl-kdd/Full Data", exist_ok=True)
    os.makedirs("data_cache", exist_ok=True)
    os.makedirs("results_tstr_simplified", exist_ok=True)
    # User needs to place 'loan_approval_dataset.csv' and 'KDDTrain+.arff' in the 'data' subdirectories.
    main()