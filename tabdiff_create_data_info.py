import os
from scipy.io.arff import loadarff
import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_AVAILABLE = True

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def read_arff_file(file_path):
    """Read an ARFF file and return a pandas DataFrame"""
    data, meta = loadarff(file_path)
    df = pd.DataFrame(data)

    # Convert byte strings to regular strings
    for col in df.columns:
        if df[col].dtype == object:  # Object type typically indicates byte strings from ARFF
            df[col] = df[col].str.decode('utf-8')

    return df, meta


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


def label_encode_cols(X, cols):
    """Label encode categorical columns"""
    X_encoded = X.copy()
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le
    return X_encoded, encoders


# def preprocess_data(X, y, discretize=True, model_name=None, cv_fold=None, n_folds=None):
#     """Preprocess data: optionally discretize continuous variables and encode categoricals
#
#     This version can selectively apply discretization using quantile binning with 7 bins,
#     which better preserves the distribution of the original data. This is especially useful
#     for certain models, while others may perform better with non-discretized data.
#
#     Parameters:
#     -----------
#     X : DataFrame
#         Features to preprocess
#     y : DataFrame or Series
#         Target variable
#     discretize : bool, default=True
#         Whether to apply discretization to continuous features
#     model_name : str, optional
#         Name of the model being trained, used for model-specific preprocessing decisions
#     cv_fold : int, optional
#         Current fold number when doing k-fold cross-validation (0-indexed)
#     n_folds : int, optional
#         Total number of folds when doing k-fold cross-validation
#     """
#     # First, handle missing values
#     # Check if there are any missing values
#     if X.isnull().any().any():
#         print("Handling missing values in the dataset...")
#
#         # For categorical columns, fill with the most frequent value
#         for col in X.select_dtypes(include=['object']).columns:
#             X[col] = X[col].fillna(X[col].mode()[0])
#
#         # For numeric columns, fill with the median
#         for col in X.select_dtypes(include=['number']).columns:
#             X[col] = X[col].fillna(X[col].median())
#
#         print("Missing values have been imputed")
#
#     # Identify column types after imputation
#     continuous_cols = X.select_dtypes(include=['number']).columns
#     categorical_cols = X.select_dtypes(include=['object']).columns
#     print("Continuous columns: ", continuous_cols)
#     print("Categorical columns: ", categorical_cols)
#
#     # Apply discretization based on the flag
#     apply_discretization = discretize
#
#     # Log the discretization status for the current model
#     if model_name:
#         if apply_discretization:
#             print(f"Note: Using discretized features for {model_name}")
#         else:
#             print(f"Note: Using non-discretized features for {model_name}")
#
#     # Create transformation pipeline with optional discretization
#     transformers = []
#     if len(continuous_cols) > 0:
#         if apply_discretization:
#             # Add discretization step to the pipeline
#             continuous_transformer = Pipeline(steps=[
#                 ('scaler', StandardScaler()),
#                 ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
#             ])
#             print("Using discretization with uniform binning (5 bins)")
#         else:
#             # Only standardize without discretization
#             continuous_transformer = Pipeline(steps=[
#                 ('scaler', StandardScaler())
#             ])
#             print("Using standardization without discretization")
#
#         transformers.append(('num', continuous_transformer, continuous_cols))
#
#     # Handle categorical columns
#     if len(categorical_cols) > 0:
#         X, encoders = label_encode_cols(X, categorical_cols)
#
#     # Apply transformations
#     preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
#     X_transformed = preprocessor.fit_transform(X)
#     X_transformed_df = pd.DataFrame(X_transformed, columns=continuous_cols.tolist() + categorical_cols.tolist())
#
#     # Handle target variable
#     if y.isnull().any().any():
#         print("Handling missing values in target variable...")
#         if y.dtypes[0] == 'object':
#             y = y.fillna(y.mode()[0])
#         else:
#             y = y.fillna(y.median())
#
#     if y.dtypes[0] == 'object':
#         label_encoder = LabelEncoder()
#         y_transformed = pd.DataFrame(label_encoder.fit_transform(y.values.ravel()), columns=y.columns)
#     else:
#         y_transformed = y
#
#     # Split data based on whether we're using cross-validation or traditional train-test split
#     if cv_fold is not None and n_folds is not None:
#         from sklearn.model_selection import KFold
#         print(f"Using {n_folds}-fold cross-validation (fold {cv_fold + 1}/{n_folds})")
#
#         # Create fold indices
#         kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
#
#         # Convert to arrays for indexing
#         X_array = X_transformed_df.values
#         y_array = y_transformed.values
#
#         # Get the train/test indices for this fold
#         train_indices = []
#         test_indices = []
#
#         for i, (train_idx, test_idx) in enumerate(kf.split(X_array)):
#             if i == cv_fold:
#                 train_indices = train_idx
#                 test_indices = test_idx
#                 break
#
#         # Split the data using the indices
#         X_train = pd.DataFrame(X_array[train_indices], columns=X_transformed_df.columns)
#         X_test = pd.DataFrame(X_array[test_indices], columns=X_transformed_df.columns)
#         y_train = pd.DataFrame(y_array[train_indices], columns=y_transformed.columns)
#         y_test = pd.DataFrame(y_array[test_indices], columns=y_transformed.columns)
#
#         return X_train, X_test, y_train, y_test
#     else:
#         # Traditional split
#         return train_test_split(X_transformed_df, y_transformed, test_size=0.2, random_state=42, stratify=y)


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


def create_data_info(dataname, dataset_info, discretize):
    X, y = load_dataset(dataname, dataset_info)
    print(X, y)
    print(y.value_counts())

    # Get data and preprocess based on discretization flag for TabDiff
    X_train_tabdiff, X_test_tabdiff, y_train_tabdiff, y_test_tabdiff = (
        preprocess_data(X, y, name=dataname, discretize=discretize, model_name='tabdiff')
    )
    print("Shape of X_train_tabdiff: ", X_train_tabdiff.shape)
    print("Shape of X_test_tabdiff: ", X_test_tabdiff.shape)

    if not os.path.exists(os.path.join("data", dataname)):
        os.makedirs(os.path.join("data", dataname), exist_ok=True)

    from _utils import save_to_csv
    # Save train data
    save_to_csv(X_train_tabdiff, y_train_tabdiff, f"data/{dataname}", "train.csv")

    # Save test data
    save_to_csv(X_test_tabdiff, y_test_tabdiff, f"data/{dataname}", "test.csv")


if __name__ == "__main__":
    create_data_info(dataname="adult", dataset_info=2, discretize=True)
    create_data_info(dataname="Rice", dataset_info=545, discretize=True)
    create_data_info(dataname="car", dataset_info=19, discretize=True)
    create_data_info(dataname="magic", dataset_info=159, discretize=True)
    create_data_info(dataname="default", dataset_info=350, discretize=True)
    create_data_info(dataname="connect4", dataset_info=26, discretize=True)
    create_data_info(dataname="letter_recog", dataset_info=59, discretize=True)
    create_data_info(dataname="maternal_health", dataset_info=863, discretize=True)
    create_data_info(dataname="nursery", dataset_info=76, discretize=True)
    create_data_info(dataname="room_occupancy", dataset_info=864, discretize=True)
    create_data_info(dataname="chess", dataset_info=22, discretize=True)
    create_data_info(dataname="nsl-kdd", dataset_info="data/nsl-kdd/Full -d/KDDTrain20.arff", discretize=True)