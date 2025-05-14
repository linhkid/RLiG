import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import subprocess
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split

class TabSynWrapper:
    """
    A wrapper for TabSyn that properly prepares data and handles the execution
    of TabSyn's training and sampling processes according to TabSyn's expected formats.
    """
    
    def __init__(self, dataset_name="custom", epochs=1000, gpu=0, nfe=50):
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.gpu = gpu
        self.nfe = nfe
        self.setup_paths()
        
    def setup_paths(self):
        """Setup directory paths required by TabSyn"""
        # TabSyn expects specific paths
        self.base_dir = os.path.join(os.getcwd(), 'tabsyn')
        self.data_dir = os.path.join(self.base_dir, 'data', self.dataset_name)
        self.info_dir = os.path.join(self.base_dir, 'data', 'Info')
        self.info_file = os.path.join(self.info_dir, f'{self.dataset_name}.json')
        self.synthetic_dir = os.path.join(self.base_dir, 'synthetic', self.dataset_name)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)
        os.makedirs(self.synthetic_dir, exist_ok=True)
    
    def prepare_data(self, X, y=None, test_size=0.1, random_state=42):
        """
        Prepare data in the format expected by TabSyn.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features dataframe
        y : pandas.Series, optional
            Target variable
        test_size : float, default=0.1
            Proportion of data to use for test set
        random_state : int, default=42
            Random seed for reproducibility
        """
        # If y is provided as a Series, convert to DataFrame column
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.to_frame()
            elif isinstance(y, np.ndarray) and y.ndim == 1:
                y = pd.DataFrame(y, columns=['target'])
        
        # Combine X and y if y exists
        if y is not None:
            data = pd.concat([X, y], axis=1)
        else:
            data = X.copy()
            
        # Identify numerical and categorical columns
        num_cols = data.select_dtypes(include=['number']).columns.tolist()
        cat_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        
        # Add a dummy numerical column if no numerical columns exist
        # TabSyn requires at least one numerical column
        if len(num_cols) == 0:
            print("No numerical columns found, adding a dummy numerical column")
            data['dummy_numerical'] = np.random.normal(0, 1, len(data))
            num_cols = ['dummy_numerical']
        
        # Split data into train and test sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        
        # Save train and test data
        train_data.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(self.data_dir, 'test.csv'), index=False)
        
        # Create column indices
        column_names = data.columns.tolist()
        num_col_idx = [i for i, col in enumerate(column_names) if col in num_cols]
        cat_col_idx = [i for i, col in enumerate(column_names) if col in cat_cols]
        
        # Determine task type based on y
        if y is not None:
            target_col_idx = [len(column_names) - 1]  # Assuming target is the last column
            if y.dtypes.iloc[0] == 'object' or y.dtypes.iloc[0] == 'category' or y.nunique().iloc[0] < 10:
                task_type = 'binclass' if y.nunique().iloc[0] <= 2 else 'multiclass'
            else:
                task_type = 'regression'
        else:
            # If no target, assume the last column is the target
            target_col_idx = [len(column_names) - 1]
            # Try to infer task type
            last_col = data.iloc[:, -1]
            if pd.api.types.is_numeric_dtype(last_col):
                if last_col.nunique() <= 2:
                    task_type = 'binclass'
                elif last_col.nunique() < 10:
                    task_type = 'multiclass'
                else:
                    task_type = 'regression'
            else:
                task_type = 'binclass' if last_col.nunique() <= 2 else 'multiclass'
        
        # Create the info.json file TabSyn expects
        info = {
            "name": self.dataset_name,
            "task_type": task_type,
            "header": 0,
            "column_names": column_names,
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "file_type": "csv",
            "data_path": os.path.join(self.data_dir, 'train.csv'),
            "test_path": os.path.join(self.data_dir, 'test.csv'),
            "column_info": {col: "float" if col in num_cols else "str" for col in column_names},
            "train_num": len(train_data),
            "test_num": len(test_data)
        }
        
        # Save info.json
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=4)

        # Also save info.json to the dataset directory as TabSyn expects
        with open(os.path.join(self.data_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        # Create numpy files expected by TabSyn
        self._create_numpy_files(train_data, test_data, num_cols, cat_cols, task_type, target_col_idx)
        
        return info
    
    def _create_numpy_files(self, train_data, test_data, num_cols, cat_cols, task_type, target_col_idx):
        """Create the numpy files expected by TabSyn"""
        # Get target column(s)
        if isinstance(target_col_idx, list) and len(target_col_idx) > 0:
            target_cols = train_data.columns[target_col_idx].tolist()
        else:
            target_cols = []
        
        # Extract features and targets
        X_train_num = train_data[num_cols].to_numpy().astype(np.float32)
        X_test_num = test_data[num_cols].to_numpy().astype(np.float32)
        
        # Encode categorical features
        X_train_cat = train_data[cat_cols].copy()
        X_test_cat = test_data[cat_cols].copy()
        
        # Apply label encoding to categorical columns
        for col in cat_cols:
            le = LabelEncoder()
            train_cat = train_data[col].astype(str)
            test_cat = test_data[col].astype(str)
            
            # Fit on train data
            le.fit(train_cat)
            
            # Transform both train and test
            X_train_cat[col] = le.transform(train_cat)
            X_test_cat[col] = le.transform(test_cat)
        
        X_train_cat = X_train_cat.to_numpy()
        X_test_cat = X_test_cat.to_numpy()
        
        # Extract targets
        if target_cols:
            if task_type == 'regression':
                y_train = train_data[target_cols].to_numpy().astype(np.float32)
                y_test = test_data[target_cols].to_numpy().astype(np.float32)
            else:
                y_train = train_data[target_cols].astype(str)
                y_test = test_data[target_cols].astype(str)
                
                # Label encode if classification
                le = LabelEncoder()
                y_train_flat = y_train.iloc[:, 0]
                y_test_flat = y_test.iloc[:, 0]
                
                le.fit(y_train_flat)
                y_train = le.transform(y_train_flat).reshape(-1, 1)
                y_test = le.transform(y_test_flat).reshape(-1, 1)
        else:
            # Create dummy targets if no target specified
            y_train = np.zeros((len(train_data), 1))
            y_test = np.zeros((len(test_data), 1))
        
        # Save numpy files
        np.save(os.path.join(self.data_dir, 'X_num_train.npy'), X_train_num)
        np.save(os.path.join(self.data_dir, 'X_cat_train.npy'), X_train_cat)
        np.save(os.path.join(self.data_dir, 'y_train.npy'), y_train)
        
        np.save(os.path.join(self.data_dir, 'X_num_test.npy'), X_test_num)
        np.save(os.path.join(self.data_dir, 'X_cat_test.npy'), X_test_cat)
        np.save(os.path.join(self.data_dir, 'y_test.npy'), y_test)
        
        # Also save to synthetic directory as TabSyn expects
        train_data.to_csv(os.path.join(self.synthetic_dir, 'real.csv'), index=False)
        test_data.to_csv(os.path.join(self.synthetic_dir, 'test.csv'), index=False)
    
    def run_process_dataset(self):
        """Run TabSyn's process_dataset.py to prepare the dataset"""
        process_cmd = [
            sys.executable,
            os.path.join(self.base_dir, 'process_dataset.py'),
            '--dataname', self.dataset_name
        ]
        
        try:
            subprocess.run(process_cmd, check=True)
            print(f"Successfully processed dataset {self.dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error processing dataset: {e}")
            return False
    
    def train_vae(self):
        """Train TabSyn's VAE model"""
        vae_cmd = [
            sys.executable,
            os.path.join(self.base_dir, 'main.py'),
            '--dataname', self.dataset_name,
            '--method', 'vae',
            '--mode', 'train',
            '--epochs', str(self.epochs),
            '--gpu', str(self.gpu)
        ]
        
        try:
            subprocess.run(vae_cmd, check=True)
            print(f"Successfully trained VAE for {self.dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error training VAE: {e}")
            return False
    
    def train_diffusion(self):
        """Train TabSyn's diffusion model"""
        diffusion_cmd = [
            sys.executable,
            os.path.join(self.base_dir, 'main.py'),
            '--dataname', self.dataset_name,
            '--method', 'tabsyn',
            '--mode', 'train',
            '--gpu', str(self.gpu)
        ]
        
        try:
            subprocess.run(diffusion_cmd, check=True)
            print(f"Successfully trained diffusion model for {self.dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error training diffusion model: {e}")
            return False
    
    def generate_samples(self, n_samples=None):
        """Generate samples using the trained models"""
        # Set default number of samples if not specified
        if n_samples is None:
            with open(self.info_file, 'r') as f:
                info = json.load(f)
                n_samples = info.get('train_num', 1000)
        
        save_path = os.path.join(self.synthetic_dir, 'synthetic.csv')
        
        sample_cmd = [
            sys.executable,
            os.path.join(self.base_dir, 'main.py'),
            '--dataname', self.dataset_name,
            '--method', 'tabsyn',
            '--mode', 'sample',
            '--gpu', str(self.gpu),
            '--steps', str(self.nfe),
            '--save_path', save_path
        ]
        
        try:
            subprocess.run(sample_cmd, check=True)
            print(f"Successfully generated {n_samples} samples for {self.dataset_name}")
            
            # Load the generated data
            synthetic_data = pd.read_csv(save_path)
            return synthetic_data
        except subprocess.CalledProcessError as e:
            print(f"Error generating samples: {e}")
            return None
    
    def fit(self, X, y=None):
        """Fit TabSyn to the data"""
        # Prepare data
        self.prepare_data(X, y)
        
        # Run process_dataset.py
        if not self.run_process_dataset():
            print("Falling back to manual processing...")
            # Don't return here, continue with our custom preprocessing
        
        # Train VAE
        if not self.train_vae():
            print("VAE training failed")
            return False
        
        # Train diffusion model
        if not self.train_diffusion():
            print("Diffusion model training failed")
            return False
        
        return True
    
    def sample(self, n_samples=None):
        """Generate samples using the trained model"""
        return self.generate_samples(n_samples)