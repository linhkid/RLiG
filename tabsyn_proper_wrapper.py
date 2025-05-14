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
        self.base_dir = os.path.abspath('tabsyn')  # Use absolute path
        
        # Main data directories
        self.data_dir = os.path.join(self.base_dir, 'data', self.dataset_name)
        self.info_dir = os.path.join(self.base_dir, 'data', 'Info')
        self.info_file = os.path.join(self.info_dir, f'{self.dataset_name}.json')
        self.synthetic_dir = os.path.join(self.base_dir, 'synthetic', self.dataset_name)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)
        os.makedirs(self.synthetic_dir, exist_ok=True)
        
        # Also create raw data directory if needed
        raw_data_dir = os.path.join(self.base_dir, 'data', self.dataset_name)
        os.makedirs(raw_data_dir, exist_ok=True)
        
        print(f"TabSyn directories: \n  Data: {self.data_dir}\n  Info: {self.info_dir}\n  Synthetic: {self.synthetic_dir}")
    
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
            # Use integers instead of floating point to avoid label encoding issues
            data['dummy_numerical'] = np.arange(len(data)) % 10  # Use modulo to keep values in a small range
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
        
        # Save info.json to multiple locations as TabSyn looks in different places
        # 1. In the Info directory
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=4)

        # 2. In the dataset directory
        with open(os.path.join(self.data_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        # 3. In relative path that TabSyn sometimes expects
        os.makedirs(os.path.join('data', 'Info'), exist_ok=True)
        with open(os.path.join('data', 'Info', f'{self.dataset_name}.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        os.makedirs(os.path.join('data', self.dataset_name), exist_ok=True)
        with open(os.path.join('data', self.dataset_name, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Created info.json files in all required locations for TabSyn")
        
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
            # Make sure we convert to strings to avoid numeric comparison issues
            train_cat = train_data[col].astype(str)
            
            # Get all unique values from both train and test
            all_values = set(train_cat.values)
            test_cat = test_data[col].astype(str)
            all_values.update(test_cat.values)
            
            # Create a mapping dictionary instead of using LabelEncoder
            # This avoids issues with unseen labels
            label_map = {val: i for i, val in enumerate(sorted(all_values))}
            
            # Apply mapping
            X_train_cat[col] = train_cat.map(label_map)
            X_test_cat[col] = test_cat.map(label_map)
            
            # Fill any NaN values (which could occur if test has values not in train)
            X_train_cat[col] = X_train_cat[col].fillna(0).astype(int)
            X_test_cat[col] = X_test_cat[col].fillna(0).astype(int)
        
        X_train_cat = X_train_cat.to_numpy()
        X_test_cat = X_test_cat.to_numpy()
        
        # Extract targets
        if target_cols:
            if task_type == 'regression':
                y_train = train_data[target_cols].to_numpy().astype(np.float32)
                y_test = test_data[target_cols].to_numpy().astype(np.float32)
            else:
                # For classification tasks, also use our manual mapping approach
                target_col = target_cols[0]  # Assume single target column
                train_target = train_data[target_col].astype(str)
                test_target = test_data[target_col].astype(str)
                
                all_target_values = set(train_target.values)
                all_target_values.update(test_target.values)
                
                target_map = {val: i for i, val in enumerate(sorted(all_target_values))}
                
                y_train = np.array([target_map[val] for val in train_target]).reshape(-1, 1)
                y_test = np.array([target_map[val] for val in test_target]).reshape(-1, 1)
        else:
            # Create dummy targets if no target specified
            y_train = np.zeros((len(train_data), 1))
            y_test = np.zeros((len(test_data), 1))
        
        # Save numpy files to multiple locations
        # 1. In the dataset directory
        np.save(os.path.join(self.data_dir, 'X_num_train.npy'), X_train_num)
        np.save(os.path.join(self.data_dir, 'X_cat_train.npy'), X_train_cat)
        np.save(os.path.join(self.data_dir, 'y_train.npy'), y_train)
        
        np.save(os.path.join(self.data_dir, 'X_num_test.npy'), X_test_num)
        np.save(os.path.join(self.data_dir, 'X_cat_test.npy'), X_test_cat)
        np.save(os.path.join(self.data_dir, 'y_test.npy'), y_test)
        
        # 2. In relative paths that TabSyn expects
        rel_data_dir = os.path.join('data', self.dataset_name)
        os.makedirs(rel_data_dir, exist_ok=True)
        
        np.save(os.path.join(rel_data_dir, 'X_num_train.npy'), X_train_num)
        np.save(os.path.join(rel_data_dir, 'X_cat_train.npy'), X_train_cat)
        np.save(os.path.join(rel_data_dir, 'y_train.npy'), y_train)
        
        np.save(os.path.join(rel_data_dir, 'X_num_test.npy'), X_test_num)
        np.save(os.path.join(rel_data_dir, 'X_cat_test.npy'), X_test_cat)
        np.save(os.path.join(rel_data_dir, 'y_test.npy'), y_test)
        
        # Also save CSV files to both synthetic and data directories
        train_data.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(self.data_dir, 'test.csv'), index=False)
        
        train_data.to_csv(os.path.join(rel_data_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(rel_data_dir, 'test.csv'), index=False)
        
        # Save to synthetic directory as TabSyn expects
        os.makedirs(os.path.join('synthetic', self.dataset_name), exist_ok=True)
        train_data.to_csv(os.path.join('synthetic', self.dataset_name, 'real.csv'), index=False)
        test_data.to_csv(os.path.join('synthetic', self.dataset_name, 'test.csv'), index=False)
        
        # And also in the absolute synthetic path
        os.makedirs(self.synthetic_dir, exist_ok=True)
        train_data.to_csv(os.path.join(self.synthetic_dir, 'real.csv'), index=False)
        test_data.to_csv(os.path.join(self.synthetic_dir, 'test.csv'), index=False)
        
        print(f"Saved data files to all required locations")
    
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
        # Copy required files to data/dataset_name directory
        train_csv_src = os.path.join(self.data_dir, 'train.csv')
        test_csv_src = os.path.join(self.data_dir, 'test.csv')
        
        # Also copy to relative path
        rel_data_dir = os.path.join('data', self.dataset_name)
        
        # Make sure the CSV files exist in both places
        for src, dst in [(train_csv_src, os.path.join(rel_data_dir, 'train.csv')),
                         (test_csv_src, os.path.join(rel_data_dir, 'test.csv'))]:
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        # Path to TabSyn main script
        main_script = os.path.join(self.base_dir, 'main.py')
        
        # Print command for debugging
        vae_cmd = [
            sys.executable,
            main_script,
            '--dataname', self.dataset_name,
            '--method', 'vae',
            '--mode', 'train',
            '--epochs', str(self.epochs),
            '--gpu', str(self.gpu)
        ]
        
        print(f"Running VAE command: {' '.join(vae_cmd)}")
        print(f"Working directory: {os.getcwd()}")
        
        try:
            subprocess.run(vae_cmd, check=True)
            print(f"Successfully trained VAE for {self.dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error training VAE: {e}")
            
            # Try with absolute path if relative fails
            try:
                abs_main_script = os.path.abspath(main_script)
                print(f"Trying with absolute path: {abs_main_script}")
                
                vae_cmd[1] = abs_main_script
                subprocess.run(vae_cmd, check=True)
                print(f"Successfully trained VAE for {self.dataset_name} with absolute path")
                return True
            except subprocess.CalledProcessError as e2:
                print(f"Error training VAE with absolute path: {e2}")
                return False
    
    def train_diffusion(self):
        """Train TabSyn's diffusion model"""
        # Path to TabSyn main script
        main_script = os.path.join(self.base_dir, 'main.py')
        
        diffusion_cmd = [
            sys.executable,
            main_script,
            '--dataname', self.dataset_name,
            '--method', 'tabsyn',
            '--mode', 'train',
            '--gpu', str(self.gpu)
        ]
        
        print(f"Running diffusion command: {' '.join(diffusion_cmd)}")
        
        try:
            subprocess.run(diffusion_cmd, check=True)
            print(f"Successfully trained diffusion model for {self.dataset_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error training diffusion model: {e}")
            
            # Try with absolute path if relative fails
            try:
                abs_main_script = os.path.abspath(main_script)
                print(f"Trying with absolute path: {abs_main_script}")
                
                diffusion_cmd[1] = abs_main_script
                subprocess.run(diffusion_cmd, check=True)
                print(f"Successfully trained diffusion model for {self.dataset_name} with absolute path")
                return True
            except subprocess.CalledProcessError as e2:
                print(f"Error training diffusion model with absolute path: {e2}")
                return False
    
    def generate_samples(self, n_samples=None):
        """Generate samples using the trained models"""
        # Set default number of samples if not specified
        if n_samples is None:
            try:
                with open(self.info_file, 'r') as f:
                    info = json.load(f)
                    n_samples = info.get('train_num', 1000)
            except:
                with open(os.path.join('data', 'Info', f'{self.dataset_name}.json'), 'r') as f:
                    info = json.load(f)
                    n_samples = info.get('train_num', 1000)
        
        # Ensure the synthetic directory exists
        os.makedirs(self.synthetic_dir, exist_ok=True)
        os.makedirs(os.path.join('synthetic', self.dataset_name), exist_ok=True)
        
        # Use multiple possible save paths
        rel_save_path = os.path.join('synthetic', self.dataset_name, 'synthetic.csv')
        abs_save_path = os.path.join(self.synthetic_dir, 'synthetic.csv')
        
        # Path to TabSyn main script
        main_script = os.path.join(self.base_dir, 'main.py')
        
        sample_cmd = [
            sys.executable,
            main_script,
            '--dataname', self.dataset_name,
            '--method', 'tabsyn',
            '--mode', 'sample',
            '--gpu', str(self.gpu),
            '--steps', str(self.nfe),
            '--save_path', rel_save_path
        ]
        
        print(f"Running sample command: {' '.join(sample_cmd)}")
        
        try:
            subprocess.run(sample_cmd, check=True)
            print(f"Successfully generated {n_samples} samples for {self.dataset_name}")
            
            # Try to load the generated data from multiple possible locations
            try:
                synthetic_data = pd.read_csv(rel_save_path)
            except:
                try:
                    synthetic_data = pd.read_csv(abs_save_path)
                except:
                    # Check if file exists in other locations
                    for path in [
                        os.path.join(os.getcwd(), 'synthetic', self.dataset_name, 'synthetic.csv'),
                        os.path.join(os.getcwd(), self.synthetic_dir, 'synthetic.csv'),
                        os.path.join(self.base_dir, 'synthetic', self.dataset_name, 'synthetic.csv')
                    ]:
                        if os.path.exists(path):
                            synthetic_data = pd.read_csv(path)
                            break
                    else:
                        print(f"Could not find synthetic data file in any expected location")
                        return None
            
            return synthetic_data
        except subprocess.CalledProcessError as e:
            print(f"Error generating samples: {e}")
            
            # Try with absolute path
            try:
                abs_main_script = os.path.abspath(main_script)
                print(f"Trying with absolute path: {abs_main_script}")
                
                sample_cmd[1] = abs_main_script
                subprocess.run(sample_cmd, check=True)
                
                # Try to load the generated data
                try:
                    synthetic_data = pd.read_csv(rel_save_path)
                except:
                    synthetic_data = pd.read_csv(abs_save_path)
                
                print(f"Successfully generated {n_samples} samples with absolute path")
                return synthetic_data
            except Exception as e2:
                print(f"Error generating samples with absolute path: {e2}")
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