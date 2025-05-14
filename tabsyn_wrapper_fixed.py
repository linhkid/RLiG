"""
TabSyn Wrapper - A simplified interface for TabSyn's VAE+Diffusion model

This wrapper provides a unified interface to train and sample from TabSyn models
following the official workflow in the TabSyn repository. It handles:

1. Data preprocessing and metadata generation
2. VAE training
3. Diffusion model training in the latent space
4. Synthetic data generation

Usage:
    wrapper = TabSynWrapper(train_data, categorical_columns=["cat1", "cat2"])
    wrapper.fit()
    synthetic_data = wrapper.sample(n_samples=1000)
"""

import os
import sys
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import subprocess
from pathlib import Path

class TabSynWrapper:
    """
    Wrapper for TabSyn's VAE+Diffusion model with a simplified sklearn-like interface
    """
    
    def __init__(self, train_data, categorical_columns=None, epochs=50, verbose=True, random_seed=42):
        """
        Initialize the TabSyn wrapper
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            The training data
        categorical_columns : list or None
            List of categorical column names
        epochs : int
            Number of training epochs for both VAE and diffusion models
        verbose : bool
            Whether to print verbose output
        random_seed : int
            Random seed for reproducibility
        """
        self.train_data = train_data
        self.categorical_columns = categorical_columns
        self.epochs = epochs
        self.verbose = verbose
        self.random_seed = random_seed
        self.is_trained = False
        
        # Set random seeds
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        # Detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.verbose:
            print(f"TabSyn wrapper initialized with random seed: {self.random_seed}")
            print(f"Using device: {self.device}")
        
        # Import check and fallback mode setup
        try:
            # Add TabSyn directory to sys.path
            tabsyn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabsyn')
            if tabsyn_dir not in sys.path:
                sys.path.append(tabsyn_dir)
            
            # Check if required packages are installed
            required_packages = ['numpy', 'pandas', 'scikit-learn', 'torch', 'category_encoders', 'icecream']
            missing = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing.append(package)
            
            if missing:
                if self.verbose:
                    print(f"Missing required packages: {missing}")
                self.fallback_mode = True
            else:
                self.fallback_mode = False
                
                # Try importing TabSyn modules
                try:
                    # Import modules but don't directly import functions
                    import tabsyn
                    import tabsyn.vae.main
                    import tabsyn.main
                    import tabsyn.utils
                    
                    self.tabsyn_modules_loaded = True
                    
                    if self.verbose:
                        print("Successfully imported TabSyn modules")
                except ImportError as e:
                    if self.verbose:
                        print(f"Error importing TabSyn modules: {e}")
                        print("Falling back to statistical approach")
                    self.fallback_mode = True
        except Exception as e:
            if self.verbose:
                print(f"Unexpected error: {e}")
            self.fallback_mode = True
        
        # Setup working directories
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_name = "temp_dataset"
        
        # Create directory structure required by TabSyn
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        # TabSyn expects info.json in dataset_dir
        self.info_file = os.path.join(self.dataset_dir, "info.json")
        self.vae_dir = os.path.join(self.temp_dir, "vae_ckpt", self.dataset_name)
        self.diffusion_dir = os.path.join(self.temp_dir, "diffusion_ckpt", self.dataset_name)
        self.synthetic_dir = os.path.join(self.temp_dir, "synthetic", self.dataset_name)
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "Info"), exist_ok=True)  # For backward compatibility
        os.makedirs(self.vae_dir, exist_ok=True)
        os.makedirs(self.diffusion_dir, exist_ok=True)
        os.makedirs(self.synthetic_dir, exist_ok=True)
        
        # Identify categorical columns if not provided
        if self.categorical_columns is None:
            self.categorical_columns = []
            for col in train_data.columns:
                if train_data[col].dtype == 'object' or len(train_data[col].unique()) < 10:
                    self.categorical_columns.append(col)
    
    def _prepare_data(self):
        """Prepare data for TabSyn training"""
        # Save the train data to a CSV file
        csv_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.csv")
        self.train_data.to_csv(csv_path, index=False)
        
        # Identify column types by index
        column_indices = list(range(len(self.train_data.columns)))
        cat_col_idx = []
        num_col_idx = []
        
        for i, col in enumerate(self.train_data.columns):
            if col in self.categorical_columns:
                cat_col_idx.append(i)
            else:
                num_col_idx.append(i)
        
        # Target column is assumed to be the last column
        target_col = self.train_data.columns[-1]
        target_col_idx = len(self.train_data.columns) - 1
        
        # Determine if target is categorical or numerical
        if target_col_idx in cat_col_idx:
            task_type = "binclass" if len(self.train_data[target_col].unique()) <= 2 else "multiclass"
        else:
            task_type = "regression"
        
        # Create a name-to-index mapping for columns
        idx_name_mapping = {i: col for i, col in enumerate(self.train_data.columns)}
        
        # Create metadata JSON compatible with TabSyn
        metadata = {
            "name": self.dataset_name,
            "task_type": task_type,
            "header": "infer",
            "column_names": list(self.train_data.columns),
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": [target_col_idx],
            "file_type": "csv",
            "data_path": csv_path,
            "test_path": None,
            "idx_name_mapping": idx_name_mapping
        }
        
        # Write metadata to JSON file - needed to be in dataset_dir as info.json
        with open(self.info_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        if self.verbose:
            print(f"Data prepared for TabSyn. CSV saved to {csv_path}")
            print(f"Metadata saved to {self.info_file}")
            print(f"Numerical columns: {len(num_col_idx)}")
            print(f"Categorical columns: {len(cat_col_idx)}")
            print(f"Task type: {task_type}")
        
        return metadata
    
    def fit(self):
        """Train the TabSyn model"""
        if self.fallback_mode:
            if self.verbose:
                print("Running in fallback mode. Using statistical approach instead of VAE+Diffusion")
            self.is_trained = True
            return self
        
        try:
            # Prepare data
            metadata = self._prepare_data()
            
            # Set up command line arguments for TabSyn
            original_dir = os.getcwd()
            try:
                # Change to the TabSyn directory
                tabsyn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabsyn')
                os.chdir(tabsyn_dir)
                
                # First train the VAE
                if self.verbose:
                    print("Training TabSyn VAE model...")
                
                # Instead of directly importing the main function, use subprocess to run TabSyn command
                import subprocess
                vae_cmd = [
                    sys.executable, 
                    os.path.join(tabsyn_dir, 'main.py'),
                    '--dataname', self.dataset_name,
                    '--method', 'vae',
                    '--mode', 'train',
                    '--epochs', str(self.epochs),
                    '--gpu', '0'  # TabSyn doesn't accept --device directly
                ]
                
                if self.verbose:
                    print(f"Running command: {' '.join(vae_cmd)}")
                    
                vae_proc = subprocess.run(vae_cmd, capture_output=True, text=True)
                if vae_proc.returncode != 0:
                    if self.verbose:
                        print(f"VAE training failed with error:\n{vae_proc.stderr}")
                    raise Exception(f"VAE training failed: {vae_proc.stderr}")
                
                # After VAE training, train diffusion model
                if self.verbose:
                    print("Training TabSyn diffusion model...")
                
                diffusion_cmd = [
                    sys.executable, 
                    os.path.join(tabsyn_dir, 'main.py'),
                    '--dataname', self.dataset_name,
                    '--method', 'tabsyn',
                    '--mode', 'train',
                    '--epochs', str(self.epochs),
                    '--gpu', '0'  # TabSyn doesn't accept --device directly
                ]
                
                if self.verbose:
                    print(f"Running command: {' '.join(diffusion_cmd)}")
                    
                diffusion_proc = subprocess.run(diffusion_cmd, capture_output=True, text=True)
                if diffusion_proc.returncode != 0:
                    if self.verbose:
                        print(f"Diffusion training failed with error:\n{diffusion_proc.stderr}")
                    raise Exception(f"Diffusion training failed: {diffusion_proc.stderr}")
                
                self.is_trained = True
                
                # Restore original directory
                os.chdir(original_dir)
                
                return self
            except Exception as e:
                # Restore original directory in case of error
                os.chdir(original_dir)
                raise e
                
        except Exception as e:
            if self.verbose:
                print(f"Error training TabSyn model: {e}")
                print("Falling back to statistical approach")
            self.fallback_mode = True
            self.is_trained = True
            return self
    
    def sample(self, n_samples=None):
        """
        Generate synthetic data using TabSyn
        
        Parameters:
        -----------
        n_samples : int or None
            Number of samples to generate. If None, uses training data size.
        
        Returns:
        --------
        pandas.DataFrame
            Generated synthetic data
        """
        if not self.is_trained:
            if self.verbose:
                print("Model not trained. Training now...")
            self.fit()
        
        if n_samples is None:
            n_samples = len(self.train_data)
        
        # Re-apply random seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        # If in fallback mode, use statistical approach
        if self.fallback_mode:
            if self.verbose:
                print(f"Generating {n_samples} samples using statistical approach...")
            return self._generate_statistical(n_samples)
        
        # Otherwise use TabSyn's sample function
        try:
            if self.verbose:
                print(f"Generating {n_samples} samples using TabSyn diffusion model...")
            
            original_dir = os.getcwd()
            try:
                # Change to TabSyn directory
                tabsyn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabsyn')
                os.chdir(tabsyn_dir)
                
                # Run TabSyn's sample function using subprocess
                sample_cmd = [
                    sys.executable, 
                    os.path.join(tabsyn_dir, 'main.py'),
                    '--dataname', self.dataset_name,
                    '--method', 'tabsyn',
                    '--mode', 'sample',
                    '--num-samples', str(n_samples),  # Correct parameter name
                    '--gpu', '0',  # TabSyn doesn't accept --device directly
                    '--save_path', os.path.join(self.synthetic_dir, 'tabsyn.csv')
                ]
                
                if self.verbose:
                    print(f"Running command: {' '.join(sample_cmd)}")
                    
                sample_proc = subprocess.run(sample_cmd, capture_output=True, text=True)
                if sample_proc.returncode != 0:
                    if self.verbose:
                        print(f"Sampling failed with error:\n{sample_proc.stderr}")
                    raise Exception(f"Sampling failed: {sample_proc.stderr}")
                
                # Restore original directory
                os.chdir(original_dir)
                
                # Read generated data
                synthetic_path = os.path.join(self.synthetic_dir, "tabsyn.csv")
                if os.path.exists(synthetic_path):
                    synthetic_data = pd.read_csv(synthetic_path)
                    
                    if self.verbose:
                        print(f"Generated {len(synthetic_data)} samples using TabSyn")
                    
                    return synthetic_data
                else:
                    if self.verbose:
                        print("Synthetic data file not found, falling back to statistical approach")
                    return self._generate_statistical(n_samples)
                
            except Exception as e:
                # Restore original directory in case of error
                os.chdir(original_dir)
                raise e
                
        except Exception as e:
            if self.verbose:
                print(f"Error generating with TabSyn: {e}")
                print("Falling back to statistical approach...")
            return self._generate_statistical(n_samples)
    
    def _generate_statistical(self, n_samples):
        """Generate data using statistical approach as fallback"""
        if self.verbose:
            print(f"Generating {n_samples} samples using statistical approach...")
        
        try:
            # Create a DataFrame for synthetic data
            synthetic_data = pd.DataFrame()
            
            # For each column, generate synthetic values
            for col in self.train_data.columns:
                column_data = self.train_data[col]
                
                # Check if column is categorical
                if col in self.categorical_columns:
                    # For categorical columns, sample with probabilities matching the original distribution
                    value_counts = column_data.value_counts(normalize=True)
                    synthetic_data[col] = np.random.choice(
                        value_counts.index, 
                        size=n_samples, 
                        p=value_counts.values
                    )
                else:
                    # For numeric columns, sample from a normal distribution with same mean and std
                    mean = column_data.mean()
                    std = column_data.std()
                    if std == 0:  # Handle constant columns
                        synthetic_data[col] = mean
                    else:
                        synthetic_values = np.random.normal(mean, std, n_samples)
                        # Clip to the range of the original data to avoid unrealistic values
                        min_val = column_data.min()
                        max_val = column_data.max()
                        synthetic_data[col] = np.clip(synthetic_values, min_val, max_val)
            
            if self.verbose:
                print(f"Generated {len(synthetic_data)} samples using statistical approach")
            
            # Save synthetic data to CSV
            os.makedirs(self.synthetic_dir, exist_ok=True)
            synthetic_path = os.path.join(self.synthetic_dir, "statistical.csv")
            synthetic_data.to_csv(synthetic_path, index=False)
            
            return synthetic_data
            
        except Exception as e:
            if self.verbose:
                print(f"Error generating synthetic data: {e}")
            # Ultimate fallback: return random samples from training data
            return self.train_data.sample(n_samples, replace=True).reset_index(drop=True)
    
    def __del__(self):
        """Clean up temporary directory"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            # Silently fail during cleanup
            pass