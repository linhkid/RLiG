"""
Test script for the TabularGAN (TabSyn) implementation

This script tests the implemented TabSyn integration on a small dataset
to verify that it correctly trains and generates synthetic data.
"""

import pandas as pd
import numpy as np
import os
import sys
from tabsyn.tabular_gan import TabularGAN

# Set random seed for reproducibility
np.random.seed(42)

# Load a small dataset for testing
print("Loading TicTacToe dataset...")
data_path = "train_data/TicTacToe_train_data.csv"
df = pd.read_csv(data_path)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Identify categorical columns
categorical_columns = []
for col in df.columns:
    if df[col].dtype == 'object' or len(df[col].unique()) < 10:
        categorical_columns.append(col)
print(f"Categorical columns: {categorical_columns}")

# Initialize TabularGAN with TabSyn
print("\nInitializing TabularGAN with TabSyn...")
try:
    model = TabularGAN(
        train_data=df,
        categorical_columns=categorical_columns,
        epochs=10,  # Use fewer epochs for testing
        verbose=True,
        random_seed=42
    )
    
    # Test if TabSyn modules are available
    import_success, missing_packages = model.test_imports()
    if not import_success:
        print(f"Missing packages: {missing_packages}")
        print("Will use fallback statistical approach")
    else:
        print("All required TabSyn imports are available")
        
except Exception as e:
    print(f"Error initializing TabularGAN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Train the model
print("\nFitting the model...")
model.fit()

# Generate synthetic data
print("\nGenerating synthetic data...")
n_samples = 100
syn_data = model.sample(n_samples)

# Display statistics about the original and synthetic data
print("\nOriginal data sample:")
print(df.head())
print("\nSynthetic data sample:")
print(syn_data.head())

# Compare distributions
print("\nDistribution of target variable:")
print("Original: ", df['target'].value_counts(normalize=True))
print("Synthetic:", syn_data['target'].value_counts(normalize=True))

# Compare column types
print("\nColumn types match:")
for col in df.columns:
    orig_type = df[col].dtype
    syn_type = syn_data[col].dtype
    print(f"{col}: {orig_type} vs {syn_type} - {'✓' if orig_type == syn_type else '✗'}")

print("\nTest completed!")