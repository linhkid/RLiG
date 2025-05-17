import os
import numpy as np
import pandas as pd


# Function to save data to CSV
def save_to_csv(X, y, save_dir, filename):
    # Convert X to DataFrame
    if isinstance(X, pd.DataFrame):
        # X is already a DataFrame
        df_X = X.copy()
    else:
        # X is a numpy array
        df_X = pd.DataFrame(X)

    # Handle y - flatten if it's 2D
    if isinstance(y, pd.Series):
        # y is already a Series
        df_y = y.copy()
    else:
        # Convert y to 1D if it's 2D
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] == 1:
            y = y.flatten()  # This flattens a 2D array with 1 column to 1D

        # Now create Series
        df_y = y

    # Rename y column
    df_y.name = "Class"  # Change this to your target column name

    # Combine X and y
    combined_df = pd.concat([df_X, df_y], axis=1)

    # Save to CSV
    file_path = os.path.join(save_dir, filename)
    combined_df.to_csv(file_path, index=False)
    print(f"Saved {filename} to {file_path}")