"""
Analyze efficiency metrics by combining accuracy and time results.

This script:
1. Reads model_averages.csv for accuracy scores
2. Scans for time_results.csv files to calculate average execution times
3. Computes efficiency metrics (accuracy/time ratio)
4. Generates an efficiency report with model rankings
"""

import os
import pandas as pd
import numpy as np
import glob

def analyze_efficiency():
    """Analyze model efficiency by combining accuracy and time results"""
    results_dir = "results"
    output_efficiency = os.path.join(results_dir, "efficiency_metrics.csv")
    
    # Load model average accuracies
    try:
        accuracy_file = os.path.join(results_dir, "model_averages.csv")
        if not os.path.exists(accuracy_file):
            print(f"Accuracy results file {accuracy_file} not found. Please run merge_results.py first.")
            return
            
        accuracy_df = pd.read_csv(accuracy_file)
        model_accuracies = dict(zip(accuracy_df['Model'], accuracy_df['Average Score']))
        print(f"Loaded accuracy data for {len(model_accuracies)} models")
    except Exception as e:
        print(f"Error loading accuracy results: {e}")
        return
    
    # Find time results files
    time_files = []
    for directory in glob.glob(f"{results_dir}/**/"):
        files = glob.glob(f"{directory}*time_results*.csv")
        time_files.extend(files)
    
    if not time_files:
        print("No time result files found")
        return
    
    print(f"Found {len(time_files)} time result files")
    
    # Read and merge time DataFrames
    time_dfs = []
    for file in time_files:
        try:
            df = pd.read_csv(file, index_col=0)
            # Get dataset name from parent folder if needed
            if df.index.size == 0 or df.index.isnull().any():
                dataset = os.path.basename(os.path.dirname(file))
                df.index = [dataset] * df.shape[0]
            
            time_dfs.append(df)
            print(f"Processed time file: {file}")
        except Exception as e:
            print(f"Error reading time file {file}: {e}")
    
    if not time_dfs:
        print("No valid time data to process")
        return
    
    # Merge time results and calculate averages
    merged_time_df = pd.concat(time_dfs)
    merged_time_df = merged_time_df.loc[~merged_time_df.index.duplicated(keep='first')]
    
    # Calculate average execution time for each model
    model_times = {}
    for col in merged_time_df.columns:
        # Only include models we have accuracy for
        if col in model_accuracies:
            values = merged_time_df[col].dropna().values
            if len(values) > 0:
                avg_time = np.mean(values)
                model_times[col] = avg_time
    
    print(f"Calculated average execution times for {len(model_times)} models")
    
    # Calculate efficiency metrics (accuracy per second)
    efficiency_metrics = {}
    for model, avg_score in model_accuracies.items():
        if model in model_times and model_times[model] > 0:
            efficiency = avg_score / model_times[model]
            efficiency_metrics[model] = efficiency
    
    # Create DataFrame with all metrics and sort by efficiency
    efficiency_df = pd.DataFrame({
        'Model': list(efficiency_metrics.keys()),
        'Average Score': [model_accuracies[m] for m in efficiency_metrics.keys()],
        'Average Time (s)': [model_times[m] for m in efficiency_metrics.keys()],
        'Efficiency (Score/Time)': list(efficiency_metrics.values())
    })
    efficiency_df = efficiency_df.sort_values('Efficiency (Score/Time)', ascending=False)
    
    # Save results
    efficiency_df.to_csv(output_efficiency, index=False)
    print(f"Efficiency metrics saved to {output_efficiency}")
    
    # Print efficiency results
    print("\nEfficiency Rankings (Accuracy/Time):")
    for _, row in efficiency_df.iterrows():
        print(f"{row['Model']}: {row['Efficiency (Score/Time)']:.6f} (Score: {row['Average Score']:.4f}, Time: {row['Average Time (s)']:.2f}s)")
    
    # Add to markdown report if it exists
    markdown_path = os.path.join(results_dir, "results_summary.md")
    if os.path.exists(markdown_path):
        try:
            with open(markdown_path, 'a') as md_file:
                md_file.write("\n\n## Efficiency Metrics\n\n")
                md_file.write("| Rank | Model | Average Score | Average Time (s) | Efficiency (Score/Time) |\n")
                md_file.write("|------|-------|---------------|------------------|-------------------------|\n")
                
                for i, (_, row) in enumerate(efficiency_df.iterrows()):
                    md_file.write(f"| {i+1} | {row['Model']} | {row['Average Score']:.4f} | {row['Average Time (s)']:.2f} | {row['Efficiency (Score/Time)']:.6f} |\n")
                
            print(f"Efficiency metrics added to {markdown_path}")
        except Exception as e:
            print(f"Error updating markdown report: {e}")

if __name__ == "__main__":
    analyze_efficiency()