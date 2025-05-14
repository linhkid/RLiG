"""
Merge results from multiple subdirectories and calculate overall averages.

"""

import os
import pandas as pd
import numpy as np
import glob

def merge_results():
    """
    Merge accuracy results from all subdirectories in results/ folder
    and calculate average performance for each model.
    """
    results_dir = "results"
    output_merged = os.path.join(results_dir, "merged_results.csv")
    output_averages = os.path.join(results_dir, "model_averages.csv")
    output_details = os.path.join(results_dir, "detailed_results.csv")
    
    # Find all accuracy CSV files
    all_files = []
    for directory in glob.glob(f"{results_dir}/**/"):
        # Find all CSV files that have 'accuracy_results' in the name
        files = glob.glob(f"{directory}*accuracy_results*.csv")
        all_files.extend(files)
    
    if not all_files:
        print(f"No result files found in {results_dir} subdirectories")
        return
    
    print(f"Found {len(all_files)} result files for processing")
    
    # Read and merge all DataFrames
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file, index_col=0)
            # Get the dataset name from the parent folder if needed
            if df.index.size == 0 or df.index.isnull().any():
                # If index is empty or has NaN values, use directory name as dataset
                dataset = os.path.basename(os.path.dirname(file))
                df.index = [dataset] * df.shape[0]
            
            # Add to list
            dfs.append(df)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid DataFrames found to merge")
        return
    
    # Concatenate all files
    merged_df = pd.concat(dfs)
    
    # Get unique row indices to remove duplicates while preserving dataset names
    merged_df = merged_df.loc[~merged_df.index.duplicated(keep='first')]
    
    # Save merged results
    merged_df.to_csv(output_merged)
    print(f"Merged results saved to {output_merged}")
    
    # Calculate average of all AVG columns for each model
    avg_columns = [col for col in merged_df.columns if col.endswith('-AVG')]
    
    # Create a new DataFrame for model averages
    model_averages = {}
    
    # Calculate average for each AVG column
    for col in avg_columns:
        model = col.split('-')[0]  # Extract model name
        values = merged_df[col].dropna().values  # Drop NaN values
        if len(values) > 0:
            avg_value = np.mean(values)
            model_averages[model] = avg_value
    
    # Convert to DataFrame and sort by average score
    avg_df = pd.DataFrame({
        'Model': list(model_averages.keys()),
        'Average Score': list(model_averages.values())
    })
    avg_df = avg_df.sort_values('Average Score', ascending=False)
    
    # Save average scores
    avg_df.to_csv(output_averages, index=False)
    print(f"Model average scores saved to {output_averages}")
    
    # Print averages
    print("\nAverage performance across all datasets:")
    for _, row in avg_df.iterrows():
        print(f"{row['Model']}: {row['Average Score']:.4f}")
    
    # Create a more detailed breakdown by model+classifier
    # Extract all performance metrics (not just AVG)
    model_classifier_avg = {}
    
    # Parse all columns except those ending with '-AVG'
    for col in merged_df.columns:
        if col.endswith('-AVG'):
            continue
            
        # Extract model and classifier names
        parts = col.split('-')
        if len(parts) != 2:
            continue
            
        model, classifier = parts
        key = f"{model}-{classifier}"
        
        # Calculate average across all datasets
        values = merged_df[col].dropna().values
        if len(values) > 0:
            avg_value = np.mean(values)
            model_classifier_avg[key] = avg_value
    
    # Create a DataFrame with model-classifier combinations
    model_classifier_df = pd.DataFrame({
        'Model-Classifier': list(model_classifier_avg.keys()),
        'Average Score': list(model_classifier_avg.values())
    })
    model_classifier_df = model_classifier_df.sort_values('Average Score', ascending=False)
    
    # Save detailed results
    model_classifier_df.to_csv(output_details, index=False)
    print(f"Detailed model-classifier results saved to {output_details}")
    
    # Generate visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Set up plot style
        plt.style.use('ggplot')
        
        # Create model comparison bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(avg_df['Model'], avg_df['Average Score'], color='skyblue')
        plt.axhline(y=avg_df['Average Score'].mean(), color='red', linestyle='--', 
                   label=f'Overall Average: {avg_df["Average Score"].mean():.4f}')
        plt.xlabel('Model')
        plt.ylabel('Average Score')
        plt.title('Average Performance Across All Datasets')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)  # Scores are between 0 and 1
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300)
        print(f"Model comparison chart saved to {os.path.join(results_dir, 'model_comparison.png')}")
        
        # Create a detailed heatmap of all results if there are enough datasets
        if len(merged_df) >= 4:
            # Prepare data for heatmap
            pivot_df = merged_df[avg_columns].copy()
            
            # Create heatmap
            plt.figure(figsize=(12, len(pivot_df) * 0.5 + 2))
            ax = plt.gca()
            
            # Create heatmap with colorbar
            im = ax.imshow(pivot_df.values, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label='Score')
            
            # Set labels
            ax.set_xticks(np.arange(len(avg_columns)))
            ax.set_yticks(np.arange(len(pivot_df.index)))
            ax.set_xticklabels([col.split('-')[0] for col in avg_columns], rotation=45, ha='right')
            ax.set_yticklabels(pivot_df.index)
            
            # Add text annotations in cells
            for i in range(len(pivot_df.index)):
                for j in range(len(avg_columns)):
                    value = pivot_df.iloc[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center', color=color, fontsize=8)
            
            plt.title('Performance by Model and Dataset')
            plt.tight_layout()
            
            # Save the heatmap
            plt.savefig(os.path.join(results_dir, 'model_dataset_heatmap.png'), dpi=300)
            print(f"Model-dataset heatmap saved to {os.path.join(results_dir, 'model_dataset_heatmap.png')}")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization generation.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        
    # Create a formatted markdown report with summary statistics
    try:
        markdown_path = os.path.join(results_dir, "results_summary.md")
        with open(markdown_path, 'w') as md_file:
            md_file.write("# Model Evaluation Results Summary\n\n")
            
            md_file.write("## Overview\n\n")
            md_file.write(f"- Total datasets evaluated: {len(merged_df)}\n")
            md_file.write(f"- Total models compared: {len(avg_df)}\n")
            md_file.write(f"- Overall average performance: {avg_df['Average Score'].mean():.4f}\n\n")
            
            md_file.write("## Model Rankings\n\n")
            md_file.write("| Rank | Model | Average Score |\n")
            md_file.write("|------|-------|---------------|\n")
            
            for i, (_, row) in enumerate(avg_df.iterrows()):
                md_file.write(f"| {i+1} | {row['Model']} | {row['Average Score']:.4f} |\n")
            
            md_file.write("\n## Top Performing Model-Classifier Combinations\n\n")
            md_file.write("| Rank | Model-Classifier | Average Score |\n")
            md_file.write("|------|-----------------|---------------|\n")
            
            top_combinations = model_classifier_df.head(10)
            for i, (_, row) in enumerate(top_combinations.iterrows()):
                md_file.write(f"| {i+1} | {row['Model-Classifier']} | {row['Average Score']:.4f} |\n")
            
            md_file.write("\n## Dataset Performance\n\n")
            md_file.write("Best performing model for each dataset:\n\n")
            md_file.write("| Dataset | Best Model | Score |\n")
            md_file.write("|---------|------------|-------|\n")
            
            # Find best model per dataset
            for dataset in merged_df.index:
                best_score = -1
                best_model = ""
                
                for col in avg_columns:
                    score = merged_df.loc[dataset, col]
                    if score > best_score:
                        best_score = score
                        best_model = col.split('-')[0]
                
                md_file.write(f"| {dataset} | {best_model} | {best_score:.4f} |\n")
                
        print(f"Markdown summary report saved to {markdown_path}")
    except Exception as e:
        print(f"Error generating markdown report: {e}")

if __name__ == "__main__":
    merge_results()