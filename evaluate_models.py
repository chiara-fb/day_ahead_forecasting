import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.losses import average_pinball_loss, coverage_within_range

def find_latest_prediction_files(output_root="output"):
    """Finds the latest prediction file for each model type."""
    root = Path(output_root)
    if not root.exists():
        return {}
    
    # Find all prediction files
    pred_files = list(root.glob("**/predictions.csv"))
    
    # Group by a representative model name
    model_files = {"linear": [], "tree_based": [], "chronos": []}

    for f in pred_files:
        # this could "linear", "tree_based", "chronos"
        model_name = f.parts[1]
        model_files[model_name].append(f)
        
    # Find the latest file for each model group
    latest_files = {}
    for model_name, files in model_files.items():
        # get the file which was updated most recently
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        latest_files[model_name] = latest_file
    return latest_files

def evaluate_model(pred_df: pd.DataFrame) -> dict:
    """Calculates all evaluation metrics for a given prediction DataFrame."""
    
    if 'true' not in pred_df.columns:
        raise ValueError("'true' column not found in prediction DataFrame.")

    true_col = pred_df[['true']]
    pred_cols = pred_df.drop(columns='true')
    
    if pred_cols.empty:
        return {
            'Avg Pinball Loss': np.nan,
            'Coverage Within Range': np.nan,
        }

    quantiles = [float(c.split('_q')[1]) for c in pred_cols.columns]
    sorted_cols = [x for _, x in sorted(zip(quantiles, pred_cols.columns))]
    pred_df_sorted = pd.concat([true_col, pred_cols[sorted_cols]], axis=1)

    metrics = {}
    metrics['Avg Pinball Loss'] = average_pinball_loss(pred_df_sorted).mean()
    metrics['Coverage Within Range'] = coverage_within_range(pred_df_sorted).mean()
    
    return metrics


if __name__ == "__main__":

    output_root = "output"
    latest_files = find_latest_prediction_files(output_root)
    
    if not latest_files:
        print("No prediction files found in 'output' directory. Run models to generate predictions first.")

    all_metrics = []

    for model_name, pred_file in latest_files.items():
        print(f"Evaluating {model_name} from {pred_file}...")
        pred_df = pd.read_csv(pred_file, index_col=0)
        
        all_metrics.append({'Model': model_name, **evaluate_model(pred_df)})

    if not all_metrics:
        print("Evaluation could not be completed for any model.")
        
    summary_df = pd.DataFrame(all_metrics).set_index('Model')
    summary_df.to_csv(f"evaluation_summary.csv")
    print(f"\nSummary saved to {output_root}/evaluation_summary.csv")

    os.makedirs("figures", exist_ok=True)

