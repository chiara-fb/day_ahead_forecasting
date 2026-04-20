import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Tuple
from tqdm import tqdm
from utils.losses import pinball_loss
import pandas as pd

class FeatureSelector:

    def __init__(self, model, config):
        
        self.model = model(config, quantiles=[0.5])
        self.column_names = np.array(
            config["features"] + 
            [f"lag_{i}" for i in config["lags"]]
            )
    
    
    def rfe_with_correlation(self, X, y, corr_threshold:float, verbose:bool=True) -> Tuple[np.ndarray, float]:
        y0_true, y0_pred, feat_importances = self.model.rolling_forecast(X, y)
        best_err = pinball_loss(y0_true, y0_pred, 0.5)
        mean_feat_importances = np.mean(feat_importances, axis=0)
        sorted_features = np.argsort(mean_feat_importances)
        N = len(sorted_features)
        best_features = self.column_names[sorted_features]
        
        if verbose:
            print("Sorted features: ", best_features)

        corr = np.corrcoef(X[:, sorted_features], rowvar=False)
        try:
            distance = 1 - np.abs(corr)
            np.fill_diagonal(distance, 0)
            linkage_matrix = linkage(squareform(distance), method='average')
            clusters = fcluster(linkage_matrix, t=1-corr_threshold, criterion='distance')
        except Exception as e:
            print("Correlation matrix is not symmetric. Correlation-based elimination deactivated.")
            clusters = np.arange(N)

        keep_features = np.ones_like(sorted_features, dtype=bool)
        
        for i in tqdm(range(N)):
            feat_cluster = sorted_features[(clusters == clusters[i]) & (keep_features)]
            other_features = sorted_features[(clusters != clusters[i]) & (keep_features)]
            
            if len(feat_cluster) > 1:
                # keep only the feature of the cluster with the highest feat importance 
                test_feat = np.append(other_features, feat_cluster[-1]) 
                drop = [sorted_features.tolist().index(l) for l in feat_cluster[:-1]]
            else:
                test_feat = other_features
                drop = i
            
            y_true, y_pred, _ = self.model.rolling_forecast(X[:,test_feat], y)
            err = pinball_loss(y_true, y_pred, 0.5)
            
            if err < best_err:
                best_err = err
                keep_features[drop] = False
                best_features = self.column_names[sorted_features[keep_features]]
                if verbose:
                    print(f"New best error {best_err:.4f} with features: {best_features}")
        
        return best_features, best_err
            

if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from process_data import make_dataset
    from models.TreeModel import TreeModel
    import yaml
    from datetime import datetime

    # Example usage:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    tree_config = config["TreeModel"]
    data = pd.read_csv("input/processed/data.csv", index_col=0, parse_dates=True)

    feature_selector  = FeatureSelector(TreeModel, tree_config)
    X, y = make_dataset(data, tree_config)
    best_err, best_feat = feature_selector.rfe_with_correlation(X,y, 0.8)
    print("Best err", best_err)
    print("Selected features", best_feat)
    
    tree_config["features"] = [f for f in best_feat if not f.startwith("lag_")]
    tree_config["lags"] = [int(f.split("_")[1]) for f in best_feat if f.startwith("lag_")]
    best_model = TreeModel(tree_config, quantiles=config["quantiles"])
    best_X, best_y = make_dataset(data, tree_config)
    y_true, y_pred, feat_importances = best_model.rolling_forecast(best_X, best_y)
    
    pred_data = np.hstack([y_true, y_pred])
    pred_cols = ['true'] + [f'pred_q{q}' for q in best_model.quantiles]
    preds = pd.DataFrame(pred_data, columns=pred_cols)
    # Note: Reconstructing the exact datetime index for rolling validation is complex
    # and depends on train/pred lengths. For simplicity, we save it with a range index.
    
    os.makedirs(f"output/tree_based/best_features", exist_ok=True)
    preds.to_csv(f"output/tree_based/best_features/predictions.csv")
    feat_importances = pd.DataFrame(feat_importances, columns=best_feat)
    feat_importances.to_csv(f"output/tree_based/best_features/feature_importances.csv")