# src/model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Tuple, Dict
import yaml
from tqdm import tqdm
import os
from datetime import datetime

class TreeModel:
    def __init__(self, config:dict, quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]):
        """Initializes the probabilistic forecaster.
        
        :param config: A dictionary containing model parameters. It is expected
                       to have a 'quantiles' key with a list of floats, and
                       other keys should be valid arguments for lgb.LGBMRegressor.
        """
        self.pred_length = config["pred_length"]
        self.train_length = config["train_length"]
        self.quantiles = quantiles
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        # Separate LGBM-specific parameters from other config keys
        self.model_kwargs = config["model_kwargs"] if "model_kwargs" in config else {}



    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains a separate LightGBM model for each defined quantile.
        Dimension of X_train: (n_samples, n_features)
        Dimension of y_train: (n_samples,)
        """
        # Suppress common LightGBM warnings (e.g., about num_threads)
        # during the model fitting process.
        
        for alpha in self.quantiles:
            print(f"Training model for quantile: {alpha}")
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=alpha,
                verbose=-1,
                **self.model_kwargs
            )
            model.fit(X_train, y_train)
            self.models[alpha] = model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions for all trained quantiles.

        Returns: 
            predictions (np.ndarray): (len(X_test), len(quantiles))
        """
        # Ensure models have been trained
        if not self.models:
            raise RuntimeError("Models have not been trained yet. Please call .train() first.")

        predictions = np.zeros((X_test.shape[0], len(self.quantiles)))
        
        for i, alpha in enumerate(self.quantiles):
            model = self.models[alpha]
            predictions[:, i] = model.predict(X_test)
        
        return predictions


    def rolling_forecast(self, X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits the model on pred_length observations and
        predict the next train_length observations.

        Returns: 
            predictions (np.ndarray): 2nd dimension is len(quantiles)
        """

        y_pred = []
        y_true = []
        feature_importance = []

        for start in tqdm(range(0, len(X), self.pred_length)):
            end = start + self.train_length
            test_end = end + self.pred_length

            if test_end > len(X):
                break

            X_train, y_train = X[start:end], y[start:end]
            X_test, y_test = X[end:test_end], y[end:test_end]
            
            self.train(X_train, y_train)
            predictions = self.predict(X_test)
            y_pred.append(predictions)
            y_true.append(y_test)

            med_q = self.quantiles[len(self.quantiles) // 2]
            model = self.models.get(0.5, self.models[med_q])
            feature_importance.append(model.feature_importances_)
            
        return np.concatenate(y_true).reshape(-1, 1), np.vstack(y_pred), np.vstack(feature_importance)

    

if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from process_data import make_dataset
    # Example usage:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    tree_config = config["TreeModel"]
    data = pd.read_csv("input/processed/data.csv", index_col=0, parse_dates=True)
    col_names = np.array(tree_config["features"] + [f"lag_{i}" for i in tree_config["lags"]])

    model = TreeModel(tree_config, quantiles=config["quantiles"])
    X, y = make_dataset(data, tree_config)
    y_true, y_pred, feat_importances = model.rolling_forecast(X, y)
    import matplotlib.pyplot as plt
    
    x_axis = np.arange(len(y_true))
    fig0, ax0 = plt.subplots(figsize=(12, 6), tight_layout=True)

    ax0.plot(x_axis, y_true, lw=2, label="True val")
    ax0.fill_between(x_axis, y_pred[:, 0], y_pred[:, -1], color="tab:blue", alpha=0.2, label="Prob. range")
    ax0.plot(x_axis, y_pred, color="tab:blue", ls="--")
    fig0.savefig("figures/tree_based_rolling_forecast.png")

    pred_data = np.hstack([y_true, y_pred])
    pred_cols = ['true'] + [f'pred_q{q}' for q in model.quantiles]
    preds = pd.DataFrame(pred_data, columns=pred_cols)
    # Note: Reconstructing the exact datetime index for rolling validation is complex
    # and depends on train/pred lengths. For simplicity, we save it with a range index.
    
    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(f"output/tree_based/{curr_time}", exist_ok=True)
    preds.to_csv(f"output/tree_based/{curr_time}/predictions.csv")
    feat_importances = pd.DataFrame(feat_importances, columns=col_names)
    feat_importances.to_csv(f"output/tree_based/{curr_time}/feature_importances.csv")

    fig1, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
    feat_importances.plot(kind="box", ax=ax1)
    ax1.set_title("Predictive feature importances")
    fig1.savefig("figures/tree_based_feature_importances.png")


    yaml.safe_dump(config, open(f"output/tree_based/{curr_time}/config.yaml", "w"))