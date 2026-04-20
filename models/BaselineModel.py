# c:\Users\c.fusarbassini\Desktop\day_ahead_forecasting\models\LinearModel.py

import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from typing import Tuple, Dict
import yaml
from tqdm import tqdm
import os
from datetime import datetime


class LinearModel:
    """
    A class for probabilistic forecasting using a separate Linear Quantile Regressor
    for each specified quantile. This serves as a linear equivalent to the LinearModel.
    """
    def __init__(self, config: dict, quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]):
        """Initializes the probabilistic forecaster.
        
        :param config: A dictionary containing model parameters. It is expected
                       to have keys that are valid arguments for 
                       sklearn.linear_model.QuantileRegressor.
        :param quantiles: A list of floats for the quantiles to be predicted.
        """
        self.pred_length = config["pred_length"]
        self.train_length = config["train_length"]
        self.quantiles = quantiles
        self.models: Dict[float, QuantileRegressor] = {}
        # Separate QuantileRegressor-specific parameters from other config keys
        self.model_kwargs = config["model_kwargs"] if "model_kwargs" in config else {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains a separate QuantileRegressor model for each defined quantile.
        
        :param X_train: Training feature data (n_samples, n_features)
        :param y_train: Training target data (n_samples,)
        """
        for alpha in self.quantiles:
            # Note: For QuantileRegressor, 'quantile' is the quantile level, 
            # and 'alpha' is the L1 regularization penalty.
            model = QuantileRegressor(
                quantile=alpha,
                **self.model_kwargs
            )
            model.fit(X_train, y_train)
            self.models[alpha] = model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions for all trained quantiles.

        :param X_test: Test feature data.
        :return: A numpy array of predictions with shape (len(X_test), len(quantiles)).
        """
        if not self.models:
            raise RuntimeError("Models have not been trained yet. Please call .train() first.")

        predictions = np.zeros((X_test.shape[0], len(self.quantiles)))
        
        for i, alpha in enumerate(self.quantiles):
            model = self.models[alpha]
            predictions[:, i] = model.predict(X_test)
        
        return predictions

    def rolling_forecast(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs rolling window forecast.
        
        Fits the model on `train_length` observations and predicts the next 
        `pred_length` observations, sliding through the entire dataset.

        :param X: The complete feature dataset.
        :param y: The complete target dataset.
        :return: A tuple of (y_true, y_pred) arrays concatenated from all windows.
        """
        if not self.pred_length or not self.train_length:
            raise ValueError("pred_length and train_length must be defined in config for rolling_forecast.")

        y_pred = []
        y_true = []
        
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

        return np.concatenate(y_true).reshape(-1, 1), np.vstack(y_pred)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.process_data import make_dataset

    # Example usage:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    data = pd.read_csv("input/processed/data.csv", index_col=0, parse_dates=True)
    model = LinearModel(config['LinearModel'], quantiles=config["quantiles"])
    X, y = make_dataset(data, config['LinearModel'])
    y_true, y_pred = model.rolling_forecast(X, y)
    import matplotlib.pyplot as plt
    
    x_axis = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

    ax.plot(x_axis, y_true, lw=2, label="True val")
    ax.fill_between(x_axis, y_pred[:, 0], y_pred[:, -1], color="tab:blue", alpha=0.2, label="Prob. range")
    ax.plot(x_axis, y_pred, color="tab:blue", ls="--")
    fig.savefig("figures/baseline_rolling_forecast.png")

    pred_data = np.hstack([y_true, y_pred])
    pred_cols = ['true'] + [f'pred_q{q}' for q in model.quantiles]
    preds = pd.DataFrame(pred_data, columns=pred_cols)
    # Note: Reconstructing the exact datetime index for rolling validation is complex
    # and depends on train/pred lengths. For simplicity, we save it with a range index.
    os.makedirs("output/baseline", exist_ok=True)
    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    preds.to_csv(f"output/baseline/{curr_time}_predictions.csv")
    yaml.safe_dump(config, open(f"output/baseline/{curr_time}_config.yaml", "w"))
