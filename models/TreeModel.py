# src/model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict

class TreeModel:
    def __init__(self, config:dict):
        """
        Initializes the probabilistic forecaster.
        :param config: Dict of model kwargs

        """
        self.quantiles = config["quantiles"]
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.config = config
        self.pred_length = config["pred_length"]


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains a separate LightGBM model for each defined quantile.
        Dimension of X_train: (n_samples, n_features)
        Dimension of y_train: (n_samples,)
        """
        
        for alpha in self.quantiles:
            print(f"Training model for quantile: {alpha}")
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=alpha,
                **self.config
            )
            model.fit(X_train, y_train)
        self.models[alpha] = model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions for all trained quantiles.

        Returns: 
            predictions (np.ndarray): (len(X_test), len(quantiles))
        """
        predictions = np.ndarray(X_test.shape[0], len(self.quantiles))
        
        for i, alpha in enumerate(self.quantiles):
            model = self.models[alpha]
            predictions[:, i] = model.predict(X_test)
        
        return predictions


class TreeTrainer:
    def __init__(self, config: dict):
        """
        Wrapper for multi-step forecasting using TreeModel.
        """
        self.model = TreeModel(config)
        self.quantiles = config["quantiles"]
        self.pred_length = config["pred_length"]
        self.train_length = config["train_length"]

    def rolling_validation(self, X, y) -> np.ndarray:
        """
        Fits the model on pred_length observations and
        predict the next train_length observations.

        Returns: 
            predictions (np.ndarray): 2nd dimension is len(quantiles)
        """

        preds = []

        for start in range(0, len(X), self.pred_length):
            end = start + self.train_length
            test_end = end + self.pred_length

            if test_end > len(X):
                break

            X_train, y_train = X[start:end], y[start:end]
            X_test, y_test = X[end:test_end], y[end:test_end]
            
            self.model.train(X_train, y_train)
            predictions = self.model.predict(X_test)
            preds.append(predictions)

        return np.vstack(preds)