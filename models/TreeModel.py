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



    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
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

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for all trained quantiles.

        Returns: 
            predictions (pd.DataFrame): (len(X_test), len(quantiles))
        """
        # Ensure models have been trained
        if not self.models:
            raise RuntimeError("Models have not been trained yet. Please call .train() first.")

        predictions = np.zeros((X_test.shape[0], len(self.quantiles)))
        pred_cols = [f'pred_q{q}' for q in self.quantiles]
        
        for i, alpha in enumerate(self.quantiles):
            model = self.models[alpha]
            predictions[:, i] = model.predict(X_test)
            
        predictions = pd.DataFrame(predictions, 
                                   index=X_test.index, 
                                   columns=pred_cols)
        
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
        feat_imp = []

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
            feat_imp.append(model.feature_importances_)
            
        y_true = pd.concat(y_true).rename("true")
        y_pred = pd.concat(y_pred)   
        feat_importances = pd.DataFrame(np.vstack(feat_imp))
        feat_importances.columns = X.columns
        
        preds = y_true.to_frame().join(y_pred)

        return preds, feat_importances
       

    

if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse
    from collections.abc import Iterable
    from process_data import make_dataset
    # Example usage:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    ### optional: change some arguments ### 
    parser = argparse.ArgumentParser()
    for k0 in config["TreeModel"]:
        if k0 == "kwargs":
            for k1 in config["TreeModel"][k0]:
                parser.add_argument(f"--{k0}_{k1}", type=type(config["TreeModel"][k0][k1]), default=config["TreeModel"][k0][k1])
            continue
        # add argument as non-iterable or list
        parser.add_argument(f"--{k0}", type=type(config["TreeModel"][k0]), default=config["TreeModel"][k0])
        
    args = parser.parse_args()
    for arg0, value in vars(args).items(): 
        if arg0.startswith("kwargs"):
            arg1 = "_".join(arg0.split("_")[1:])
            config["TreeModel"]["kwargs"][arg1] = getattr(args, arg0)
        else:
            config["TreeModel"][arg0] = getattr(args, arg0)

    data = pd.read_csv("input/processed/data.csv", index_col=0, parse_dates=True)

    model = TreeModel(config["TreeModel"], quantiles=config["quantiles"])
    X, y = make_dataset(data, config["TreeModel"])
    preds, feat_importances = model.rolling_forecast(X, y)

    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"output/tree_based/{curr_time}"
    os.makedirs(output_dir, exist_ok=True)
    preds.to_csv(f"{output_dir}/predictions.csv")
    feat_importances.to_csv(f"{output_dir}/feature_importances.csv")
    yaml.safe_dump(config, open(f"{output_dir}/config.yaml", "w"))

    import matplotlib.pyplot as plt
    
    x_axis = preds.index # np.arange(len(preds))
    fig0, ax0 = plt.subplots(figsize=(12, 6), tight_layout=True)
    
    ax0.fill_between(x_axis, preds.iloc[:, 1].values, preds.iloc[:, -1].values, color="tab:blue", alpha=0.2, label="Prob. range")
    ax0.plot(x_axis, preds.iloc[:, 1:].values, color="tab:blue", ls="--")
    ax0.plot(x_axis, preds['true'].values, lw=2, label="True val")
    fig0.savefig("figures/tree_based_rolling_forecast.png")
    
    fig1, ax1 = plt.subplots(figsize=(12, 6), tight_layout=True)
    feat_importances.plot(kind="box", ax=ax1)
    ax1.set_title("Predictive feature importances")
    fig1.savefig("figures/tree_based_feature_importances.png")
    