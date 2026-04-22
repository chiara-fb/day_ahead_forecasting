# c:\Users\c.fusarbassini\Desktop\day_ahead_forecasting\models\ChronosModel.py

import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from typing import Tuple
import yaml
from tqdm import tqdm
import os
from datetime import datetime

class ChronosModel:
    """
    A class for probabilistic forecasting using Amazon's Chronos foundation model.
    Chronos is a zero-shot model, so it doesn't require explicit training, 
    but uses the recent history (context) to predict the future.
    """
    def __init__(self, config: dict, quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]):
        """Initializes the probabilistic forecaster.
        
        :param config: A dictionary containing model parameters.
        :param quantiles: A list of floats for the quantiles to be predicted.
        """
        self.pred_length = config["pred_length"]
        self.train_length = config["train_length"]
        self.features = config["features"]
        self.target_col = config["target_col"]
        self.quantiles = quantiles
        
        # Chronos-specific configuration
        self.model_name = config["kwargs"].get("model_name", "amazon/chronos-2")
        self.num_samples = config["kwargs"].get("num_samples", 20)
        
        print(f"Loading Chronos model: {self.model_name}")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.context = None


    def rolling_forecast(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Performs rolling window forecast.
        """
        if not self.pred_length or not self.train_length:
            raise ValueError("pred_length and train_length must be defined in config for rolling_forecast.")

        y_pred = []
        y_true = []
        df = data.copy().reset_index()
        df["id"] = 0
        
        for start in tqdm(range(0, len(df), self.pred_length)):
            end = start + self.train_length
            test_end = end + self.pred_length

            if test_end > len(df):
                break
            
            context = df[["id", "DateTime"] + self.features + [self.target_col]].iloc[start:end]
            future = df[["id", "DateTime"] + self.features].iloc[end:test_end]
            true = df.set_index("DateTime")[self.target_col].iloc[end:test_end]
            
            y_true.append(true)
            
            pred = self.pipeline.predict_df(
                    context,
                    future_df=future,
                    prediction_length=self.pred_length,  # Number of steps to forecast
                    quantile_levels=self.quantiles,  # Quantile for probabilistic forecast
                    id_column="id",  # Column identifying different time series
                    timestamp_column="DateTime",  # Column with datetime information
                    target=self.target_col,  # Column(s) with time series values to predict
                )
            
            q_names = [f'pred_q{q}' for q in self.quantiles]
            pred = pred.rename(columns={str(q): q_name for q, q_name in zip(self.quantiles, q_names)})
            pred = pred.set_index("DateTime")[q_names]
            y_pred.append(pred)
        
        y_true = pd.concat(y_true).rename("true")
        y_pred = pd.concat(y_pred)
        preds = y_true.to_frame().join(y_pred)

        return preds


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from process_data import make_dataset

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create an initial configuration template for Chronos if not found
    chronos_config = config["ChronosModel"]
    
    data = pd.read_csv("input/processed/data.csv", index_col=0, parse_dates=True)
    
    model = ChronosModel(chronos_config, quantiles=config["quantiles"])
    X, y = make_dataset(data, chronos_config)
    preds = model.rolling_forecast(pd.concat([X, y], axis=1))
    
    import matplotlib.pyplot as plt
    
    x_axis = preds.index
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

    ax.plot(x_axis, preds['true'].values, lw=2, label="True val")
    ax.fill_between(x_axis, preds.iloc[:, 1].values, preds.iloc[:, -1], color="tab:purple", alpha=0.2, label="Prob. range")
    ax.plot(x_axis, preds.iloc[:, 1:].values, color="tab:purple", ls="--")
    ax.set_title("Chronos Rolling Forecast")
    
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/chronos_rolling_forecast.png")
    
    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"output/chronos/{curr_time}"
    os.makedirs(output_dir, exist_ok=True)
    preds.to_csv(f"{output_dir}/predictions.csv")
    yaml.safe_dump(config["ChronosModel"], open(f"{output_dir}/config.yaml", "w"))
