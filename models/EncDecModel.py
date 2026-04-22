import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import yaml
from tqdm import tqdm
import os
from datetime import datetime

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden, cell):
        # x shape: [batch, seq_len, input_dim] (future known features)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        # outputs shape: [batch, seq_len, hidden_dim]
        preds = self.fc(outputs) 
        # preds shape: [batch, seq_len, output_dim]
        return preds

class EncoderDecoder(nn.Module):
    def __init__(self, enc_input_dim, dec_input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.encoder = Encoder(enc_input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(dec_input_dim, hidden_dim, output_dim, num_layers)
        
    def forward(self, enc_x, dec_x):
        hidden, cell = self.encoder(enc_x)
        preds = self.decoder(dec_x, hidden, cell)
        return preds

class TorchEncoderDecoderModel:
    """
    A class for probabilistic forecasting using a PyTorch LSTM Encoder-Decoder architecture.
    It outputs specific quantiles by optimizing the pinball loss.
    """
    def __init__(self, config: dict, quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]):
        self.test_length = config.get("test_length")
        self.train_length = config.get("train_length")
        self.pred_length = config.get("pred_length")
        self.quantiles = quantiles
        
        # Seq2Seq specific configuration
        self.context_length = config.get("context_length", 168)  # Lookback window for the encoder
        self.hidden_dim = config.get("hidden_dim", 64)
        self.num_layers = config.get("num_layers", 1)
        self.lr = config.get("lr", 1e-3)
        self.epochs = config.get("epochs", 15)
        self.batch_size = config.get("batch_size", 32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Scalers to help neural network convergence
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def _create_sequences(self, X: np.ndarray, y: np.ndarray):
        """Slides over the training dataset to produce sequence batches."""
        enc_x, dec_x, target = [], [], []
        for i in range(len(X) - self.context_length - self.pred_length + 1):
            # Encoder uses past features AND past target
            past_x = X[i : i + self.context_length]
            past_y = y[i : i + self.context_length].reshape(-1, 1)
            enc_x.append(np.concatenate([past_x, past_y], axis=1))
            
            # Decoder uses future known features
            dec_x.append(X[i + self.context_length : i + self.context_length + self.pred_length])
            # Target to predict
            target.append(y[i + self.context_length : i + self.context_length + self.pred_length])
            
        return np.array(enc_x), np.array(dec_x), np.array(target)

    def quantile_loss(self, preds: torch.Tensor, target: torch.Tensor):
        """Computes average pinball loss across all quantiles."""
        loss = 0
        target = target.unsqueeze(-1)  # [batch, seq_len, 1]
        for i, q in enumerate(self.quantiles):
            error = target[:, :, 0] - preds[:, :, i]
            loss += torch.max((q - 1) * error, q * error).mean()
        return loss

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Initializes and trains the Seq2Seq model using Pinball Loss."""
        X_scaled = self.scaler_x.fit_transform(X_train.values)
        y_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        enc_input_dim = X_scaled.shape[1] + 1  # features + 1 target col
        dec_input_dim = X_scaled.shape[1]      # only features
        output_dim = len(self.quantiles)
        
        self.model = EncoderDecoder(
            enc_input_dim, dec_input_dim, self.hidden_dim, output_dim, self.num_layers
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        enc_x, dec_x, target = self._create_sequences(X_scaled, y_scaled)
        if len(enc_x) == 0:
            raise ValueError("Not enough data to create sequences. Check train_length/context_length/pred_length.")
            
        dataset = TensorDataset(torch.FloatTensor(enc_x), torch.FloatTensor(dec_x), torch.FloatTensor(target))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_enc_x, batch_dec_x, batch_target in loader:
                batch_enc_x = batch_enc_x.to(self.device)
                batch_dec_x = batch_dec_x.to(self.device)
                batch_target = batch_target.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_enc_x, batch_dec_x)
                loss = self.quantile_loss(preds, batch_target)
                loss.backward()
                optimizer.step()

    def predict(self, X_context: pd.DataFrame, y_context: pd.Series, X_test: pd.DataFrame) -> pd.DataFrame:
        """Predicts the target quantiles given the most recent context and upcoming features."""
        self.model.eval()
        with torch.no_grad():
            X_context_scaled = self.scaler_x.transform(X_context.values)
            y_context_scaled = self.scaler_y.transform(y_context.values.reshape(-1, 1)).flatten()
            X_test_scaled = self.scaler_x.transform(X_test.values)
            
            enc_x = np.concatenate([X_context_scaled, y_context_scaled.reshape(-1, 1)], axis=1)
            enc_x = torch.FloatTensor(enc_x).unsqueeze(0).to(self.device)  # Add batch dim
            
            dec_x = torch.FloatTensor(X_test_scaled).unsqueeze(0).to(self.device)
            
            preds = self.model(enc_x, dec_x)  # [1, pred_length, num_quantiles]
            preds = preds[0].cpu().numpy()    # Drop batch dim: [pred_length, num_quantiles]
            
        # Inverse transform scaling predictions back to original domain
        preds_inv = self.scaler_y.inverse_transform(preds)
            
        pred_cols = [f'pred_q{q}' for q in self.quantiles]
        return pd.DataFrame(preds_inv, index=X_test.index, columns=pred_cols)

    def rolling_forecast(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Performs the rolling window evaluation over the entire test dataset."""
        if not self.pred_length or not self.train_length:
            raise ValueError("pred_length and train_length must be defined in config.")

        y_pred = []
        y_true = []
        
        for start in tqdm(range(0, len(X), self.test_length)):
            end = start + self.train_length
            test_end = end + self.pred_length

            if test_end > len(X):
                break

            X_train, y_train = X.iloc[start:end], y.iloc[start:end]
            X_test, y_test = X.iloc[end:test_end], y.iloc[end:test_end]
            
            self.train(X_train, y_train)
            
            # Gather the required lookback context immediately preceding the test window
            X_context = X_train.iloc[-self.context_length:]
            y_context = y_train.iloc[-self.context_length:]
            
            predictions = self.predict(X_context, y_context, X_test)
            y_pred.append(predictions)
            y_true.append(y_test)

        y_true = pd.concat(y_true).rename("true")
        y_pred = pd.concat(y_pred)
        preds = y_true.to_frame().join(y_pred)

        return preds


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from process_data import make_dataset
    import matplotlib.pyplot as plt
    from collections.abc import Iterable
    import argparse


    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    ### optional: change some arguments ### 
    parser = argparse.ArgumentParser()
    for k0 in config["EncDecModel"]:
        if k0 == "kwargs":
            for k1 in config["EncDecModel"][k0]:
                parser.add_argument(f"--{k0}_{k1}", type=type(config["EncDecModel"][k0][k1]), default=config["EncDecModel"][k0][k1])
            continue
        # add argument as non-iterable or list
        parser.add_argument(f"--{k0}", type=type(config["EncDecModel"][k0]), default=config["EncDecModel"][k0])
        
    args = parser.parse_args()
    for arg0, value in vars(args).items(): 
        if arg0.startswith("kwargs"):
            arg1 = "_".join(arg0.split("_")[1:])
            config["EncDecModel"]["kwargs"][arg1] = getattr(args, arg0)
        else:
            config["EncDecModel"][arg0] = getattr(args, arg0)
    


    
    data = pd.read_csv("input/processed/data.csv", index_col=0, parse_dates=True)
    
    model = TorchEncoderDecoderModel(config["EncDecModel"], quantiles=config["quantiles"])
    X, y = make_dataset(data, config["EncDecModel"])
    preds = model.rolling_forecast(X, y)

    curr_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"output/encdec/{curr_time}"
    os.makedirs(output_dir, exist_ok=True)
    preds.to_csv(f"{output_dir}/predictions.csv")
    yaml.safe_dump(config["EncDecModel"], open(f"{output_dir}/config.yaml", "w"))

    x_axis = preds.index
    fig0, ax0 = plt.subplots(figsize=(12, 6), tight_layout=True)
    ax0.fill_between(x_axis, preds.iloc[:, 1].values, preds.iloc[:, -1].values, color="tab:red", alpha=0.2, label="Prob. range")
    ax0.plot(x_axis, preds.iloc[:, 1:].values, color="tab:red", ls="--")
    ax0.plot(x_axis, preds['true'].values, lw=2, label="True val")
    os.makedirs("figures", exist_ok=True)
    fig0.savefig("figures/encdec_rolling_forecast.png")