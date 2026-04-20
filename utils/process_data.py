# src/data_processor.py
import pandas as pd
import numpy as np
import os
import holidays
from typing import Tuple



def add_engineer_features(df: pd.DataFrame, target_col: str = 'Price') -> pd.DataFrame:
    """
    Creates time-based and autoregressive features required for day-ahead forecasting.
    """
    data = df.copy()

    # Remove NaNs from cols with daily freq / not enough values
    daily_freq_cols = data.filter(regex="Coal|Gas|Oil|EU").columns
    data[daily_freq_cols] = data[daily_freq_cols].ffill()
    data = data.dropna(axis=1)
    
    # 1. Calendar Features (Capturing seasonality and demand patterns)
    data['weekday'] = data.index.dayofweek
    data['hour'] = data.index.hour
    data['month'] = data.index.month
    
    weekday_dummies = pd.get_dummies(data['weekday'], prefix='weekday').astype(int)
    month_dummies = pd.get_dummies(data['month'], prefix='month').astype(int)
    data = pd.concat([data, weekday_dummies, month_dummies], axis=1)
    data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
    
    # Positional encodings for hour of the day and month (compatible with ML models)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    
    # Holiday dummy for German holidays
    de_holidays = holidays.country_holidays('DE')
    data['is_holiday'] = data.index.map(lambda x: x.date() in de_holidays).astype(int)
    
    # Dummies for specific market events
    data['DE_AT_split'] = (data.index >= '2018-10-01').astype(int)
    
    # Difference of Price
    data[f'{target_col}_diff'] = data[target_col].diff()

    return data


def make_dataset(data: pd.DataFrame, config:dict) -> Tuple[pd.DataFrame, pd.Series]:
    # 2. Autoregressive Lags (Crucial for day-ahead prediction)
    # Assuming hourly resolution, we use 24h (yesterday), 48h, and 168h (one week ago)
    lag_cols = []
    
    for lag in config["lags"]:
        lag_cols.append(f'lag_{lag}h')
        data[f'lag_{lag}h'] = data[config["target_col"]].shift(lag)

    data = data.dropna()
    
    X = data[config["features"] + lag_cols].values
    y = data[config["target_col"]].values
    
    return X, y




if __name__ == "__main__":
    # Example usage:
    filepath = r"input/raw/4ApplicationDSEE_prices_DE_utc.csv"
    os.makedirs("input/processed", exist_ok=True)
    raw_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = add_engineer_features(raw_data)
    data.to_csv("input/processed/data.csv")
    
