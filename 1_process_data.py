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
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Positional encodings for hour of the day and month
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Holiday dummy for German holidays
    de_holidays = holidays.Germany(years=data.index.year.unique().tolist())
    holiday_dates = pd.to_datetime(list(de_holidays.keys()))
    data['is_holiday'] = data.index.normalize().isin(holiday_dates).astype(int)
    
    # Dummies for specific market events
    data['DE_AT_split'] = (data.index >= '2018-10-01').astype(int)
    data['15_gran'] = (data.index >= '2025-10-01').astype(int)
    
    # Difference of Price
    data[f'{target_col}_diff'] = data[target_col].diff()
    
    # 2. Autoregressive Lags (Crucial for day-ahead prediction)
    # Assuming hourly resolution, we use 24h (yesterday), 48h, and 168h (one week ago)
    data['lag_1h'] = data[target_col].shift(1)
    data['lag_24h'] = data[target_col].shift(24)
    data['lag_48h'] = data[target_col].shift(48)
    data['lag_168h'] = data[target_col].shift(168)
    # Drop rows with NaN values resulting from the lag shift
    data.dropna(inplace=True)
    
    return data



if __name__ == "__main__":
    # Example usage:
    filepath = r"input/raw/4ApplicationDSEE_prices_DE_utc.csv"
    os.makedirs("input/processed", exist_ok=True)
    raw_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = add_engineer_features(raw_data)
    data.to_csv("input/processed/data.csv")
    
