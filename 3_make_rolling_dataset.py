import pandas as pd
import numpy as np
from typing import Generator, Tuple

class RollingWindow:
    def __init__(self, train_size: int, pred_size: int):
        """
        Initializes the rolling window cross-validation splitter.
        
        :param train_length: Number of time steps (rows) for the training set.
        :param pred_length: Number of time steps (rows) for the validation set.
        :param lookback: Number of historical steps to prepend to the validation set. 
                         Essential for sequence models (Encoder-Decoder) to generate the 
                         initial input sequences for the validation phase.
        """
        self.train_size = train_size
        self.pred_size = pred_size
        self.step_size = step_size
        self.lookback = lookback

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yields train and validation DataFrames for each rolling fold.
        """
        total_len = len(df)
        start_idx = 0

        while start_idx + self.train_size + self.val_size <= total_len:
            train_end = start_idx + self.train_size
            val_end = train_end + self.val_size

            # 1. Train split
            train_df = df.iloc[start_idx:train_end]

            # 2. Validation split
            # - Tree models use lookback=0.
            # - Encoder-Decoders use lookback=sequence_length to form their first input context.
            val_start = train_end - self.lookback
            if val_start < 0:
                val_start = 0
                
            val_df = df.iloc[val_start:val_end]
            
            # For Encoder-Decoders, you might want to mark which part of val_df is strictly 
            # history (lookback) vs the actual prediction horizon. You can handle this in 
            # your Dataset/DataLoader class based on the length.

            yield train_df, val_df

            start_idx += self.step_size
