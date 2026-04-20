# src/visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

class DataVisualizer:
    def __init__(self, data: pd.DataFrame, target_col: str = 'Price'):
        """
        Initializes the visualizer with the dataset.
        
        :param data: Processed DataFrame containing features and the target variable.
        :param target_col: The name of the day-ahead price column.
        """
        self.data = data
        self.target_col = target_col
        # Set visualization style
        sns.set_theme(style="whitegrid")

    def plot_target(self):

        fig, ax = plt.subplots(1, figsize=(14, 5))
        self.data[self.target_col].plot(ax=ax)
        return fig, ax

    def show_generic_statistics(self):
        """Prints generic descriptive statistics and plots distributions."""
        
        # Plot distributions for the target and a few key features
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(self.data[self.target_col], bins=50, kde=True, ax=axes[0], color='blue')
        axes[0].set_title(f'Distribution of {self.target_col}')
        axes[0].set_xlabel('Price (€/MWh)')
        
        sns.boxplot(x=self.data.index.hour, y=self.data[self.target_col], ax=axes[1], palette='viridis')
        axes[1].set_title(f'{self.target_col} Distribution by Hour of Day')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Price (€/MWh)')
        
        return fig, axes

    def plot_feature_correlations(self):
        """Plots a heatmap of correlations between all available features."""
        corr_matrix = self.data.corr()
        
        fig, axes = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True, 
            square=True,
            linewidths=0.5
        )
        plt.title('Feature Correlation Heatmap')
        return fig, axes

    def plot_lag_correlations(self, max_lag: int = 168):
        """
        Calculates and plots a heatmap of correlations between the target variable 
        and the lags of all features up to max_lag.
        """
        print(f"Calculating correlations up to lag {max_lag}...")
        
        lags = list(range(1, max_lag + 1))
        features = self.data.columns
        
        # Initialize a dataframe to hold the correlation matrix
        corr_matrix = pd.DataFrame(index=features, columns=lags, dtype=float)
        
        for feature in features:
            for lag in lags:
                corr = self.data[self.target_col].corr(self.data[feature].shift(lag))
                corr_matrix.loc[feature, lag] = corr
                
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, 
            cmap='coolwarm', 
            cbar=True,
            ax=ax, 
            xticklabels=max(1, max_lag // 20)  # Skips ticks to prevent overcrowding
        )
        
        ax.set_title(f'Correlation of {self.target_col} with Lagged Features (up to {max_lag}h)')
        ax.set_xlabel('Features')
        ax.set_ylabel('Lag (Hours)')
        ax.invert_yaxis()
        return fig, ax

    def plot_autoregressivity(self, lags: int = 168):
        """
        Plots the Autocorrelation Function (ACF) and 
        Partial Autocorrelation Function (PACF) to visualize autoregressivity.
        """
        # Drop NaNs to ensure statsmodels functions correctly
        series = self.data[self.target_col].dropna()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), tight_layout=True)
        
        # Autocorrelation
        plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'Autocorrelation Function (ACF) of {self.target_col}')
        axes[0].set_xlabel('Lags (Hours)')
        axes[0].set_ylabel('Autocorrelation')
        
        # Partial Autocorrelation
        # PACF shows the direct effect of a lag, removing the effect of intermediate lags
        plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].set_title(f'Partial Autocorrelation Function (PACF) of {self.target_col}')
        axes[1].set_xlabel('Lags (Hours)')
        axes[1].set_ylabel('Partial Autocorrelation')
        
        return fig, axes

if __name__ == "__main__":
    # Example usage:

    filepath = r"input/processed/data.csv"
    # only features in the original dataset
    features = ["Price","Price_diff", 
                "Load_DA","Load_Act","Solar_DA","Solar_Act","WindOn_DA","WindOn_Act",
                "WindOff_DA","WindOff_Act","Temp_Act","Coal_fM","Gas_fD","Gas_fM",
                "Gas_fQ","Gas_fY","Oil_fM","EUA_fM","EUR_USD"]
    
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)     
        data = data[features]   
        visualizer = DataVisualizer(data, target_col='Price_diff')  
        
        fig0, ax0 = visualizer.plot_target()
        fig0.savefig("figures/target.png")
        fig1, axes1 = visualizer.show_generic_statistics()
        fig1.savefig("figures/statistics.png")
        fig2, axes2 = visualizer.plot_feature_correlations()
        fig2.savefig("figures/feature_correlations.png")
        fig3, axes3 = visualizer.plot_autoregressivity(lags=168)
        fig3.savefig("figures/autoregressivity.png")
        fig4, ax4 = visualizer.plot_lag_correlations(max_lag=168)
        fig4.savefig("figures/lag_correlations.png")
    else:
        print(f"Data file not found at {filepath}. Run main.py first to generate data.")
