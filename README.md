# CDF: Chiara's Day-ahead Forecasting repository

Repository to forecast day-ahead prices with different probabilistic methods


## Content

```
.
├── figures/                    # Visualizations for the presentation
├── input/                      # Processed and raw dataset (empty in the remote repo)
│
├── models/
│   ├── ChronosModel.py         # Transformer-based trained on avg pinball loss
│   └── EncDecModel.py          # Encoder-Decoder model with recursive NN
│   └── LinearModel.py          # Quantile Regression model 
│   └── TreeModel.py            # Gradient boosted quantile regression trees 
├── utils/
│   └── losses.py               # Includes pinball and other losses
│
├── config.yaml                 # Global configuration file
├── evaluate_data.py            # automatic evaluation of the last run for each model category
├── process_data.py             # adds time features (holiday, one-hot, pos encodings), lags, and forward fills daily values
├── select_features.py          # recursive feat elimination by feature importances (tree-based) or coefficients (quantile regression) accounting for correlations
├── visualize_data.py           # plots statistics and correlations in the data

