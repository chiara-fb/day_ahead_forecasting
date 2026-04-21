# CDA: Chiara's Day-Ahead forecasting repository

Repository to forecast day-ahead prices with different probabilistic methods


## Content

```
.
├── input/                      # Processed and raw dataset
├── utils/
│   └── losses.py               # Includes pinball and other losses
│
├── models/
│   ├── ChronosModel.py         # Transformer-based trained on avg pinball loss
│   └── LinearModel.py          # Quantile Regression model 
│   └── TreeModel.py            # Gradient boosted quantile regression trees 
│
├── process_data.py             # adds time features (holiday, one-hot, pos encodings), lags, and forward fills daily values
├── select_features.py          # recursive feat elimination by feature importances (tree-based) r coefficients (quantile regression) accounting for correlations
├── visualize_data.py           # plots statistics and correlations in the data

