import numpy as np
import pandas as pd
        


def pinball_loss(y_true, y_pred, q, return_average=True):
    """
    Calculates the pinball loss, a metric for evaluating quantile forecasts.

    The pinball loss is defined as:
    L_q(y, f) = (y - f) * q       if y >= f
              = (f - y) * (1 - q) if y < f
    where y is the true value, f is the quantile forecast, and q is the quantile.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted quantile values.
        q (float): The quantile level (between 0 and 1).
        return_average (bool, optional): If True, returns the mean loss over all samples. 
                                         If False, returns the loss for each sample. 
                                         Defaults to True.

    Returns:
        float or np.ndarray: The average pinball loss or an array of losses for each sample.
    """
    error = y_true - y_pred
    loss = np.maximum(q * error, (q - 1) * error)
    if return_average:
        return np.mean(loss)
    else:
        return loss


def quantile_coverage(y_true, y_pred, q, return_average=True):
    """
    Calculates the coverage of a quantile forecast.

    This function checks if the true value falls on the expected side of the
    predicted quantile.
    - For quantiles q <= 0.5, it checks if y_true >= y_pred. The expected coverage is 1-q.
    - For quantiles q > 0.5, it checks if y_true <= y_pred. The expected coverage is q.

    Note: This is one way to assess reliability. A more common definition of
    quantile coverage is the proportion of observations where y_true < y_pred,
    which should ideally be equal to q.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted quantile values.
        q (float): The quantile level (between 0 and 1).
        return_average (bool, optional): If True, returns the mean coverage. 
                                         If False, returns a boolean array indicating
                                         coverage for each sample. Defaults to True.

    Returns:
        float or np.ndarray: The average coverage or a boolean array of coverage for each sample.
    """
    if q <= 0.5:
        coverage = y_true >= y_pred
    else:
        coverage = y_true <= y_pred
    if return_average:
        return np.mean(coverage)
    else:
        return coverage
    

def average_pinball_loss(pred_df) -> pd.Series:
    """
    Calculates the average pinball loss across multiple quantile forecasts.

    This function assumes the input DataFrame has the true values in the first
    column and predictions for various quantiles in the subsequent columns.
    The quantile level is extracted from the column names, which are expected
    to be in the format '..._qX.X' (e.g., 'pred_q0.1', 'q0.9').

    Args:
        pred_df (pd.DataFrame): A DataFrame with the first column as true values
                                and other columns as quantile predictions.

    Returns:
        pd.Series: A Series containing the average pinball loss for each sample,
                   averaged across all quantiles.
    """
    y_true, y_preds = pred_df.iloc[:,0], pred_df.iloc[:,1:]
    loss = 0

    for name in y_preds: 
        q = float(name.split("_q")[1])
        q_loss = pinball_loss(y_true, y_preds[name], q, return_average=False)
        loss += q_loss
    
    return loss / y_preds.shape[1]


def average_absolute_error(pred_df, q=0.5) -> pd.Series:
    """
    Calculates the average absolute error for one quantile.
    """
    y_true = pred_df["true"]
    y_pred = pred_df[f"pred_q{q}"]
    return (y_true - y_pred).abs()

def average_deviation(pred_df, q=0.5) -> pd.Series:
    """
    Calculates the average deviation for one quantile.
    """
    y_true = pred_df["true"]
    y_pred = pred_df[f"pred_q{q}"]
    return (y_true - y_pred)




def coverage_within_range(pred_df) -> pd.Series:
    """
    Calculates whether the true values fall within the range of the outermost predicted quantiles.

    This function assumes the input DataFrame has the true values in the first
    column and predictions for various quantiles in the subsequent columns,
    ordered from lowest to highest quantile. It uses the first prediction column 
    as the minimum bound and the last prediction column as the maximum bound.

    Args:
        pred_df (pd.DataFrame): A DataFrame with the first column as true values
                                and other columns as quantile predictions.

    Returns:
        pd.Series: A boolean Series indicating whether each true value falls 
                   within the range of the lowest and highest predicted quantiles.
    """
    y_true, y_preds = pred_df.iloc[:,0], pred_df.iloc[:,1:]
    pred_min, pred_max = y_preds.iloc[:,0], y_preds.iloc[:,-1]

    q_min = float(pred_min.name.split("_q")[1])
    cov_min = quantile_coverage(y_true, pred_min, q_min, return_average=False)
    q_max = float(pred_max.name.split("_q")[1])
    cov_max = quantile_coverage(y_true, pred_max, q_max, return_average=False)

    return cov_min & cov_max