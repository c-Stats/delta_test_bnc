"""tms module"""

import pandas as pd
import numpy as np
import scipy
import math
from numpy import ndarray
from scipy.optimize import minimize
from typing import Dict, Union, Any, List
import statistics


def smape(y_pred: List[float], y_true: List[float], return_residuals: bool = False) -> float:
    """"Symetric mape.
    
    Args:
        y_pred: prediction
        y_true: truth
        return_residuals: return residuals

    Returns:
        smape
    
    """
    index = [i for i,x in enumerate(y_true) if x != 0]
    yp = np.array(y_pred)[index]
    yt = np.array(y_true)[index]

    mape_1 = abs(yp - yt) / abs(yt)
    mape_2 = abs(yp - yt) / abs(yt)

    mape = 100 * np.array([max([x, y]) for (x,y) in zip(mape_1, mape_2)])    
    if return_residuals:
        return mape

    return statistics.mean(mape)


def loss(beta: ndarray, X: ndarray, Y: ndarray) -> float:
    """"Compute the loss for a linear model.
    
    Args:
        beta: beta
        X: independent variables
        y: dependent variables

    Returns:
        the loss
    
    """
    y_pred = np.matmul(X, beta)
    return smape(y_pred, Y)


def extract_axes(
    tms_df: pd.DataFrame, date_col: str = "Date", value_col: str = "Rows"
) -> Union[Dict[str, Any], None]:
    """Returns a t and y axis taken from a pandas DataFrame containing a date and a value column.

    Args:
        tms_df: a pandas DataFrame
        date_col: name of a the column containing the date or timestamps
        value_col: name of the column containing the time series

    Returns:
        A dictionary containing the time series axes

    """
    if date_col not in tms_df.columns or value_col not in tms_df.columns:
        print(f"ERROR: column(s) not found in {tms_df.columns}")
        return None

    tms_df = tms_df.sort_values([date_col], ascending=True).reset_index(drop=True)

    try:
        min_date = tms_df[date_col].min()
        t = np.array([int((x - min_date).total_seconds() / 60) for x in tms_df[date_col]])  # pylint: disable=C0103,R0915

    except ValueError:
        print(
            f"ERROR: column {date_col} is not of type datetime.date or datetime.datetime"
        )
        return None

    y = np.array(tms_df[value_col])
    return {"t": t, "y": y, "y_start": min_date}


def timeseries_model(
    tms_df: pd.DataFrame,
    date_col: str = "Date",
    value_col: str = "Rows",
    alpha: float = 0.05,
    use_empirical_residuals_pdf: bool = False,
) -> Union[Dict[str, Any], None]:  # pylint: disable=R0914
    """Returns the model of type a + b*t + c*sin(phi*t) + d*y_lag.

    Args:
        tms_df: a pandas DataFrame
        date_col: name of a the column containing the date
        value_col: name of the column containing the time series
        alpha: statistical test alpha
        use_empirical_residuals_pdf: is False, residuals are treated as Gaussian

    Returns:
        A dictionary containing the model

    """
    axes = extract_axes(tms_df, date_col, value_col)
    if not axes:
        return None
    
    axes = extract_axes(tms_df, date_col, value_col)

    last_t = axes["t"][-1]
    last_y = axes["y"][-1]

    # Use the mode of time difference to generate equally spaced points alongside the time domain
    t_delta = np.diff(axes["t"])
    t_delta_mode = scipy.stats.mode(t_delta, keepdims=False)[0]

    # Use interpolation to generate y(t) values
    n_points = int(axes["t"][-2] / t_delta_mode)
    t = np.linspace(start=0, stop=axes["t"][-2], num=n_points + 1)
    y = np.interp(t, axes["t"][:-1], axes["y"][:-1])

    # Detrend y
    X = np.vstack([np.ones(len(t)), t]).T
    c, b = np.linalg.lstsq(X, y, rcond=None)[0]
    y_detrend = y - (b * t + c)
    y_detrend_cs = (y_detrend - np.mean(y_detrend)) / np.sqrt(np.var(y_detrend))

    # Find the highest frequency
    acf = np.correlate(y_detrend_cs, y_detrend_cs, "full")[-len(y_detrend) :] / len(
        y_detrend
    )

    all_freq = np.flip(np.argsort(acf))[1:]
    max_freq = len(t) / 2
    for i in range(0, len(all_freq)):
        freq = all_freq[i]
        if freq < max_freq:
            break

    t_freq = freq * (t[1] - t[0])

    # Values for model with lag 
    index = np.where(axes["t"] >= t_freq)[0]
    y_final = axes["y"][index][:-1]
    t_final = axes["t"][index][:-1]
    y_lag = np.interp(t_final - t_freq, axes["t"][:-1], axes["y"][:-1])

    X = np.vstack([np.ones(len(t_final)), t_final, y_lag]).T
    Y = y_final

    beta_init = np.array([0]*X.shape[1])
    beta_init[0] = statistics.mean(y_final)

    result = minimize(loss, beta_init, args=(X, Y), method = 'BFGS', options = {'maxiter': 500})
    coefs = result.x

    fitted_val = np.matmul(X, np.array(coefs))
    residuals = y_final - fitted_val
    r_squared = np.round(np.var(fitted_val) / np.var(y_final), 2)

    last_y_lag = np.interp(last_t - t_freq, axes["t"][:-1], axes["y"][:-1])
    E_last_y = float(np.matmul(np.array([1, last_t, last_y_lag]).reshape(1, -1), coefs)[0])
    delta = E_last_y - last_y

    residuals_smape = 1 + smape(fitted_val, y_final, True)
    last_y_smape = 1 + float(smape([E_last_y], [last_y], True)[0])

    r_smape_logval = np.array([math.log(x) for x in residuals_smape])
    theta = statistics.mean(residuals_smape * r_smape_logval) + statistics.mean(residuals_smape) * statistics.mean(r_smape_logval)
    k = statistics.mean(residuals_smape)/theta

    n = len(residuals_smape)
    theta = (n / (n - 1)) * theta
    k = k - (1/n)*(3*k - (2/3)*(k/(1+k) - (4/5)*(k/(1+k)**2)))

    cdf_value = scipy.stats.gamma.cdf(last_y_smape, k, scale = theta)
    p_val = round(2*min([cdf_value, 1-cdf_value]), 4)

    res_pred_cor = round(scipy.stats.pearsonr(fitted_val, residuals)[0], 2)

    baseline = round(coefs[0])
    trend_term = round(coefs[1]*(24*60))
    lag_term = round(coefs[2], 4)
    lag_value = np.round(t_freq / (24 * 60), 2) 
    test_result = "FAIL" if p_val <= alpha else "PASS"
    residuals_distribution = {"type": "Gamma", "shape": round(k, 4), "scale": round(theta, 4)}

    last_y = round(last_y)
    E_last_y = round(E_last_y)

    print("------ DELTA TIMESERIES TEST RESULT ------")
    print(f"Model r-squared: {r_squared}")
    print(f"Correlation between residuals and fitted values: {res_pred_cor}")
    print(f"Baseline: {baseline}")
    print(f"Daily trend: {trend_term}")
    print(f"Y lagged term: {lag_term}")
    print(f"Lag (days): {lag_value}")
    print("*****")
    print(f"Symmetric MAPE+1 residuals distribution: {residuals_distribution}")
    print(f"Observed volume: {last_y}")
    print(f"Expected volume: {E_last_y}")
    print(f"p-value: {p_val}")
    print(f"Test alpha: {alpha}")
    print(f"Result: {test_result}")
    print("------------------------------------------")    

    return {
        "coefs": {"baseline": baseline, "daily_trend": trend_term, "lagged_value": lag_term},
        "lag_(days)": lag_value,
        "r_squared": r_squared,
        "fitted_val_and_residuals_correlation": res_pred_cor,
        "volume": {"observed": last_y, "expected": E_last_y, "delta": delta, "mape": round(delta / last_y, 4)},
        "p_value": p_val,
        "test_result": test_result,
        "alpha": alpha,
        "symmetric_mape+1_residuals_distribution": residuals_distribution,
    }
