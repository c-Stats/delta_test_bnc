"""tms module"""

import math
import scipy
import numpy as np
import pandas as pd
from sklearn import linear_model
from typing import Dict, Union, Any


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
    freq = np.flip(np.argsort(acf))[1]
    t_freq = freq * (t[1] - t[0])

    # Check if lag is reasonable
    has_lag = freq <= len(t) / 2

    # Lasso regression
    # Model with lag terms
    if has_lag:
        # Values for model with lag and sinunoidal term
        index = np.where(axes["t"] >= t_freq)[0]
        y_final = axes["y"][index][:-1]
        t_final = axes["t"][index][:-1]
        y_lag = np.interp(t_final - t_freq, axes["t"][:-1], axes["y"][:-1])

        X = np.vstack([np.ones(len(t_final)), t_final, y_lag]).T
        lasso = linear_model.LassoCV(cv=5, random_state=0).fit(X, y_final)

        fitted_val = lasso.predict(X)
        residuals = y_final - fitted_val
        r_squared = np.round(np.var(fitted_val) / np.var(y_final), 2)

        last_y_lag = np.interp(last_t - t_freq, axes["t"][:-1], axes["y"][:-1])
        E_last_y = lasso.predict(
            np.array(
                [1, last_t, last_y_lag]
            ).reshape(1, -1)
        )[0]

        # If lag terms are insignificant, then act as if there is no seasonality
        coefs = lasso.coef_
        has_lag = has_lag and not coefs[2] == 0

    # Model with no seasonality
    if not has_lag:
        y_final = axes["y"][:-1]
        t_final = axes["t"][:-1]

        X = np.vstack([np.ones(len(t_final)), t_final]).T
        lasso = linear_model.LassoCV(cv=5, random_state=0).fit(X, y_final)
        coefs = np.append(lasso.coef_, np.array([0]))

        fitted_val = lasso.predict(X)
        residuals = y_final - fitted_val
        r_squared = np.round(np.var(fitted_val) / np.var(y), 2)

        last_y_lag = np.interp(last_t - t_freq, axes["t"][:-1], axes["y"][:-1])
        E_last_y = lasso.predict(np.array([1, last_t]).reshape(1, -1))[0]

    if not use_empirical_residuals_pdf:
        rmse = math.sqrt(np.mean(residuals**2))
        z_score = (last_y - E_last_y) / rmse
        p_val = np.round(scipy.stats.norm.sf(abs(z_score)) * 2, 4)
        CI = np.round(
            np.array([-1, 1]) * scipy.stats.norm.ppf(1 - alpha / 2) * rmse + E_last_y, 4
        )
    else:
        rmse = math.sqrt(np.mean(residuals**2))
        residue = last_y - E_last_y
        cdf_val = sum(1 if x >= residue else 0 for x in residuals) / len(residuals)
        p_val = np.round(2 * min([cdf_val, 1-cdf_val]), 4)
        CI = np.round(
            scipy.stats.mstats.mquantiles((residuals), [alpha / 2, 1 - alpha / 2])
            + E_last_y,
            4,
        )

    trend_term = np.round(coefs[1]*(24*60), 4) if coefs[1] !=0 else "NULL"
    lag_term = np.round(coefs[2], 4) if coefs[2] != 0 else "NULL"
    lag_value = np.round(t_freq / (24 * 60), 4) if has_lag else "NULL"
    test_result = "FAIL" if p_val <= alpha else "PASS"
    cdf_type = "Gaussian" if not use_empirical_residuals_pdf else "Empirical"

    print("------ DELTA TIMESERIES TEST RESULT ------")
    print(f"Model r-squared: {r_squared}")
    print(f"Daily trend: {trend_term}")
    print(f"Y lagged term: {lag_term}")
    print(f"Lag (days): {lag_value}")
    print("*****")
    print(f"Test alpha: {alpha}")
    print(f"Residuals distribution: {cdf_type}")
    print(f"Observed volume: {last_y}")
    print(f"Expected volume: {np.round(E_last_y, 4)}")
    print(f"{1-alpha} CI: ({CI[0]}, {CI[1]})")
    print(f"p-value: {p_val}")
    print(f"Result: {test_result}")
    print("------------------------------------------")

    return {
        "coefs": coefs,
        "r_squared": r_squared,
        "rmse": rmse,
        "freq": t_freq,
        "has_lag": has_lag,
        "p_value": p_val,
        "CI": CI,
        "test_result": test_result,
        "alpha": alpha,
        "use_empirical_residuals_pdf": use_empirical_residuals_pdf,
    }
