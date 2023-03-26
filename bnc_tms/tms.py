import numpy as np
import pandas as pd
import math
import scipy

from typing import Dict, Union, Any


def extract_axes(
    df: pd.DataFrame, date_col: str = "Date", value_col: str = "Rows"
) -> Union[Dict[str, np.array], None]:
    """Returns a t and y axis taken from a pandas DataFrame containing a date and a value column.

    Args:
        df: a pandas DataFrame
        date_col: name of a the column containing the date or timestamps
        value_col: name of the column containing the time series

    Returns:
        A dictionary containing the time series axes

    """
    if date_col not in df.columns or value_col not in df.columns:
        print(f"ERROR: column(s) not found in {df.columns}")
        return None

    df = df.sort_values([date_col], ascending=True).reset_index(drop=True)

    try:
        min_date = df[date_col].min()
        t = np.array([(x - min_date).total_seconds() / 86400 for x in df[date_col]])

    except ValueError:
        print(
            f"ERROR: column {date_col} is not of type datetime.date or datetime.datetime"
        )
        return None

    y = np.array(df[value_col])
    return {"t": t, "y": y, "y_start" : min_date}


def timeseries_model(
    df: pd.DataFrame, alpha: float, date_col: str = "Date", value_col: str = "Rows"
) -> Dict[str, Any]:
    """Returns the model of type a + b*t + c*sin(phi*t) + d*y_lag.

    Args:
        df: a pandas DataFrame
        alpha: statistical test alpha
        date_col: name of a the column containing the date
        value_col: name of the column containing the time series

    Returns:
        A dictionary containing the model

    """
    axes = extract_axes(df, date_col, value_col)
    n_points = 2*int(axes["t"][-1])

    t = np.array(
        [x for x in np.linspace(start=0, stop=axes["t"][-1], num=n_points + 1)]
    )
    y = np.interp(t, axes["t"], axes["y"])

    last_t = t[-1]
    last_y = y[-1]
    t = t[:-2]
    y = y[:-2]

    X = np.vstack([np.ones(len(t)), t]).T
    c, b = np.linalg.lstsq(X, y, rcond=None)[0]
    y_detrend = y - (b * t + c)

    acf = np.correlate(y_detrend, y_detrend, "full")[-len(y_detrend) :]
    freq = np.flip(np.argsort(acf))[1]
    t_freq = freq * (t[1] - t[0])
    x_seasonality = 1 + np.sin(t * 2 * math.pi / t_freq)

    has_lag = freq <= len(t) / 2 and t_freq >= 1
    if has_lag:

        y_final = y[freq:]
        y_lag = y[: (len(y) - freq)]
        t_final = t[freq:]
        x_seasonality_final = x_seasonality[freq:]

        X = np.vstack([np.ones(len(t_final)), t_final, x_seasonality_final, y_lag]).T
        coefs = np.linalg.lstsq(X, y_final, rcond=None)[0]

        fitted_val = (
            coefs[0]
            + coefs[1] * t_final
            + coefs[2] * x_seasonality_final
            + coefs[3] * y_lag
        )
        residuals = y_final - fitted_val
        r_squared = np.round(np.var(fitted_val) / np.var(y_final), 2)

        last_y_lag = np.interp(last_t - t_freq, t, y)

        E_last_y = (
            coefs[0]
            + coefs[1] * last_t
            + coefs[2] * (1 + np.sin(last_t * 2 * math.pi / t_freq))
            + coefs[3] * last_y_lag
        ) 

    else:
        X = np.vstack([np.ones(len(t)), t]).T
        coefs = np.append(np.linalg.lstsq(X, y, rcond=None)[0], np.array([0, 0]))

        fitted_val = coefs[0] + coefs[1] * t
        residuals = y - fitted_val
        r_squared = np.round(np.var(fitted_val) / np.var(y), 2)
        E_last_y = coefs[0] + coefs[1] * last_t

    rmse = math.sqrt(np.mean(residuals**2))
    z_score = (last_y - E_last_y)/rmse
    p_val = np.round(scipy.stats.norm.sf(abs(z_score))*2, 4)
    CI = np.round(np.array([-1, 1])*scipy.stats.norm.ppf(1-alpha/2)*rmse + E_last_y, 4)
    test_result = "FAIL" if p_val <= alpha else "PASS"

    sin_term = lag_term = np.round(coefs[2], 4) if has_lag else "NULL"
    lag_term = np.round(coefs[3], 4) if has_lag else "NULL"
    lag_value = t_freq if has_lag else "NULL"

    print("------ DELTA TIMESERIES TEST RESULT ------")
    print(f"Model r-squared: {r_squared}")
    print("***")
    print(f"Daily trend: {np.round(coefs[1], 4)}")
    print(f"Sinusoidal term: {sin_term}")
    print(f"Lag term: {lag_term}")
    print(f"Lag (days): {lag_value}")
    print("***")
    print(f"Test alpha: {alpha}")
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
        "has_lag": freq <= len(t) / 2,
        "z_score": z_score,
        "p_value" : p_val,
        "CI": CI,
        "test_result": test_result,
        "alpha": alpha
    }
