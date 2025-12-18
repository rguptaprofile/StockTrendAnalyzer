try:
    import yfinance as yf  # type: ignore
except Exception:
    # yfinance may not be installed in this environment; allow the module to load
    yf = None

import pandas as pd
import numpy as np
import warnings

# Prefer statsmodels ARIMA if available, otherwise fall back to a lightweight predictor
try:
    import importlib
    arima_mod = importlib.import_module("statsmodels.tsa.arima.model")
    ARIMA = getattr(arima_mod, "ARIMA")
    _HAS_STATSMODELS = True
except Exception:
    ARIMA = None
    _HAS_STATSMODELS = False


def _naive_return_forecast(close: pd.Series, forecast_days: int = 5) -> pd.Series:
    """Fallback forecast based on recent average daily returns.

    This is a fast, dependency-free baseline used when statsmodels is not installed.
    It compounds the mean daily return computed over the last 30 trading days.
    """
    returns = close.pct_change().dropna()
    if returns.empty:
        # no returns to base prediction on: repeat last price
        last_price = float(close.iloc[-1])
        preds = [last_price] * forecast_days
    else:
        recent = returns.tail(30)
        avg_daily_return = float(recent.mean())
        last_price = float(close.iloc[-1])
        preds = [last_price * ((1 + avg_daily_return) ** (i + 1)) for i in range(forecast_days)]

    # Build a date index starting the day after the last observation
    start = pd.to_datetime(close.index[-1])
    idx = pd.date_range(start=start + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    return pd.Series(preds, index=idx)


def predict_short_term_price(ticker: str, forecast_days: int = 2) -> pd.Series:
    """Predict short-term future close prices for `ticker`.

    Behavior:
    - If `statsmodels` is available, fit a simple ARIMA(5,1,0) on daily close prices and forecast.
    - If `statsmodels` is missing, emit a warning and return a lightweight return-compounding forecast.

    Returns a pandas Series with a daily DatetimeIndex for the forecast horizon.
    """
    data = yf.download(ticker, period="6mo", interval="1d")
    if data is None or "Close" not in data:
        raise RuntimeError(f"No data returned for ticker {ticker}")

    close = data["Close"].dropna()
    if close.empty:
        raise RuntimeError(f"No close prices available for {ticker}")

    if _HAS_STATSMODELS:
        try:
            model = ARIMA(close, order=(5, 1, 0))  # Simple ARIMA model
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_days)
            # If the ARIMA forecast already has a datetime index, return it.
            if isinstance(forecast, pd.Series) and isinstance(forecast.index, pd.DatetimeIndex):
                out = forecast.copy()
                out.name = out.name or 'predicted_mean'
                return out

            # Otherwise synthesize a daily DatetimeIndex starting the day after the last observed date
            last_ts = pd.to_datetime(close.index[-1])
            idx = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            return pd.Series(np.asarray(forecast), index=idx, name='predicted_mean')
        except Exception as e:
            warnings.warn(f"ARIMA fit failed, falling back to naive predictor: {e}")
            return _naive_return_forecast(close, forecast_days)
    else:
        warnings.warn("statsmodels not installed — using a simple return-based fallback predictor.")
        return _naive_return_forecast(close, forecast_days)


if __name__ == '__main__':
    forecast = predict_short_term_price("AAPL")
    print("5-day forecast:")
    print(forecast)
