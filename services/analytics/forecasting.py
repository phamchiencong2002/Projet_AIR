import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_temperature_next_year(df_multi_year: pd.DataFrame, periods: int = 365) -> pd.DataFrame:
    df = df_multi_year.copy()
    if 'date' in df.index.names:
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    ts = df.set_index('date')['temperature_2m_mean']
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=365)
    model_fit = model.fit()
    forecast_values = model_fit.forecast(periods)
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.DataFrame({
        'date': forecast_dates,
        'temperature_2m_mean_predite': forecast_values.values
    })


