from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from core.interfaces import GeocodingProvider, WeatherProvider
from data.transformer import DataTransformer


class WeatherService:
    def __init__(self, geocoder: GeocodingProvider, provider: WeatherProvider, transformer: DataTransformer):
        self._geocoder = geocoder
        self._provider = provider
        self._transformer = transformer

    def get_today_vs_last_year(self, city: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        geoloc = self._geocoder.geocode(city)
        if not geoloc:
            return None, None

        today_json = self._provider.daily_today(geoloc)
        if not today_json:
            return None, None
        df_today = self._transformer.create_daily_dataframe(today_json)
        if df_today.empty:
            return None, None

        one_year_ago = datetime.now() - timedelta(days=365)
        date_last_year = one_year_ago.strftime("%Y-%m-%d")
        last_year_json = self._provider.daily_same_day_last_year(geoloc, date_last_year)
        if not last_year_json:
            return df_today, None
        df_last_year = self._transformer.create_daily_dataframe(last_year_json)
        if df_last_year.empty:
            return df_today, None
        return df_today, df_last_year

    def get_weather_range(self, city: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        geoloc = self._geocoder.geocode(city)
        if not geoloc:
            return None
        api_response = self._provider.daily_range(geoloc, start_date, end_date)
        if not api_response:
            return None
        df = self._transformer.create_daily_dataframe(api_response)
        if df.empty:
            return None
        return df

    def get_multi_year_data(self, city: str, years: int = 3, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        if end_date is None:
            end = datetime.now()
            end_date_str = end.strftime("%Y-%m-%d")
        else:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            end_date_str = end_date
        start = end - timedelta(days=years * 365)
        start_date_str = start.strftime("%Y-%m-%d")
        return self.get_weather_range(city, start_date_str, end_date_str)


