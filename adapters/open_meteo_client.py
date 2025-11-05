from typing import Dict, Any, Optional

from core.interfaces import GeocodingProvider, WeatherProvider
from adapters.api_client import (
    get_geocoding_data,
    get_daily_weather_data,
    get_forecast_today,
    get_historical_same_day_last_year,
)


class OpenMeteoClient(GeocodingProvider, WeatherProvider):
    def geocode(self, city: str) -> Optional[Dict[str, Any]]:
        return get_geocoding_data(city)

    def daily_today(self, geoloc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return get_forecast_today(geoloc)

    def daily_range(self, geoloc: Dict[str, Any], start: str, end: str) -> Optional[Dict[str, Any]]:
        return get_daily_weather_data(geoloc, start, end)

    def daily_same_day_last_year(self, geoloc: Dict[str, Any], date_last_year: str) -> Optional[Dict[str, Any]]:
        return get_historical_same_day_last_year(geoloc, date_last_year)


