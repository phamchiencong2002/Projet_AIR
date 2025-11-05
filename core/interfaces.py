from typing import Protocol, Dict, Any, Optional


class GeocodingProvider(Protocol):
    def geocode(self, city: str) -> Optional[Dict[str, Any]]:
        ...


class WeatherProvider(Protocol):
    def daily_today(self, geoloc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ...

    def daily_range(self, geoloc: Dict[str, Any], start: str, end: str) -> Optional[Dict[str, Any]]:
        ...

    def daily_same_day_last_year(self, geoloc: Dict[str, Any], date_last_year: str) -> Optional[Dict[str, Any]]:
        ...


