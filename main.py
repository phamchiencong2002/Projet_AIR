from datetime import datetime, timedelta
from adapters.open_meteo_client import OpenMeteoClient
from services.weather_service import WeatherService
from data.transformer import DataTransformer


def get_today_vs_last_year(city):
    om = OpenMeteoClient()
    transformer = DataTransformer()
    svc = WeatherService(geocoder=om, provider=om, transformer=transformer)
    return svc.get_today_vs_last_year(city)


def get_weather_data(city, start_date, end_date):
    om = OpenMeteoClient()
    transformer = DataTransformer()
    svc = WeatherService(geocoder=om, provider=om, transformer=transformer)
    return svc.get_weather_range(city, start_date, end_date)


def get_multi_year_data(city, years=3, end_date=None):
    om = OpenMeteoClient()
    transformer = DataTransformer()
    svc = WeatherService(geocoder=om, provider=om, transformer=transformer)
    return svc.get_multi_year_data(city, years=years, end_date=end_date)


def main():
    city = "Lyon"
    start_date = "2024-10-01"
    end_date = "2024-10-15"
    years = 5
    return {
        'today': get_today_vs_last_year(city)[0],
        'last_year': get_today_vs_last_year(city)[1],
        'short_period': get_weather_data(city, start_date, end_date),
        'multi_year': get_multi_year_data(city, years=years)
    }

if __name__ == "__main__":
    data = main()


