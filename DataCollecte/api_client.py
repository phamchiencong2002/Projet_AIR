import os
import requests
from dotenv import load_dotenv

load_dotenv()

HISTORICAL_API_URL = os.getenv("HISTORICAL_API_URL")
GEOCODING_API_URL = os.getenv("GEOCODING_API_URL")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
FORECAST_API_URL = os.getenv("FORECAST_API_URL", "https://api.open-meteo.com/v1/forecast")

def get_geocoding_data(city):
    try:
        query_params = {"name": city, "limit": 1, "language": "fr", "format": "json"}
        response = requests.get(GEOCODING_API_URL, params=query_params)
        response.raise_for_status()
        data = response.json()
        if data and "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            filtered_data = {
                "latitude": result.get("latitude"),
                "longitude": result.get("longitude"),
                "country_code": result.get("country_code"),
                "timezone": result.get("timezone")
            }
            return filtered_data
        return None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la connexion à l'API : {e}")
        return None

def get_forecast_today(geolocalisation):
    """
    Récupère les prévisions météorologiques pour la journée actuelle.
    
    Args:
        geolocalisation: Dictionnaire contenant latitude, longitude et timezone
    
    Returns:
        Réponse JSON de l'API contenant les données daily pour aujourd'hui
    """
    try:
        query_params = {
            "latitude": geolocalisation["latitude"], 
            "longitude": geolocalisation["longitude"], 
            "daily": "precipitation_sum,sunshine_duration,apparent_temperature_max,temperature_2m_min,temperature_2m_max,apparent_temperature_mean,temperature_2m_mean,relative_humidity_2m_mean,uv_index_max,rain_sum,precipitation_probability_mean,wind_gusts_10m_mean,wind_speed_10m_mean",
            "forecast_days": 1
        }
        response = requests.get(FORECAST_API_URL, params=query_params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la connexion à l'API forecast : {e}")
        return None

def get_historical_same_day_last_year(geolocalisation, date_last_year):
    """
    Récupère les données météorologiques historiques pour une date spécifique (même jour l'année dernière).
    
    Args:
        geolocalisation: Dictionnaire contenant latitude, longitude et timezone
        date_last_year: Date au format 'YYYY-MM-DD' (même jour mais l'année dernière)
    
    Returns:
        Réponse JSON de l'API contenant les données daily pour cette date
    """
    params = {
        "latitude": geolocalisation["latitude"],
        "longitude": geolocalisation["longitude"],
        "start_date": date_last_year,        # 'YYYY-MM-DD'
        "end_date": date_last_year,          # cùng 1 ngày
        "daily": ",".join([
            "weathercode",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_mean",
            "wind_speed_10m_max",
            "sunshine_duration",
            "precipitation_sum",
            "shortwave_radiation_sum",
        ]),
        "timezone": geolocalisation["timezone"],
    }
    r = requests.get(HISTORICAL_API_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_daily_weather_data(geolocalisation, start_date, end_date):
    """
    Récupère les données météorologiques quotidiennes (daily) pour une période donnée.
    
    Args:
        geolocalisation: Dictionnaire contenant latitude, longitude et timezone
        start_date: Date de début au format 'YYYY-MM-DD'
        end_date: Date de fin au format 'YYYY-MM-DD'
    
    Returns:
        Réponse JSON de l'API contenant les données daily
    """
    params = {
        "latitude": geolocalisation["latitude"],
        "longitude": geolocalisation["longitude"],
        "start_date": start_date,            # 'YYYY-MM-DD'
        "end_date": end_date,                # 'YYYY-MM-DD'
        "daily": ",".join([
            "weathercode",
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_mean",
            "wind_speed_10m_max",
            "sunshine_duration",
            "precipitation_sum",
            "shortwave_radiation_sum",
        ]),
        "timezone": geolocalisation["timezone"],
    }
    r = requests.get(HISTORICAL_API_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


#print(r.url)
#print(r.status_code, r.text[:400])
