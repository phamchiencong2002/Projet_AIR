import pandas as pd

def create_daily_dataframe(api_response):
    """
    Crée un DataFrame pandas à partir de la réponse API contenant les données daily.
    Compatible avec les API forecast et historical.
    
    Args:
        api_response: Réponse JSON de l'API Open-Meteo
    
    Returns:
        DataFrame pandas avec les données daily indexé par date
    """
    if not api_response or "daily" not in api_response:
        print("Erreur : Aucune donnée daily dans la réponse de l'API")
        return pd.DataFrame()
    
    daily = api_response["daily"]
    
    possible_columns = [
        "time",
        "weathercode",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_mean",
        "apparent_temperature_max",
        "wind_speed_10m_max",
        "sunshine_duration",
        "precipitation_sum",
        "shortwave_radiation_sum"
    ]
    
    df = pd.DataFrame({col: daily[col] for col in possible_columns if col in daily})
    
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.rename(columns={"time": "date"}, inplace=True)
        df.set_index("date", inplace=True)
    
    return df