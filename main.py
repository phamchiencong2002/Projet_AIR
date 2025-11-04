from datetime import datetime, timedelta
from DataCollecte.api_client import get_geocoding_data, get_forecast_today, get_historical_same_day_last_year, get_daily_weather_data
from DataCollecte.data_cleaning import create_daily_dataframe


def get_today_vs_last_year(city):
    """
    Récupère les données météo du jour actuel et du même jour l'année dernière.
    
    Args:
        city: Nom de la ville (str)
    
    Returns:
        Tuple (df_today, df_last_year) - Deux DataFrames pandas
    """
    # Étape 1 : Récupérer les coordonnées de la ville
    geolocalisation = get_geocoding_data(city)
    
    if not geolocalisation:
        print(f"❌ Erreur : Impossible de trouver les coordonnées pour '{city}'")
        return None, None
    
    today = datetime.now()
    
    response_today = get_forecast_today(geolocalisation)
    
    if not response_today:
        print("❌ Erreur : Impossible de récupérer les données d'aujourd'hui")
        return None, None
    
    df_today = create_daily_dataframe(response_today)
    
    if df_today.empty:
        print("❌ Erreur : Le DataFrame d'aujourd'hui est vide")
        return None, None
    
    one_year_ago = today - timedelta(days=365)
    date_last_year = one_year_ago.strftime("%Y-%m-%d")
    
    response_last_year = get_historical_same_day_last_year(geolocalisation, date_last_year)
    
    if not response_last_year:
        print("❌ Erreur : Impossible de récupérer les données de l'année dernière")
        return df_today, None
    
    df_last_year = create_daily_dataframe(response_last_year)
    
    if df_last_year.empty:
        print("❌ Erreur : Le DataFrame de l'année dernière est vide")
        return df_today, None
    
    
    if 'temperature_2m_mean' in df_today.columns and 'temperature_2m_mean' in df_last_year.columns:
        temp_today = df_today['temperature_2m_mean'].values[0]
        temp_last_year = df_last_year['temperature_2m_mean'].values[0]
        diff = temp_today - temp_last_year
        print(f"Température moyenne : {temp_today:.1f}°C (aujourd'hui) vs {temp_last_year:.1f}°C (l'an dernier) → Différence : {diff:+.1f}°C")
    
    if 'precipitation_sum' in df_today.columns and 'precipitation_sum' in df_last_year.columns:
        precip_today = df_today['precipitation_sum'].values[0]
        precip_last_year = df_last_year['precipitation_sum'].values[0]
        print(f"Précipitations : {precip_today:.1f} mm (aujourd'hui) vs {precip_last_year:.1f} mm (l'an dernier)")
    
    if 'sunshine_duration' in df_today.columns and 'sunshine_duration' in df_last_year.columns:
        sun_today = df_today['sunshine_duration'].values[0] / 3600
        sun_last_year = df_last_year['sunshine_duration'].values[0] / 3600
        print(f"Ensoleillement : {sun_today:.1f}h (aujourd'hui) vs {sun_last_year:.1f}h (l'an dernier)")
    
    return df_today, df_last_year


def get_weather_data(city, start_date, end_date):
    """
    Récupère et traite les données météorologiques quotidiennes pour une ville et une période données.
    
    Args:
        city: Nom de la ville (str)
        start_date: Date de début au format 'YYYY-MM-DD' (str)
        end_date: Date de fin au format 'YYYY-MM-DD' (str)
    
    Returns:
        DataFrame pandas contenant les données daily pour la période spécifiée
    """
    geolocalisation = get_geocoding_data(city)
    
    if not geolocalisation:
        print(f"Erreur : Impossible de trouver les coordonnées pour '{city}'")
        return None
    
    api_response = get_daily_weather_data(geolocalisation, start_date, end_date)
    
    if not api_response:
        print("Erreur : Impossible de récupérer les données météorologiques")
        return None
    
    df = create_daily_dataframe(api_response)
    
    if df.empty:
        print("Erreur : Le DataFrame est vide")
        return None
 
    return df


def get_multi_year_data(city, years=3, end_date=None):
    """
    Récupère les données météorologiques quotidiennes sur plusieurs années.
    
    Cette fonction est optimisée pour récupérer de longues périodes (3-5 ans ou plus).
    Pandas gère facilement plusieurs milliers de lignes, donc pas de problème de performance.
    
    Args:
        city: Nom de la ville (str)
        years: Nombre d'années à récupérer (int, par défaut 3)
        end_date: Date de fin au format 'YYYY-MM-DD' (str, par défaut aujourd'hui)
    
    Returns:
        DataFrame pandas contenant les données daily sur N années
    
    Exemples:
        # Récupérer les 3 dernières années
        df = get_multi_year_data("Paris", years=3)
        
        # Récupérer 5 ans se terminant le 31/12/2024
        df = get_multi_year_data("Lyon", years=5, end_date="2024-12-31")
    """
    
    if end_date is None:
        end = datetime.now()
        end_date_str = end.strftime("%Y-%m-%d")
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_str = end_date
    
    start = end - timedelta(days=years*365)
    start_date_str = start.strftime("%Y-%m-%d")
    
    df = get_weather_data(city, start_date_str, end_date_str)
    
    return df


def main():
    """
    Fonction principale - Exemple d'utilisation des 3 fonctions disponibles
    """
    city = "Lyon"
    start_date = "2024-10-01"
    end_date = "2024-10-15"
    years = 5

    # Fonction 1 : Comparaison aujourd'hui vs année dernière
    df_today, df_last_year = get_today_vs_last_year(city)
    
    # Fonction 2 : Période personnalisée
    df_short_period = get_weather_data(city, start_date, end_date)
    
    # Fonction 3 : DataFrame pour prédiction
    df_multi_year = get_multi_year_data(city, years=years)
    
    return {
        'today': df_today,
        'last_year': df_last_year,
        'short_period': df_short_period,
        'multi_year': df_multi_year
    }

if __name__ == "__main__":
    data = main()


