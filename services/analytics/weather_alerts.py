"""Service de gestion des alertes météorologiques."""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class WeatherAlert:
    """Représente une alerte météorologique."""
    emoji: str
    title: str
    level: str
    message: str
    color: str
    severity: int


class WeatherAlertService:
    """Service responsable de l'évaluation des alertes météorologiques."""
    
    # Seuils de température
    TEMP_EXTREME = 38.0  # °C
    TEMP_HIGH = 35.0
    TEMP_MODERATE = 30.0
    
    # Seuils de précipitations (mm)
    RAIN_EXTREME = 80.0
    RAIN_HIGH = 40.0
    RAIN_MODERATE = 20.0
    
    # Seuils de vent (km/h)
    WIND_EXTREME = 100.0
    WIND_HIGH = 70.0
    WIND_MODERATE = 40.0
    
    # Seuils de froid (°C)
    COLD_EXTREME = -5.0
    COLD_MODERATE = 0.0
    COLD_LOW = 5.0
    
    def evaluate_alerts(self, df_today: pd.DataFrame) -> List[WeatherAlert]:
        """Évalue les alertes météorologiques pour les données du jour."""
        alerts = []
        
        # Alerte chaleur
        alerts.extend(self._check_temperature_alerts(df_today))
        
        # Alerte pluie
        alerts.extend(self._check_rain_alerts(df_today))
        
        # Alerte vent
        alerts.extend(self._check_wind_alerts(df_today))
        
        # Alerte froid
        alerts.extend(self._check_cold_alerts(df_today))
        
        # Trier par sévérité (plus sévère en premier)
        alerts.sort(key=lambda x: x.severity, reverse=True)
        return alerts
    
    def _check_temperature_alerts(self, df: pd.DataFrame) -> List[WeatherAlert]:
        """Vérifie les alertes de température."""
        alerts = []
        if "temperature_2m_max" not in df.columns or df.empty:
            return alerts
        
        try:
            temp_max = float(df["temperature_2m_max"].iloc[0])
        except (IndexError, ValueError, TypeError):
            return alerts
        
        if temp_max >= self.TEMP_EXTREME:
            alerts.append(WeatherAlert(
                emoji="🌡️",
                title="Alerte chaleur extrême",
                level="Extrême",
                message="Alerte canicule : restez au frais, surveillez les personnes vulnérables.",
                color="#ff4444",
                severity=3
            ))
        elif temp_max >= self.TEMP_HIGH:
            alerts.append(WeatherAlert(
                emoji="🌡️",
                title="Alerte chaleur extrême",
                level="Élevé",
                message="Risque de coup de chaleur. Évitez les activités physiques.",
                color="#ff9933",
                severity=2
            ))
        elif temp_max >= self.TEMP_MODERATE:
            alerts.append(WeatherAlert(
                emoji="🌡️",
                title="Alerte chaleur extrême",
                level="Modéré",
                message="Chaleur importante prévue aujourd'hui. Hydratez-vous.",
                color="#ffdd44",
                severity=1
            ))
        
        return alerts
    
    def _check_rain_alerts(self, df: pd.DataFrame) -> List[WeatherAlert]:
        """Vérifie les alertes de précipitations."""
        alerts = []
        rain_24h = 0.0
        
        if "rain_sum" in df.columns and not df.empty:
            try:
                rain_24h = float(df["rain_sum"].iloc[0])
            except (IndexError, ValueError, TypeError):
                pass
        elif "precipitation_sum" in df.columns and not df.empty:
            try:
                rain_24h = float(df["precipitation_sum"].iloc[0])
            except (IndexError, ValueError, TypeError):
                pass
        
        if rain_24h > self.RAIN_EXTREME:
            alerts.append(WeatherAlert(
                emoji="🌧️",
                title="Alerte pluie intense / risque d'inondation locale",
                level="Extrême",
                message="Risque d'inondation localisée.",
                color="#ff4444",
                severity=3
            ))
        elif rain_24h > self.RAIN_HIGH:
            alerts.append(WeatherAlert(
                emoji="🌧️",
                title="Alerte pluie intense / risque d'inondation locale",
                level="Fort",
                message="Fortes pluies : vigilance sur les routes.",
                color="#ff9933",
                severity=2
            ))
        elif rain_24h > self.RAIN_MODERATE:
            alerts.append(WeatherAlert(
                emoji="🌧️",
                title="Alerte pluie intense / risque d'inondation locale",
                level="Risque modéré",
                message="Pluies modérées attendues.",
                color="#ffdd44",
                severity=1
            ))
        
        return alerts
    
    def _check_wind_alerts(self, df: pd.DataFrame) -> List[WeatherAlert]:
        """Vérifie les alertes de vent."""
        alerts = []
        wind_speed_kmh = 0.0
        
        if "wind_gusts_10m_mean" in df.columns and not df.empty:
            try:
                wind_ms = float(df["wind_gusts_10m_mean"].iloc[0])
                wind_speed_kmh = wind_ms * 3.6  # Conversion m/s vers km/h
            except (IndexError, ValueError, TypeError):
                pass
        elif "wind_speed_10m_mean" in df.columns and not df.empty:
            try:
                wind_ms = float(df["wind_speed_10m_mean"].iloc[0])
                wind_speed_kmh = wind_ms * 3.6
            except (IndexError, ValueError, TypeError):
                pass
        
        if wind_speed_kmh > self.WIND_EXTREME:
            alerts.append(WeatherAlert(
                emoji="💨",
                title="Alerte vent violent",
                level="Violent",
                message="Risque de dégâts : évitez les déplacements.",
                color="#ff4444",
                severity=3
            ))
        elif wind_speed_kmh > self.WIND_HIGH:
            alerts.append(WeatherAlert(
                emoji="💨",
                title="Alerte vent violent",
                level="Fort",
                message="Rafales fortes : attention aux objets légers.",
                color="#ff9933",
                severity=2
            ))
        elif wind_speed_kmh > self.WIND_MODERATE:
            alerts.append(WeatherAlert(
                emoji="💨",
                title="Alerte vent violent",
                level="Modéré",
                message="Vent soutenu prévu.",
                color="#ffdd44",
                severity=1
            ))
        
        return alerts
    
    def _check_cold_alerts(self, df: pd.DataFrame) -> List[WeatherAlert]:
        """Vérifie les alertes de froid."""
        alerts = []
        if "temperature_2m_min" not in df.columns or df.empty:
            return alerts
        
        try:
            temp_min = float(df["temperature_2m_min"].iloc[0])
        except (IndexError, ValueError, TypeError):
            return alerts
        
        if temp_min < self.COLD_EXTREME:
            alerts.append(WeatherAlert(
                emoji="❄️",
                title="Alerte froid / gel",
                level="Froid intense",
                message="Grand froid : prudence à l'extérieur.",
                color="#ff4444",
                severity=3
            ))
        elif temp_min < self.COLD_MODERATE:
            alerts.append(WeatherAlert(
                emoji="❄️",
                title="Alerte froid / gel",
                level="Gel possible",
                message="Risque de gel : protégez les plantes et canalisations.",
                color="#ff9933",
                severity=2
            ))
        elif temp_min < self.COLD_LOW:
            alerts.append(WeatherAlert(
                emoji="❄️",
                title="Alerte froid / gel",
                level="Frais",
                message="Températures basses.",
                color="#ffdd44",
                severity=1
            ))
        
        return alerts

