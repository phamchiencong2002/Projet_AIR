"""Service de présentation et formatage des données météorologiques pour l'UI."""
from typing import Optional
import pandas as pd


class WeatherPresenter:
    """Service responsable du formatage et de la préparation des données pour l'interface utilisateur."""
    
    @staticmethod
    def format_temperature(value: Optional[float], unit: str = "°C") -> str:
        """Formate une température pour l'affichage."""
        if value is None:
            return "–"
        return f"{value:.1f}{unit}"
    
    @staticmethod
    def format_precipitation(value: Optional[float], unit: str = " mm") -> str:
        """Formate une précipitation pour l'affichage."""
        if value is None:
            return "–"
        return f"{value:.1f}{unit}"
    
    @staticmethod
    def format_sunshine(value: Optional[float], unit: str = " h") -> str:
        """Formate un ensoleillement pour l'affichage."""
        if value is None:
            return "–"
        return f"{value:.1f}{unit}"
    
    @staticmethod
    def format_percentage(value: Optional[float], unit: str = "%") -> str:
        """Formate un pourcentage pour l'affichage."""
        if value is None:
            return "–"
        return f"{value:.1f}{unit}"
    
    @staticmethod
    def format_delta(value: Optional[float], unit: str = "") -> str:
        """Formate une différence pour l'affichage (avec signe + ou -)."""
        if value is None:
            return ""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1f}{unit}"
    
    @staticmethod
    def prepare_temperature_chart_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prépare les données de température pour les graphiques."""
        plot_cols = [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min"
        ]
        available_cols = [c for c in plot_cols if c in df.columns]
        if not available_cols:
            return None
        if "time" not in df.columns:
            return None
        return df.set_index("time")[available_cols]
    
    @staticmethod
    def prepare_precipitation_chart_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prépare les données de précipitations pour les graphiques."""
        if "precipitation_sum" not in df.columns or "time" not in df.columns:
            return None
        return df.set_index("time")[["precipitation_sum"]]
    
    @staticmethod
    def prepare_temperature_comparison_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prépare les données pour comparer température réelle vs ressentie."""
        required_cols = ["temperature_2m_mean", "apparent_temperature_mean", "time"]
        if not all(col in df.columns for col in required_cols):
            return None
        
        cmp_df = df.set_index("time")[
            ["temperature_2m_mean", "apparent_temperature_mean"]
        ].copy()
        cmp_df["ecart_ressenti"] = (
            cmp_df["apparent_temperature_mean"] - cmp_df["temperature_2m_mean"]
        )
        return cmp_df
    
    @staticmethod
    def convert_sunshine_duration_to_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Convertit la durée d'ensoleillement de secondes en heures."""
        df = df.copy()
        if "sunshine_duration" in df.columns and "sunshine_hours" not in df.columns:
            df["sunshine_hours"] = df["sunshine_duration"] / 3600.0
        return df

