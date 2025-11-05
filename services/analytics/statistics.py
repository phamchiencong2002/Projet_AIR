from typing import Optional
import pandas as pd


class StatisticsService:
    """Service responsable des calculs statistiques sur les données météorologiques."""
    
    @staticmethod
    def safe_mean(df: pd.DataFrame, col: str) -> Optional[float]:
        """Calcule la moyenne d'une colonne de manière sécurisée."""
        if col not in df.columns:
            return None
        if df[col].empty or not pd.notnull(df[col]).any():
            return None
        return float(df[col].mean())
    
    @staticmethod
    def safe_sum(df: pd.DataFrame, col: str) -> Optional[float]:
        """Calcule la somme d'une colonne de manière sécurisée."""
        if col not in df.columns:
            return None
        if df[col].empty or not pd.notnull(df[col]).any():
            return None
        return float(df[col].sum())
    
    @staticmethod
    def calculate_rainy_days_percentage(
        df: pd.DataFrame, 
        precipitation_col: str = "precipitation_sum",
        threshold_mm: float = 1.0
    ) -> Optional[float]:
        """Calcule le pourcentage de jours de pluie."""
        if precipitation_col not in df.columns or df.empty:
            return None
        total_days = len(df)
        if total_days == 0:
            return None
        rainy_days = int((df[precipitation_col].astype(float) >= threshold_mm).sum())
        return 100.0 * rainy_days / total_days
    
    @staticmethod
    def calculate_sunny_days_percentage(
        df: pd.DataFrame,
        sunshine_col: str = "sunshine_hours",
        threshold_h: float = 8.0
    ) -> Optional[float]:
        """Calcule le pourcentage de jours ensoleillés."""
        if sunshine_col not in df.columns or df.empty:
            return None
        total_days = len(df)
        if total_days == 0:
            return None
        sunny_days = int((df[sunshine_col].astype(float) >= threshold_h).sum())
        return 100.0 * sunny_days / total_days
    
    @staticmethod
    def calculate_average_sunshine_hours(
        df: pd.DataFrame,
        sunshine_col: str = "sunshine_hours"
    ) -> Optional[float]:
        """Calcule la moyenne d'ensoleillement en heures."""
        return StatisticsService.safe_mean(df, sunshine_col)
    
    @staticmethod
    def prepare_comparison_data(
        df_today: pd.DataFrame,
        df_last_year: pd.DataFrame,
        column: str
    ) -> Optional[tuple[float, float, float]]:
        """Prépare les données pour comparaison (valeur aujourd'hui, valeur N-1, différence)."""
        if column not in df_today.columns or column not in df_last_year.columns:
            return None
        if df_today.empty or df_last_year.empty:
            return None
        try:
            value_today = float(df_today[column].iloc[0])
            value_last_year = float(df_last_year[column].iloc[0])
            diff = value_today - value_last_year
            return (value_today, value_last_year, diff)
        except (IndexError, ValueError, TypeError):
            return None

