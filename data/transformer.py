from typing import Dict, Any

import pandas as pd


class DataTransformer:
    def create_daily_dataframe(self, api_response: Dict[str, Any]) -> pd.DataFrame:
        if not api_response or "daily" not in api_response:
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
            "shortwave_radiation_sum",
        ]

        df = pd.DataFrame({col: daily[col] for col in possible_columns if col in daily})

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.rename(columns={"time": "date"}, inplace=True)
            df.set_index("date", inplace=True)

        return df


