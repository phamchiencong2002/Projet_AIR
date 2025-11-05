from typing import Dict, Any

import pandas as pd

from data.transformer import DataTransformer

_TRANSFORMER = DataTransformer()


def create_daily_dataframe(api_response: Dict[str, Any]) -> pd.DataFrame:
    return _TRANSFORMER.create_daily_dataframe(api_response)

