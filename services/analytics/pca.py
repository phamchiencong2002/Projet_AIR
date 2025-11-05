import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def acp_temperature(df_multi_year: pd.DataFrame, start_date: str, end_date: str):
    df = df_multi_year.copy()
    if 'date' in df.index.names:
        df = df.reset_index(drop=False)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    vars_explicatives = [
        'apparent_temperature_mean',
        'wind_speed_10m_max',
        'sunshine_duration',
        'precipitation_sum',
        'shortwave_radiation_sum'
    ]
    for v in vars_explicatives:
        if v not in df.columns:
            raise KeyError(f"La colonne '{v}' est absente du DataFrame.")
    X_scaled = StandardScaler().fit_transform(df[vars_explicatives])
    pca = PCA()
    pcs = pca.fit_transform(X_scaled)
    df_pcs = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    df_pcs['date'] = df['date'].values
    df_pcs['temperature_2m_mean'] = df['temperature_2m_mean'].values
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pcs.shape[1])],
        index=vars_explicatives
    )
    explained_var = pca.explained_variance_ratio_
    return df_pcs, loadings, explained_var


