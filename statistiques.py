import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from main import get_multi_year_data, get_weather_data  # fonctions existantes
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
####################### Fonction graph_ressentie_vs_reel #######################
def graph_resentie_vs_reel(city: str, start_date: str, end_date: str):
    """
    Crée un tableau et un graphique à partir des colonnes 1, 3 et 6 du DataFrame météo récupéré via get_weather_data().
    """
    # --- Récupération des données ---
    df_multi_year = get_weather_data(city, start_date, end_date)
 
    # --- Vérification du nombre de colonnes ---
    if df_multi_year.shape[1] < 6:
        raise ValueError("Le DataFrame ne contient pas assez de colonnes (au moins 6 nécessaires).")
 
    # --- Sélection des colonnes 1 et 5 ---
    df_selected = df_multi_year.iloc[:, [1, 4]].copy()  # 0=col1, 2=col3, 5=col6
 
    # --- Renommage des colonnes pour plus de clarté ---
    df_selected.columns = ["temperature_2m_mean", "apparent_temperature_mean"]
 
    # --- Ajout d'un index temporel si absent ---
    if "date" not in df_multi_year.columns:
        df_selected["date"] = pd.date_range(start=start_date, end=end_date, periods=len(df_selected))
 
    # --- Création du graphique ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
 
    # Courbe 1 : température moyenne
    sns.lineplot(
        data=df_selected,
        x="date",
        y="temperature_2m_mean",
        label="Température moyenne (°C)",
        color="tab:red",
        ax=ax1
    )
    ax1.set_ylabel("Température moyenne (°C)", color="tab:red")
 
    # Courbe 2 : température ressentie
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_selected,
        x="date",
        y="apparent_temperature_mean",
        label="Temp_ressentie_moy (C°)",
        color="tab:blue",
        ax=ax2
    )
    ax2.set_ylabel("Temp_ressentie_moy (C°)", color="tab:blue")
 
    plt.title(f"Température et vent max à {city.capitalize()}\n({start_date} → {end_date})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
 
    return df_selected, fig
 
########################## Fonction prévisions ###################
def forecast_temperature_next_year(df_multi_year, periods=365):
    """
    Prévision de la température moyenne pour le mois suivant à partir du DataFrame complet.
    """
    df = df_multi_year.copy()
    # Si 'date' est dans l'index, on le remet comme colonne
    if 'date' in df.index.names:
        df = df.reset_index()
 
    # Conversion en datetime et tri
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
 
    # --- Sélection colonne température moyenne ---
    ts = df.set_index('date')['temperature_2m_mean']
 
    # --- Modèle Holt-Winters avec saisonnalité annuelle ---
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=365)
    model_fit = model.fit()
 
    # --- Prévision ---
    forecast_values = model_fit.forecast(periods)
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=periods)
 
    df_forecast = pd.DataFrame({
        'date': forecast_dates,
        'temperature_2m_mean_predite': forecast_values.values
    })
 
    # --- Graphique ---
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(ts.index, ts.values, label='Historique')
    ax.plot(df_forecast['date'], df_forecast['temperature_2m_mean_predite'], 
            label=f'Prévision {periods} jours', linestyle='--', color='red')
    ax.set_title("Prévision de la température moyenne pour le mois suivant")
    ax.set_xlabel("Date")
    ax.set_ylabel("Température moyenne (°C)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
 
    return df_forecast, fig
 
####################### Fonction ACP #######################
def acp_temperature(df_multi_year, start_date, end_date):
    """
    Réalise une ACP sur les 5 variables explicatives pour expliquer la température moyenne.
    """
    df = df_multi_year.copy()
 
    # S'assurer que 'date' est bien une colonne
    if 'date' in df.index.names:
        df = df.reset_index(drop=False)
 
    # Conversion en datetime et filtrage
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
 
    # Variables explicatives
    vars_explicatives = [
        'apparent_temperature_mean',
        'wind_speed_10m_max',
        'sunshine_duration',
        'precipitation_sum',
        'shortwave_radiation_sum'
    ]
 
    # Vérification que les colonnes existent
    for v in vars_explicatives:
        if v not in df.columns:
            raise KeyError(f"La colonne '{v}' est absente du DataFrame.")
 
    # Standardisation
    X_scaled = StandardScaler().fit_transform(df[vars_explicatives])
 
    # ACP
    pca = PCA()
    pcs = pca.fit_transform(X_scaled)
 
    # DataFrame des composantes principales
    df_pcs = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pcs.shape[1])])
    df_pcs['date'] = df['date'].values
    df_pcs['temperature_2m_mean'] = df['temperature_2m_mean'].values
 
    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pcs.shape[1])],
        index=vars_explicatives
    )
 
    explained_var = pca.explained_variance_ratio_
 
    return df_pcs, loadings, explained_var
 
def plot_acp(df_pcs, loadings, explained_var):
    """
    Graphique ACP (PC1 & PC2) avec température moyenne et contributions des variables.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
 
    # Scatter des observations colorées par température
    scatter = ax.scatter(df_pcs['PC1'], df_pcs['PC2'],
                         c=df_pcs['temperature_2m_mean'],
                         cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, label="Température moyenne (°C)")
 
    # Flèches de contribution (loadings)
    for i, var in enumerate(loadings.index):
        ax.arrow(0, 0, loadings.loc[var, 'PC1']*3, loadings.loc[var, 'PC2']*3,
                 color='black', alpha=0.8, head_width=0.05)
        ax.text(loadings.loc[var, 'PC1']*3.2, loadings.loc[var, 'PC2']*3.2, var,
                fontsize=12, color='black')
 
    # Mise en forme
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
    ax.set_title("ACP — Influence des variables sur la température moyenne")
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
 
####################### EXEMPLE D’UTILISATION #######################
if __name__ == "__main__":
    city = "Lyon"
    start_date = "2021-10-01"
    end_date = "2024-10-31"
 
    # Graphiques ressentie vs réel
    df_result, fig = graph_resentie_vs_reel(city, start_date, end_date)
    print(df_result.head())
    plt.show()
 
    # Prévision température
    df_multi_year = get_multi_year_data(city, 5)
    df_forecast, fig_forecast = forecast_temperature_next_year(df_multi_year, periods=365)
    print(df_forecast.head())
    plt.show()
 
    # ACP
    df_pcs, loadings, explained_var = acp_temperature(df_multi_year, start_date, end_date)
    print(df_pcs.head())  # Projection des observations
    print(loadings)       # Loadings
    print(explained_var)  # Variance expliquée
 
    # Graphique ACP
    plot_acp(df_pcs, loadings, explained_var)