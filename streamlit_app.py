import streamlit as st
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt

# ==== SERVICES LAYER ====
from adapters.open_meteo_client import OpenMeteoClient
from services.weather_service import WeatherService
from services.analytics.forecasting import forecast_temperature_next_year
from services.analytics.pca import acp_temperature
from services.analytics.statistics import StatisticsService
from services.analytics.weather_alerts import WeatherAlertService
from services.presentation.weather_presenter import WeatherPresenter
from data.transformer import DataTransformer

# ==== UI COMPONENTS ====
from ui.components.metrics import (
    render_weather_metrics_grid,
    render_secondary_metrics_grid,
    render_comparison_metrics
)
from ui.components.alerts import render_alerts_section
from ui.components.charts import (
    render_temperature_chart,
    render_precipitation_chart,
    render_temperature_comparison_chart,
    render_forecast_chart
)

# ============================================
#              SERVICE INITIALIZATION
# ============================================
def create_services():
    """Crée et retourne toutes les instances de services nécessaires."""
    om = OpenMeteoClient()
    transformer = DataTransformer()
    weather_service = WeatherService(geocoder=om, provider=om, transformer=transformer)
    statistics_service = StatisticsService()
    alert_service = WeatherAlertService()
    presenter = WeatherPresenter()
    return weather_service, statistics_service, alert_service, presenter


# Services globaux (instanciés une seule fois)
_weather_service, _statistics_service, _alert_service, _presenter = create_services()

# ============================================
#              CACHED DATA FETCHERS
# ============================================
@st.cache_data(ttl=900)
def fetch_geocode(city: str):
    """Récupère les coordonnées géographiques d'une ville."""
    om = OpenMeteoClient()
    return om.geocode(city)


@st.cache_data(ttl=900)
def fetch_daily_df(city: str, start_str: str, end_str: str):
    """Récupère les données météorologiques pour une période donnée."""
    return _weather_service.get_weather_range(city, start_str, end_str)


@st.cache_data(ttl=600)
def fetch_today_vs_last_year(city: str):
    """Récupère les données d'aujourd'hui et de l'année dernière."""
    return _weather_service.get_today_vs_last_year(city)


@st.cache_data(ttl=3600)
def fetch_multi_year_df(city: str, years: int = 5):
    """Récupère les données multi-années pour une ville."""
    return _weather_service.get_multi_year_data(city, years=years)


@st.cache_data(ttl=3600)
def compute_hw_forecast(city: str, years: int = 5, periods: int = 365):
    """Calcule la prévision de température pour l'année à venir."""
    df_multi = _weather_service.get_multi_year_data(city, years=years)
    if df_multi is None or getattr(df_multi, "empty", True):
        return None
    df_forecast = forecast_temperature_next_year(df_multi, periods=periods)
    return df_forecast


# ============================================
#              HELPER FUNCTIONS
# ============================================
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare un DataFrame pour l'affichage (gestion des index et colonnes)."""
    if df is None:
        return pd.DataFrame()
    
    df = df.copy()
    if not df.empty and 'date' in df.index.names:
        df = df.reset_index().rename(columns={'date': 'time'})
    
    # Conversion de l'ensoleillement
    df = _presenter.convert_sunshine_duration_to_hours(df)
    
    return df


def render_pca_correlation_circle(loadings: pd.DataFrame, explained_var: pd.Series):
    """Affiche le cercle des corrélations pour l'ACP."""
    if not {"PC1", "PC2"}.issubset(loadings.columns):
        st.warning("Les colonnes PC1 et PC2 ne sont pas disponibles dans les loadings.")
        return
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Cercle unité
    circle = plt.Circle((0, 0), 1.0, color="#cccccc", fill=False, linestyle="--")
    ax.add_artist(circle)
    
    # Axes
    ax.axhline(0, color="#999999", linewidth=1)
    ax.axvline(0, color="#999999", linewidth=1)
    
    # Flèches pour chaque variable
    for var in loadings.index:
        x = float(loadings.loc[var, "PC1"]) if not pd.isna(loadings.loc[var, "PC1"]) else 0.0
        y = float(loadings.loc[var, "PC2"]) if not pd.isna(loadings.loc[var, "PC2"]) else 0.0
        ax.arrow(
            0, 0, x, y,
            head_width=0.03, head_length=0.05,
            fc="tab:blue", ec="tab:blue",
            length_includes_head=True
        )
        ax.text(x * 1.07, y * 1.07, var, fontsize=10, color="tab:blue")
    
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var. expl.)")
    if len(explained_var) > 1:
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var. expl.)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Cercle des corrélations")
    st.pyplot(fig, use_container_width=True)


def render_pca_loadings_table(loadings: pd.DataFrame):
    """Affiche le tableau des loadings triés par importance."""
    st.markdown("**Contributions des variables (loadings)**")
    load_sorted = loadings.copy()
    
    # Ordonner par l'importance absolue sur PC1 puis PC2
    sort_cols = []
    if "PC1" in load_sorted.columns:
        sort_cols.append(load_sorted["PC1"].abs())
    if "PC2" in load_sorted.columns:
        sort_cols.append(load_sorted["PC2"].abs())
    
    if sort_cols:
        key = sum((col.rank(ascending=False) for col in sort_cols))
        load_sorted = load_sorted.loc[key.sort_values().index]
    
    st.dataframe(load_sorted.style.format("{:.3f}"), use_container_width=True)


# ============================================
#              STREAMLIT UI
# ============================================
st.set_page_config(page_title="Projet météo", layout="wide")

st.title("🌤️ Projet dashboard météo")

# Sidebar
with st.sidebar:
    st.header("📌 Navigation")
    page = st.radio(
        "Aller à",
        [
            "Stat global",
            "Prévisions",
            "J vs N-1",
            "ACP",
        ],
    )

    st.divider()
    st.header("⚙️ Paramètres")
    default_city = "Paris"
    city = st.text_input("Ville", value=default_city)
    today = date.today()
    start_dt = st.date_input("Début", value=today - timedelta(days=30))
    end_dt = st.date_input("Fin", value=today)

# Validation de la ville
geoloc = fetch_geocode(city) if city else None
if not geoloc:
    st.info("Saisissez une ville valide dans la barre latérale pour commencer.")
    st.stop()

# Préparation des dates
start_str = start_dt.strftime("%Y-%m-%d")
end_str = end_dt.strftime("%Y-%m-%d")

# Récupération et préparation des données principales
df = fetch_daily_df(city, start_str, end_str)
df = prepare_dataframe(df)

# ============================================
#              PAGE ROUTING
# ============================================
if page == "Stat global":
    # Métriques principales
    render_weather_metrics_grid(df, _statistics_service, _presenter)
    
    st.divider()
    
    # Métriques secondaires
    render_secondary_metrics_grid(df, _statistics_service, _presenter)
    
    # Graphiques
    st.subheader("Courbes principales")
    render_temperature_chart(df, _presenter)
    render_precipitation_chart(df, _presenter)
    render_temperature_comparison_chart(df, _presenter)

elif page == "Prévisions":
    st.subheader("Prévisions")
    st.markdown("**Prévision statistique de la température moyenne**")
    
    with st.spinner("Calcul de la prévision à partir de l'historique multi‑années..."):
        df_pred = compute_hw_forecast(city, years=5, periods=365)
    
    render_forecast_chart(df_pred)
    
    if df_pred is not None and not df_pred.empty:
        with st.expander("Données de prévision (quotidiennes)"):
            st.dataframe(df_pred, use_container_width=True)

elif page == "J vs N-1":
    st.subheader("Comparaison aujourd'hui vs année dernière")
    
    with st.spinner(f"Chargement des données pour {city}..."):
        df_today, df_last_year = fetch_today_vs_last_year(city)
    
    # Validation des données
    if df_today is None or df_today.empty:
        st.error("Impossible de récupérer les données d'aujourd'hui.")
    elif df_last_year is None or df_last_year.empty:
        st.warning("Données d'aujourd'hui disponibles, mais impossible de récupérer les données de l'année dernière.")
        st.info("**Données d'aujourd'hui**")
        st.dataframe(df_today, use_container_width=True)
    else:
        # Préparation des données
        df_today = prepare_dataframe(df_today)
        df_last_year = prepare_dataframe(df_last_year)
        
        # Métriques de comparaison
        render_comparison_metrics(
            df_today, df_last_year, _statistics_service, _presenter
        )
        
        st.divider()
        
        # Alertes météorologiques
        render_alerts_section(df_today, _alert_service)

elif page == "ACP":
    st.subheader("ACP – analyse en composantes principales")
    
    # Préparation du DataFrame pour l'ACP
    df_acp = df.copy()
    if "time" in df_acp.columns and "date" not in df_acp.columns:
        df_acp["date"] = df_acp["time"]
    
    # Calcul de l'ACP
    try:
        df_pcs, loadings, explained_var = acp_temperature(df_acp, start_str, end_str)
    except Exception as e:
        st.error(f"Erreur lors du calcul de l'ACP: {e}")
        df_pcs, loadings, explained_var = None, None, None
    
    # Affichage des résultats
    if loadings is not None and explained_var is not None:
        render_pca_correlation_circle(loadings, explained_var)
        render_pca_loadings_table(loadings)
    else:
        st.warning("Impossible d'afficher les résultats de l'ACP.")
