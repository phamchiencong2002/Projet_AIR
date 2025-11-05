import streamlit as st
import pandas as pd
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt

# ==== SERVICES LAYER ====
from adapters.open_meteo_client import OpenMeteoClient
from services.weather_service import WeatherService
from services.analytics.forecasting import forecast_temperature_next_year
from services.analytics.pca import acp_temperature
from data.transforms import create_daily_dataframe
from data.transformer import DataTransformer

# ============================================
#              HELPER FUNCTIONS
# ============================================
def _json_daily_to_df(j):
    df = create_daily_dataframe(j)
    if not df.empty and 'date' in df.index.names:
        df_reset = df.reset_index()
        if 'date' in df_reset.columns:
            df_reset.rename(columns={'date': 'time'}, inplace=True)
        return df_reset
    return pd.DataFrame()

_om = OpenMeteoClient()
_transformer = DataTransformer()
_svc = WeatherService(geocoder=_om, provider=_om, transformer=_transformer)

@st.cache_data(ttl=900)
def fetch_geocode(city: str):
    return _om.geocode(city)

@st.cache_data(ttl=900)
def fetch_daily_df(city: str, start_str: str, end_str: str):
    return _svc.get_weather_range(city, start_str, end_str)

@st.cache_data(ttl=600)
def fetch_today_vs_last_year(city: str):
    return _svc.get_today_vs_last_year(city)

@st.cache_data(ttl=3600)
def fetch_multi_year_df(city: str, years: int = 5):
    return _svc.get_multi_year_data(city, years=years)

@st.cache_data(ttl=3600)
def compute_hw_forecast(city: str, years: int = 5, periods: int = 365):
    df_multi = _svc.get_multi_year_data(city, years=years)
    if df_multi is None or getattr(df_multi, "empty", True):
        return None
    df_forecast = forecast_temperature_next_year(df_multi, periods=periods)
    return df_forecast

def safe_mean(df, col):
    return float(df[col].mean()) if col in df and pd.notnull(df[col]).any() else None

def safe_sum(df, col):
    return float(df[col].sum()) if col in df and pd.notnull(df[col]).any() else None

# ===== PCA helpers =====

def _standardize(X: np.ndarray):
    """Return standardized matrix (z-score), means and stds (avoid divide-by-zero)."""
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    Xc = (X - means) / stds
    Xc = np.nan_to_num(Xc, nan=0.0)
    return Xc, means, stds

def pca_from_df(df: pd.DataFrame, cols: list, n_components: int = 2):
    """Compute PCA via SVD (no sklearn). Returns scores, loadings, explained_var_ratio."""
    if not cols:
        return None
    data = df[cols].dropna().values
    if data.shape[0] < 2:
        return None
    Xc, means, stds = _standardize(data)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (Xc.shape[0] - 1)
    evr = var / var.sum()
    scores = U @ np.diag(S)
    loadings = Vt.T
    scores = scores[:, :n_components]
    loadings = loadings[:, :n_components]
    evr = evr[:n_components]
    return {
        "scores": scores,
        "loadings": loadings,
        "evr": evr,
        "cols": cols,
        "means": means,
        "stds": stds,
    }

st.set_page_config(page_title="Projet météo", layout="wide")

st.title("🌤️ Projet dashboard météo")

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

geoloc = fetch_geocode(city) if city else None
if not geoloc:
    st.info("Saisissez une ville valide dans la barre latérale pour commencer.")
    st.stop()

start_str = start_dt.strftime("%Y-%m-%d")
end_str = end_dt.strftime("%Y-%m-%d")

df = fetch_daily_df(city, start_str, end_str)
if df is None:
    df = pd.DataFrame()
if not df.empty and 'date' in df.index.names:
    df = df.reset_index().rename(columns={'date': 'time'})

if "sunshine_duration" in df.columns:
    df["sunshine_hours"] = df["sunshine_duration"] / 3600.0



if page == "Stat global":

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_mean = safe_mean(df, "temperature_2m_mean")
        st.metric("Temp. moyenne (°C)", f"{t_mean:.1f}" if t_mean is not None else "–")
    with c2:
        t_max = safe_mean(df, "temperature_2m_max")
        st.metric("Temp. max moy. (°C)", f"{t_max:.1f}" if t_max is not None else "–")
    with c3:
        p_sum = safe_sum(df, "precipitation_sum")
        st.metric("Précipitations totales (mm)", f"{p_sum:.1f}" if p_sum is not None else "–")
    with c4:
        sun_h = safe_sum(df, "sunshine_hours") if "sunshine_hours" in df else None
        st.metric("Ensoleillement total (h)", f"{sun_h:.1f}" if sun_h is not None else "–")
    
    st.divider()

    rain_threshold_mm = 1.0
    sun_threshold_h = 8.0
    k1, k2, k3 = st.columns(3)
    if "precipitation_sum" in df and not df.empty:
        total_days = len(df)
        rainy_days = int((df["precipitation_sum"].astype(float) >= rain_threshold_mm).sum())
        pct_rain = 100.0 * rainy_days / max(total_days, 1)
        with k1:
            st.metric("% jours de pluie", f"{pct_rain:.1f}%", help=f"Seuil: ≥ {rain_threshold_mm:.1f} mm/j")
    else:
        with k1:
            st.metric("% jours de pluie", "–")
    if "sunshine_hours" in df and not df.empty:
        avg_sun = float(df["sunshine_hours"].mean())
        sunny_days = int((df["sunshine_hours"].astype(float) >= sun_threshold_h).sum())
        pct_sunny = 100.0 * sunny_days / max(len(df), 1)
        with k2:
            st.metric("Ens. moyen (h/j)", f"{avg_sun:.1f}")
        with k3:
            st.metric("% jours ≥ 8 h ens.", f"{pct_sunny:.1f}%", help=f"Seuil: ≥ {sun_threshold_h:.1f} h/j")
    else:
        with k2:
            st.metric("Ens. moyen (h/j)", "–")
        with k3:
            st.metric("% jours ≥ 8 h ens.", "–")

    st.subheader("Courbes principales")

    st.markdown("**Température (moy/max/min)**")
    plot_cols = [c for c in ["temperature_2m_mean","temperature_2m_max","temperature_2m_min"] if c in df]
    if plot_cols:
        st.line_chart(df.set_index("time")[plot_cols])
    else:
        st.info("Pas de colonnes température disponibles.")

    st.markdown("**Cumul précipitations**")
    if "precipitation_sum" in df:
        st.bar_chart(df.set_index("time")["precipitation_sum"])
    else:
        st.info("Pas de données de précipitations.")

    st.markdown("**Température réelle vs ressentie**")
    if "temperature_2m_mean" in df.columns and "apparent_temperature_mean" in df.columns:
        cmp_df = df.set_index("time")[
            ["temperature_2m_mean", "apparent_temperature_mean"]
        ].copy()
        cmp_df["ecart_ressenti"] = (
            cmp_df["apparent_temperature_mean"] - cmp_df["temperature_2m_mean"]
        )
        st.line_chart(cmp_df)
    else:
        st.info("Colonnes manquantes pour comparer la température ressentie.")

elif page == "Prévisions":
    st.subheader("Prévisions")

    st.markdown("**Prévision statistique de la température moyenne**")
    with st.spinner("Calcul de la prévision à partir de l'historique multi‑années..."):
        df_pred = compute_hw_forecast(city, years=5, periods=365)
    if df_pred is None or getattr(df_pred, "empty", True):
        st.info("Prévision indisponible (historique insuffisant ou données manquantes).")
    else:
        df_plot = df_pred.copy()
        if "date" in df_plot.columns:
            df_plot = df_plot.set_index("date").sort_index()
        st.line_chart(df_plot[["temperature_2m_mean_predite"]])
        with st.expander("Données de prévision (quotidiennes)"):
            st.dataframe(df_pred, use_container_width=True)

elif page == "J vs N-1":
    st.subheader("Comparaison aujourd'hui vs année dernière")
    
    with st.spinner(f"Chargement des données pour {city}..."):
        df_today, df_last_year = fetch_today_vs_last_year(city)
    
    if df_today is None or df_today.empty:
        st.error("Impossible de récupérer les données d'aujourd'hui.")
    elif df_last_year is None or df_last_year.empty:
        st.warning("Données d'aujourd'hui disponibles, mais impossible de récupérer les données de l'année dernière.")
        st.info("**Données d'aujourd'hui**")
        st.dataframe(df_today, use_container_width=True)
    else:
        if "sunshine_duration" in df_today.columns:
            df_today["sunshine_hours"] = df_today["sunshine_duration"] / 3600.0
        if "sunshine_duration" in df_last_year.columns:
            df_last_year["sunshine_hours"] = df_last_year["sunshine_duration"] / 3600.0
        
        cc1, cc2, cc3, cc4 = st.columns(4)
        
        if "temperature_2m_mean" in df_today.columns and "temperature_2m_mean" in df_last_year.columns:
            temp_today = float(df_today["temperature_2m_mean"].iloc[0])
            temp_last_year = float(df_last_year["temperature_2m_mean"].iloc[0])
            diff_temp = temp_today - temp_last_year
            cc1.metric("Temp. moyenne (°C)", f"{temp_today:.1f}", f"{diff_temp:+.1f}°C vs N-1")
        
        if "temperature_2m_max" in df_today.columns and "temperature_2m_max" in df_last_year.columns:
            tmax_today = float(df_today["temperature_2m_max"].iloc[0])
            tmax_last_year = float(df_last_year["temperature_2m_max"].iloc[0])
            diff_tmax = tmax_today - tmax_last_year
            cc2.metric("Temp. max (°C)", f"{tmax_today:.1f}", f"{diff_tmax:+.1f}°C vs N-1")
        
        if "precipitation_sum" in df_today.columns and "precipitation_sum" in df_last_year.columns:
            precip_today = float(df_today["precipitation_sum"].iloc[0])
            precip_last_year = float(df_last_year["precipitation_sum"].iloc[0])
            diff_precip = precip_today - precip_last_year
            cc3.metric("Précipitations (mm)", f"{precip_today:.1f}", f"{diff_precip:+.1f} mm vs N-1")
        
        if "sunshine_hours" in df_today.columns and "sunshine_hours" in df_last_year.columns:
            sun_today = float(df_today["sunshine_hours"].iloc[0])
            sun_last_year = float(df_last_year["sunshine_hours"].iloc[0])
            diff_sun = sun_today - sun_last_year
            cc4.metric("Ensoleillement (h)", f"{sun_today:.1f}", f"{diff_sun:+.1f}h vs N-1")
        
        st.divider()
        st.subheader("🚨 Alertes météorologiques")
        
        def show_alert(emoji, title, level, message, color):
            if level:
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: {color}; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #000;">{emoji} {title}</h4>
                    <p style="margin: 5px 0 0 0; color: #000;"><strong>{level}:</strong> {message}</p>
                </div>
                """, unsafe_allow_html=True)
        
        alerts = []
        
        if "temperature_2m_max" in df_today.columns:
            temp_max = float(df_today["temperature_2m_max"].iloc[0])
            if temp_max >= 38:
                alerts.append(("🌡️", "Alerte chaleur extrême", "Extrême", 
                              "Alerte canicule : restez au frais, surveillez les personnes vulnérables.", "#ff4444"))
            elif temp_max >= 35:
                alerts.append(("🌡️", "Alerte chaleur extrême", "Élevé", 
                              "Risque de coup de chaleur. Évitez les activités physiques.", "#ff9933"))
            elif temp_max >= 30:
                alerts.append(("🌡️", "Alerte chaleur extrême", "Modéré", 
                              "Chaleur importante prévue aujourd'hui. Hydratez-vous.", "#ffdd44"))
        
        rain_24h = 0
        if "rain_sum" in df_today.columns:
            rain_24h = float(df_today["rain_sum"].iloc[0])
        elif "precipitation_sum" in df_today.columns:
            rain_24h = float(df_today["precipitation_sum"].iloc[0])
        
        if rain_24h > 80:
            alerts.append(("🌧️", "Alerte pluie intense / risque d'inondation locale", "Extrême", 
                          "Risque d'inondation localisée.", "#ff4444"))
        elif rain_24h > 40:
            alerts.append(("🌧️", "Alerte pluie intense / risque d'inondation locale", "Fort", 
                          "Fortes pluies : vigilance sur les routes.", "#ff9933"))
        elif rain_24h > 20:
            alerts.append(("🌧️", "Alerte pluie intense / risque d'inondation locale", "Risque modéré", 
                          "Pluies modérées attendues.", "#ffdd44"))
        
        # 3. Alerte vent violent
        if "wind_gusts_10m_mean" in df_today.columns:
            wind_gust = float(df_today["wind_gusts_10m_mean"].iloc[0]) * 3.6  # m/s to km/h
            if wind_gust > 100:
                alerts.append(("💨", "Alerte vent violent", "Violent", 
                              "Risque de dégâts : évitez les déplacements.", "#ff4444"))
            elif wind_gust > 70:
                alerts.append(("💨", "Alerte vent violent", "Fort", 
                              "Rafales fortes : attention aux objets légers.", "#ff9933"))
        elif "wind_speed_10m_mean" in df_today.columns:
            wind_speed = float(df_today["wind_speed_10m_mean"].iloc[0]) * 3.6  # m/s to km/h
            if wind_speed > 40:
                alerts.append(("💨", "Alerte vent violent", "Modéré", 
                              "Vent soutenu prévu.", "#ffdd44"))
        
        # 4. Alerte froid / gel
        if "temperature_2m_min" in df_today.columns:
            temp_min = float(df_today["temperature_2m_min"].iloc[0])
            if temp_min < -5:
                alerts.append(("❄️", "Alerte froid / gel", "Froid intense", 
                              "Grand froid : prudence à l'extérieur.", "#ff4444"))
            elif temp_min < 0:
                alerts.append(("❄️", "Alerte froid / gel", "Gel possible", 
                              "Risque de gel : protégez les plantes et canalisations.", "#ff9933"))
            elif temp_min < 5:
                alerts.append(("❄️", "Alerte froid / gel", "Frais", 
                              "Températures basses.", "#ffdd44"))
        
        # Afficher les alertes
        if alerts:
            for emoji, title, level, message, color in alerts:
                show_alert(emoji, title, level, message, color)
        else:
            st.info("✅ Aucune alerte météorologique pour aujourd'hui.")


elif page == "ACP":
    st.subheader("ACP – analyse en composantes principales")

    # Adapter le DataFrame courant au format attendu par statistiques.acp_temperature
    df_acp = df.copy()
    if "time" in df_acp.columns and "date" not in df_acp.columns:
        df_acp["date"] = df_acp["time"]

    # Appel à la fonction ACP (retours DataFrames uniquement)
    try:
        df_pcs, loadings, explained_var = acp_temperature(df_acp, start_str, end_str)
    except Exception as e:
        st.error(f"Erreur lors du calcul de l'ACP: {e}")
        df_pcs, loadings, explained_var = None, None, None

    if {"PC1", "PC2"}.issubset(loadings.columns):
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
            ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.05, fc="tab:blue", ec="tab:blue", length_includes_head=True)
            ax.text(x * 1.07, y * 1.07, var, fontsize=10, color="tab:blue")
        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var. expl.)")
        if len(explained_var) > 1:
            ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var. expl.)")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Cercle des corrélations")
        st.pyplot(fig, use_container_width=True)

        # Loadings triés par importance
        st.markdown("**Contributions des variables (loadings)**")
        load_sorted = loadings.copy()
        # Ordonner par l'importance absolue sur PC1 puis PC2 si disponibles
        sort_cols = []
        if "PC1" in load_sorted.columns:
            sort_cols.append(load_sorted["PC1"].abs())
        if "PC2" in load_sorted.columns:
            sort_cols.append(load_sorted["PC2"].abs())
        if sort_cols:
            # Créer une clé de tri combinée en concaténant (PC1 abs, PC2 abs)
            key = sum((col.rank(ascending=False) for col in sort_cols))
            load_sorted = load_sorted.loc[key.sort_values().index]
        st.dataframe(load_sorted.style.format("{:.3f}"), use_container_width=True)
