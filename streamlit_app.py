import streamlit as st
import pandas as pd
from datetime import date, timedelta
import numpy as np

# ==== API LAYER (reuse your repo helpers) ====
from DataCollecte.api_client import (
    get_geocoding_data,
    get_daily_weather_data,
    get_forecast_today,
)
from main import get_today_vs_last_year

# ============================================
#              HELPER FUNCTIONS
# ============================================

def _json_daily_to_df(j):
    """Convert Open‑Meteo daily JSON to a tidy pandas DataFrame.
    Expects keys: 'daily' (dict with 'time' and variables) and optionally 'daily_units'.
    """
    if not j or "daily" not in j or "time" not in j["daily"]:
        return pd.DataFrame()
    df = pd.DataFrame(j["daily"]).copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])  # YYYY‑MM‑DD
        df = df.sort_values("time").reset_index(drop=True)
    return df

@st.cache_data(ttl=900)
def fetch_geocode(city: str):
    return get_geocoding_data(city)

@st.cache_data(ttl=900)
def fetch_daily(geoloc: dict, start_str: str, end_str: str):
    return get_daily_weather_data(geoloc, start_str, end_str)

@st.cache_data(ttl=600)
def fetch_forecast(geoloc: dict):
    """Wrapper for your repo's forecast helper. Expected to return JSON with 'daily' and/or 'hourly'."""
    return get_forecast_today(geoloc)

# Safe stats helpers

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
    # SVD on standardized data (equivalent to PCA)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Explained variance ratio
    var = (S ** 2) / (Xc.shape[0] - 1)
    evr = var / var.sum()
    # Scores and loadings
    scores = U @ np.diag(S)  # rows x comps
    loadings = Vt.T          # cols x comps
    # Trim to n_components
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
# ============================================
#                  UI LAYOUT
# ============================================

st.set_page_config(page_title="Projet AIR – Météo", layout="wide")

# -- TOP BAR / HEADER --
st.title("🌤️ Projet AIR – Dashboard météo")

# Sidebar (navigation style like your sketch)
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
    seuil_pluie = st.number_input("Seuil précipitations (mm)", value=10.0, step=0.5)
    seuil_uv = st.number_input("Seuil ensoleillement (h)", value=8.0, step=0.5)

# -- DATA FETCH (shared for pages that need it) --
geoloc = fetch_geocode(city) if city else None
if not geoloc:
    st.info("Saisissez une ville valide dans la barre latérale pour commencer.")
    st.stop()

start_str = start_dt.strftime("%Y-%m-%d")
end_str = end_dt.strftime("%Y-%m-%d")

j = fetch_daily(geoloc, start_str, end_str)
df = _json_daily_to_df(j)

# Convert sunshine from seconds to hours if present
if "sunshine_duration" in df.columns:
    df["sunshine_hours"] = df["sunshine_duration"] / 3600.0


# ============================================
#                   PAGES
# ============================================

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

    st.subheader("Courbes principales")

    left, right = st.columns([2, 1])
    with left:
        # Combo: rentrée vs Noël (proxy: montrer deux périodes côte à côte)
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

    with right:
        st.markdown("**Temps d'ensoleillement vs seuil (alerte)**")
        if "sunshine_hours" in df:
            last = float(df["sunshine_hours"].iloc[-1])
            delta = last - seuil_uv
            st.metric("Ensoleillement (dernier jour, h)", f"{last:.1f}", f"{delta:+.1f} vs seuil")
            st.progress(min(1.0, max(0.0, last / max(seuil_uv, 0.0001))), text="Avancement vers le seuil")
        else:
            st.info("Pas de données d'ensoleillement.")

elif page == "Prévisions":
    st.subheader("Prévisions")

    jf = fetch_forecast(geoloc)
    if not jf:
        st.error("Impossible de récupérer les prévisions.")
    else:
        # ---- DAILY FORECAST ----
        dfF_day = _json_daily_to_df(jf)
        if "sunshine_duration" in dfF_day.columns:
            dfF_day["sunshine_hours"] = dfF_day["sunshine_duration"] / 3600.0

        if not dfF_day.empty:
            st.markdown("**Prévisions quotidiennes**")
            kc1, kc2, kc3, kc4 = st.columns(4)
            if "temperature_2m_max" in dfF_day:
                kc1.metric("Temp. max demain (°C)", f"{float(dfF_day['temperature_2m_max'].iloc[0]):.1f}")
            if "temperature_2m_min" in dfF_day:
                kc2.metric("Temp. min demain (°C)", f"{float(dfF_day['temperature_2m_min'].iloc[0]):.1f}")
            if "precipitation_sum" in dfF_day:
                kc3.metric("Pluie (demain, mm)", f"{float(dfF_day['precipitation_sum'].iloc[0]):.1f}")
            if "wind_speed_10m_max" in dfF_day:
                kc4.metric("Vent max (demain, m/s)", f"{float(dfF_day['wind_speed_10m_max'].iloc[0]):.1f}")

            chart_cols = [c for c in [
                "temperature_2m_max","temperature_2m_min","precipitation_sum","sunshine_hours","wind_speed_10m_max","shortwave_radiation_sum"
            ] if c in dfF_day.columns]

            if chart_cols:
                st.line_chart(dfF_day.set_index("time")[chart_cols])
            with st.expander("Données brutes (daily)"):
                st.dataframe(dfF_day, use_container_width=True)

        # ---- HOURLY FORECAST (optional) ----
        if "hourly" in jf and isinstance(jf["hourly"], dict) and "time" in jf["hourly"]:
            dfF_hr = pd.DataFrame(jf["hourly"]).copy()
            if "time" in dfF_hr.columns:
                dfF_hr["time"] = pd.to_datetime(dfF_hr["time"])  # precise timestamps
                dfF_hr = dfF_hr.set_index("time").sort_index()
            st.markdown("**Prochaines heures (aperçu)**")
            # pick a few common variables if present
            hourly_cols = [c for c in [
                "temperature_2m","relative_humidity_2m","precipitation","wind_speed_10m"
            ] if c in dfF_hr.columns]
            if hourly_cols:
                st.line_chart(dfF_hr[hourly_cols].iloc[:48])  # 48h
            with st.expander("Données brutes (hourly)"):
                st.dataframe(dfF_hr.reset_index().head(200), use_container_width=True)

elif page == "Historique 3 ans":
    st.subheader("Historique 3 ans – vue agrégée")
    # Étendre la période automatiquement à 3 ans
    start3 = (date.today() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    end3 = date.today().strftime("%Y-%m-%d")
    j3 = fetch_daily(geoloc, start3, end3)
    df3 = _json_daily_to_df(j3)
    if "sunshine_duration" in df3.columns:
        df3["sunshine_hours"] = df3["sunshine_duration"] / 3600.0
    if not df3.empty:
        st.line_chart(df3.set_index("time")[[
            c for c in ["temperature_2m_mean","precipitation_sum","sunshine_hours"] if c in df3
        ]])
        st.dataframe(df3.tail(30), use_container_width=True)
    else:
        st.info("Aucune donnée disponible sur 3 ans pour cette zone.")

elif page == "J vs N-1":
    st.subheader("Comparaison Aujourd'hui vs Année dernière")
    
    with st.spinner(f"Chargement des données pour {city}..."):
        df_today, df_last_year = get_today_vs_last_year(city)
    
    if df_today is None or df_today.empty:
        st.error("Impossible de récupérer les données d'aujourd'hui.")
    elif df_last_year is None or df_last_year.empty:
        st.warning("Données d'aujourd'hui disponibles, mais impossible de récupérer les données de l'année dernière.")
        st.info("**Données d'aujourd'hui**")
        st.dataframe(df_today, use_container_width=True)
    else:
        # Conversion sunshine en heures si nécessaire
        if "sunshine_duration" in df_today.columns:
            df_today["sunshine_hours"] = df_today["sunshine_duration"] / 3600.0
        if "sunshine_duration" in df_last_year.columns:
            df_last_year["sunshine_hours"] = df_last_year["sunshine_duration"] / 3600.0
        
        # Métriques comparatives
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
        
        # Fonction pour afficher une alerte avec couleur et émoji
        def show_alert(emoji, title, level, message, color):
            if level:
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: {color}; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #000;">{emoji} {title}</h4>
                    <p style="margin: 5px 0 0 0; color: #000;"><strong>{level}:</strong> {message}</p>
                </div>
                """, unsafe_allow_html=True)
        
        alerts = []
        
        # 1. Alerte chaleur extrême
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
        
        # 2. Alerte pluie intense / risque d'inondation locale
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

elif page == "Seuils & Alertes":
    st.subheader("Surveillance par seuils")
    if not df.empty:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Alerte précipitations**")
            if "precipitation_sum" in df:
                exceed = df[df["precipitation_sum"] >= seuil_pluie]
                st.write(f"Jours dépassant {seuil_pluie} mm: {len(exceed)}")
                st.dataframe(exceed[["time", "precipitation_sum"]])
            else:
                st.info("Pas de données de précipitations.")
        with a2:
            st.markdown("**Alerte ensoleillement**")
            if "sunshine_hours" in df:
                under = df[df["sunshine_hours"] < seuil_uv]
                st.write(f"Jours sous {seuil_uv} h: {len(under)}")
                st.dataframe(under[["time", "sunshine_hours"]])
            else:
                st.info("Pas de données d'ensoleillement.")
    else:
        st.info("Aucune donnée pour calculer les alertes.")

elif page == "ACP":
    st.subheader("ACP – Analyse en composantes principales")

    # Proposer des variables numériques pertinentes
    candidate_cols = [
        c for c in [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "sunshine_hours",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
        ] if c in df.columns
    ]
    if not candidate_cols:
        st.info("Aucune colonne numérique disponible pour l'ACP sur la période choisie.")
    else:
        sel = st.multiselect(
            "Variables à inclure",
            candidate_cols,
            default=candidate_cols,
            help="Choisissez 2+ variables pour l'ACP.",
        )
        n_comp = st.slider("Nombre de composantes", 2, min(5, len(sel)) if len(sel) >= 2 else 2, 2)
        if len(sel) < 2:
            st.warning("Sélectionnez au moins 2 variables.")
        else:
            res = pca_from_df(df, sel, n_components=n_comp)
            if not res:
                st.info("Données insuffisantes (ou trop de valeurs manquantes).")
            else:
                scores = res["scores"]
                loadings = res["loadings"]
                evr = res["evr"]
                cols_sel = res["cols"]

                # Scree plot (variance expliquée)
                st.markdown("**Variance expliquée**")
                evr_df = pd.DataFrame({"Composante": [f"PC{i+1}" for i in range(len(evr))], "Part": evr})
                st.bar_chart(evr_df.set_index("Composante"))
                st.caption("Somme variance expliquée: {:.1f}%".format(float(evr.sum() * 100)))

                # Scatter sur PC1, PC2
                if scores.shape[1] >= 2:
                    st.markdown("**Projection (PC1 x PC2)**")
                    sc_df = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"])  # lignes après dropna
                    # Tenter d'aligner l'index avec les dates conservées
                    aligned = df[sel].dropna().copy()
                    if "time" in df.columns:
                        sc_df.index = aligned.index  # mêmes lignes que dropna
                        sc_df["time"] = df.loc[aligned.index, "time"].values
                        sc_df = sc_df.set_index("time")
                    st.scatter_chart(sc_df)

                # Loadings (contributions des variables)
                st.markdown("**Contributions (loadings)**")
                load_df = pd.DataFrame(loadings, index=cols_sel, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
                st.dataframe(load_df.style.format("{:.3f}"), use_container_width=True)

