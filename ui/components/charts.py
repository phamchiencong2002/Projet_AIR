"""Composants Streamlit réutilisables pour l'affichage des graphiques météorologiques."""
import streamlit as st
import pandas as pd
from services.presentation.weather_presenter import WeatherPresenter


def render_temperature_chart(df, presenter: WeatherPresenter):
    """Affiche le graphique des températures."""
    st.markdown("**Température (moy/max/min)**")
    chart_data = presenter.prepare_temperature_chart_data(df)
    if chart_data is not None:
        st.line_chart(chart_data)
    else:
        st.info("Pas de colonnes température disponibles.")


def render_precipitation_chart(df, presenter: WeatherPresenter):
    """Affiche le graphique des précipitations."""
    st.markdown("**Cumul précipitations**")
    chart_data = presenter.prepare_precipitation_chart_data(df)
    if chart_data is not None:
        st.bar_chart(chart_data)
    else:
        st.info("Pas de données de précipitations.")


def render_temperature_comparison_chart(df, presenter: WeatherPresenter):
    """Affiche le graphique de comparaison température réelle vs ressentie."""
    st.markdown("**Température réelle vs ressentie**")
    chart_data = presenter.prepare_temperature_comparison_data(df)
    if chart_data is not None:
        st.line_chart(chart_data)
    else:
        st.info("Colonnes manquantes pour comparer la température ressentie.")


def render_forecast_chart(df_forecast):
    """Affiche le graphique de prévision."""
    if df_forecast is None or df_forecast.empty:
        st.info("Prévision indisponible (historique insuffisant ou données manquantes).")
        return
    
    df_plot = df_forecast.copy()
    if "date" in df_plot.columns:
        df_plot = df_plot.set_index("date").sort_index()
    
    if "temperature_2m_mean_predite" in df_plot.columns:
        st.line_chart(df_plot[["temperature_2m_mean_predite"]])

