"""Composants Streamlit réutilisables pour l'affichage des alertes météorologiques."""
import streamlit as st
from services.analytics.weather_alerts import WeatherAlert, WeatherAlertService


def render_alert(alert: WeatherAlert):
    """Affiche une alerte météorologique individuelle."""
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {alert.color}; margin-bottom: 10px;">
        <h4 style="margin: 0; color: #000;">{alert.emoji} {alert.title}</h4>
        <p style="margin: 5px 0 0 0; color: #000;"><strong>{alert.level}:</strong> {alert.message}</p>
    </div>
    """, unsafe_allow_html=True)


def render_weather_alerts(alerts: list[WeatherAlert]):
    """Affiche toutes les alertes météorologiques."""
    if not alerts:
        st.info("✅ Aucune alerte météorologique pour aujourd'hui.")
        return
    
    for alert in alerts:
        render_alert(alert)


def render_alerts_section(df_today, alert_service: WeatherAlertService):
    """Affiche la section complète des alertes météorologiques."""
    st.subheader("🚨 Alertes météorologiques")
    alerts = alert_service.evaluate_alerts(df_today)
    render_weather_alerts(alerts)

