"""Composants Streamlit réutilisables pour l'affichage des métriques météorologiques."""
import streamlit as st
from typing import Optional
from services.presentation.weather_presenter import WeatherPresenter


def render_metric_card(
    label: str,
    value: Optional[float],
    formatter: callable,
    delta: Optional[str] = None,
    help_text: Optional[str] = None
):
    """Affiche une métrique dans une carte Streamlit."""
    formatted_value = formatter(value)
    st.metric(label, formatted_value, delta=delta, help=help_text)


def render_weather_metrics_grid(df, statistics_service, presenter: WeatherPresenter):
    """Affiche une grille de métriques météorologiques principales."""
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        t_mean = statistics_service.safe_mean(df, "temperature_2m_mean")
        render_metric_card(
            "Temp. moyenne (°C)",
            t_mean,
            presenter.format_temperature
        )
    
    with c2:
        t_max = statistics_service.safe_mean(df, "temperature_2m_max")
        render_metric_card(
            "Temp. max moy. (°C)",
            t_max,
            presenter.format_temperature
        )
    
    with c3:
        p_sum = statistics_service.safe_sum(df, "precipitation_sum")
        render_metric_card(
            "Précipitations totales (mm)",
            p_sum,
            presenter.format_precipitation
        )
    
    with c4:
        sun_h = statistics_service.safe_sum(df, "sunshine_hours")
        render_metric_card(
            "Ensoleillement total (h)",
            sun_h,
            presenter.format_sunshine
        )


def render_secondary_metrics_grid(df, statistics_service, presenter: WeatherPresenter):
    """Affiche une grille de métriques secondaires (pourcentages, moyennes)."""
    k1, k2, k3 = st.columns(3)
    
    # Pourcentage de jours de pluie
    with k1:
        pct_rain = statistics_service.calculate_rainy_days_percentage(
            df, "precipitation_sum", threshold_mm=1.0
        )
        render_metric_card(
            "% jours de pluie",
            pct_rain,
            presenter.format_percentage,
            help_text="Seuil: ≥ 1.0 mm/j"
        )
    
    # Ensoleillement moyen
    with k2:
        avg_sun = statistics_service.calculate_average_sunshine_hours(df)
        render_metric_card(
            "Ens. moyen (h/j)",
            avg_sun,
            presenter.format_sunshine
        )
    
    # Pourcentage de jours ensoleillés
    with k3:
        pct_sunny = statistics_service.calculate_sunny_days_percentage(
            df, "sunshine_hours", threshold_h=8.0
        )
        render_metric_card(
            "% jours ≥ 8 h ens.",
            pct_sunny,
            presenter.format_percentage,
            help_text="Seuil: ≥ 8.0 h/j"
        )


def render_comparison_metrics(
    df_today,
    df_last_year,
    statistics_service,
    presenter: WeatherPresenter
):
    """Affiche les métriques de comparaison entre aujourd'hui et l'année dernière."""
    cc1, cc2, cc3, cc4 = st.columns(4)
    
    # Température moyenne
    temp_data = statistics_service.prepare_comparison_data(
        df_today, df_last_year, "temperature_2m_mean"
    )
    if temp_data:
        temp_today, _, diff_temp = temp_data
        with cc1:
            render_metric_card(
                "Temp. moyenne (°C)",
                temp_today,
                presenter.format_temperature,
                delta=f"{presenter.format_delta(diff_temp, '°C')} vs N-1"
            )
    
    # Température max
    tmax_data = statistics_service.prepare_comparison_data(
        df_today, df_last_year, "temperature_2m_max"
    )
    if tmax_data:
        tmax_today, _, diff_tmax = tmax_data
        with cc2:
            render_metric_card(
                "Temp. max (°C)",
                tmax_today,
                presenter.format_temperature,
                delta=f"{presenter.format_delta(diff_tmax, '°C')} vs N-1"
            )
    
    # Précipitations
    precip_data = statistics_service.prepare_comparison_data(
        df_today, df_last_year, "precipitation_sum"
    )
    if precip_data:
        precip_today, _, diff_precip = precip_data
        with cc3:
            render_metric_card(
                "Précipitations (mm)",
                precip_today,
                presenter.format_precipitation,
                delta=f"{presenter.format_delta(diff_precip, ' mm')} vs N-1"
            )
    
    # Ensoleillement
    sun_data = statistics_service.prepare_comparison_data(
        df_today, df_last_year, "sunshine_hours"
    )
    if sun_data:
        sun_today, _, diff_sun = sun_data
        with cc4:
            render_metric_card(
                "Ensoleillement (h)",
                sun_today,
                presenter.format_sunshine,
                delta=f"{presenter.format_delta(diff_sun, 'h')} vs N-1"
            )

