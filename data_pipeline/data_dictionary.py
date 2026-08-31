"""Generate a compact data dictionary for the benchmark matrix."""

from __future__ import annotations

import pandas as pd

from .config import get_config

ID_COLUMNS = {"FIPS", "year", "state_fips"}
TARGET_COLUMNS = {"yield_bu_acre", "yield_trend", "yield_anomaly", "yield_trend_past"}


def build_data_dictionary(matrix_path=None) -> pd.DataFrame:
    """Return one row per column in the benchmark matrix.

    The dictionary is intentionally lightweight and rule-based. It gives users a
    reliable map of feature groups without hand-maintaining 600+ descriptions.
    """
    cfg = get_config()
    path = matrix_path or cfg.data_dir / "benchmark_matrix.csv"
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    rows = [describe_column(column) for column in columns]
    return pd.DataFrame(rows)


def describe_column(column: str) -> dict[str, str]:
    """Describe one benchmark column."""
    group, source, timing = classify_column(column)
    return {
        "column": column,
        "role": role_for(column),
        "feature_group": group,
        "source": source,
        "timing": timing,
        "notes": notes_for(column),
    }


def role_for(column: str) -> str:
    if column in ID_COLUMNS:
        return "identifier"
    if column == "yield_bu_acre":
        return "target"
    if column in TARGET_COLUMNS:
        return "diagnostic_or_baseline"
    return "feature"


def classify_column(column: str) -> tuple[str, str, str]:
    if column in ID_COLUMNS:
        return "identifier", "dataset", "static"
    if column in {"yield_bu_acre", "yield_trend", "yield_anomaly", "yield_trend_past"}:
        return "yield", "USDA NASS Quick Stats", "annual"
    if column.startswith("nass_wk_") or column.startswith("nass_cond_"):
        return "crop_progress_condition", "USDA NASS Quick Stats", "weekly_to_state_year"
    if column.startswith(("tmax_", "tmin_", "precip_", "vpd_", "srad_", "eto_")):
        return "weather", "GRIDMET", "monthly"
    if column.startswith(("gdd_", "kdd_", "edd_")):
        return "heat_stress", "GRIDMET", "monthly"
    if column.startswith(("spei90_", "spi90_", "pdsi_")):
        return "drought", "GRIDMET drought", "monthly"
    if column.startswith(("ndvi_", "evi_", "ndwi_", "gci_", "lai_", "fpar_", "lst_day_", "lst_night_")):
        return "satellite_vegetation", "MODIS", "monthly"
    if column.startswith("soil_"):
        return "soil", "POLARIS / soil products", "static"
    if column.startswith("weekly_") or "_week" in column:
        return "weekly_weather", "GRIDMET", "weekly"
    if column.startswith("yield_lag_") or column.startswith("anomaly_lag"):
        return "historical_yield", "USDA NASS Quick Stats", "past_years_only"
    if column.startswith("lgrip_") or column.startswith("irrig_"):
        return "irrigation", "LGRIP30 / USDA Census", "static_or_census_year"
    if column in {"lat", "lon", "area_km2", "elevation_m"}:
        return "geography", "Census / SRTM", "static"
    return "other", "derived", "varies"


def notes_for(column: str) -> str:
    if column == "yield_bu_acre":
        return "Main prediction target in all cleaned benchmark experiments."
    if column == "yield_trend":
        return "Full-record descriptive trend; do not use as operational forecast input."
    if column == "yield_anomaly":
        return "Raw yield minus full-record trend; used for diagnostics/drought interpretation."
    if column == "yield_trend_past":
        return "Past-only trend feature generated inside reproducible_benchmark.data."
    if column.startswith("yield_lag_") or column.startswith("anomaly_lag"):
        return "Computed using only previous years for the same reporting unit."
    return ""
