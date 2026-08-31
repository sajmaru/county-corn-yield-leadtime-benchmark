"""Feature definitions for paper tables and lead-time experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PAST_TREND


@dataclass(frozen=True)
class ForecastDate:
    """Feature availability definition for one forecast issue date."""

    label: str
    lead_months: int
    max_month: int
    nass_features: tuple[str, ...]
    description: str


SMART_SAT_ONLY = [
    "spei90_07",
    "spei90_08",
    "spi90_07",
    "spi90_08",
    "tmax_07",
    "kdd_07",
    "edd_07",
    "gdd_07",
    "precip_06",
    "precip_07",
    "evi_06",
    "evi_07",
    "ndvi_07",
    "pdsi_08",
]

NASS_CORE = [
    "nass_cond_end_gex",
    "nass_cond_jul_gex",
    "nass_cond_mean_gex",
    "nass_wk_plant50",
    "nass_wk_silk50",
]

NEW_VEG_INDICES = [f"ndwi_{m:02d}" for m in (6, 7, 8)] + [
    f"gci_{m:02d}" for m in (6, 7, 8)
]

WEATHER_MONTHLY = ["tmax", "precip", "vpd", "srad", "eto"]
HEATSTRESS_MONTHLY = ["gdd", "kdd", "edd"]
DROUGHT_MONTHLY = ["spei90", "spi90", "pdsi"]
SATELLITE_MONTHLY = [
    "ndvi",
    "evi",
    "ndwi",
    "gci",
    "lai",
    "fpar",
    "lst_day",
    "lst_night",
]
MONTHLY_PREFIXES = (
    WEATHER_MONTHLY + HEATSTRESS_MONTHLY + DROUGHT_MONTHLY + SATELLITE_MONTHLY
)

NASS_PLANTING = ("nass_wk_plant50", "nass_wk_plant90")
NASS_JULY = ("nass_wk_silk50", "nass_cond_jul_gex")
NASS_AUGUST = ("nass_cond_aug_gex",)
NASS_END = ("nass_cond_end_gex", "nass_cond_mean_gex", "nass_cond_min_gex")

FORECAST_DATES = [
    ForecastDate("Apr 1", 6, 0, (), "Pre-season: soil, history, trend only"),
    ForecastDate("Jun 1", 4, 5, NASS_PLANTING, "Jan-May weather and planting"),
    ForecastDate("Jul 1", 3, 6, NASS_PLANTING, "June weather and vegetation"),
    ForecastDate(
        "Aug 1",
        2,
        7,
        NASS_PLANTING + NASS_JULY,
        "Full July, silking, and July condition",
    ),
    ForecastDate(
        "Sep 1",
        1,
        8,
        NASS_PLANTING + NASS_JULY + NASS_AUGUST,
        "August weather, vegetation, and condition",
    ),
    ForecastDate(
        "End of season",
        0,
        10,
        NASS_PLANTING + NASS_JULY + NASS_AUGUST + NASS_END,
        "All season features and final condition",
    ),
]


def valid(data: pd.DataFrame, features: list[str] | tuple[str, ...]) -> list[str]:
    """Keep only features present in the input matrix, preserving order."""
    return [feature for feature in features if feature in data.columns]


def unique(features: list[str]) -> list[str]:
    """Remove duplicate feature names while preserving order."""
    return list(dict.fromkeys(features))


def polaris_soil_features(data: pd.DataFrame) -> list[str]:
    """Return POLARIS multi-depth soil features.

    The cached matrix also contains broad soil features such as ``soil_awc``.
    The paper's 81-feature core uses the 46 multi-depth POLARIS columns selected
    by the underscore rule below.
    """
    return [c for c in data.columns if c.startswith("soil_") and "_" in c[5:]]


def monthly_up_to(data: pd.DataFrame, max_month: int) -> list[str]:
    """Return monthly features available through ``max_month``."""
    features: list[str] = []
    for prefix in MONTHLY_PREFIXES:
        for month in range(1, max_month + 1):
            column = f"{prefix}_{month:02d}"
            if column in data.columns:
                features.append(column)
    return features


def static_features(data: pd.DataFrame) -> list[str]:
    """Features available before the season begins."""
    return valid(
        data,
        [PAST_TREND]
        + polaris_soil_features(data)
        + ["lat", "lon", "area_km2", "elevation_m", "lgrip_irrigated_frac"]
        + ["yield_lag_mean5", "yield_lag_std5", "yield_lag_trend"],
    )


def leadtime_features(data: pd.DataFrame, date: ForecastDate, with_nass: bool) -> list[str]:
    """Build the feature list available at one forecast date."""
    features = static_features(data) + monthly_up_to(data, date.max_month)
    if with_nass:
        features += valid(data, date.nass_features)
    return unique(features)


def core_81_features(data: pd.DataFrame, with_nass: bool = True) -> list[str]:
    """Return the paper's compact end-of-season feature core."""
    features = (
        [PAST_TREND]
        + SMART_SAT_ONLY
        + (NASS_CORE if with_nass else [])
        + NEW_VEG_INDICES
        + polaris_soil_features(data)
        + ["lat", "lon", "elevation_m", "area_km2"]
        + ["yield_lag_mean5", "yield_lag_std5", "yield_lag_trend"]
        + ["lgrip_irrigated_frac"]
    )
    return unique(valid(data, features))


def ablation_steps(data: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Progressive feature groups for the paper ablation table."""
    a0 = [PAST_TREND]
    a1 = a0 + ["tmax_07", "precip_06", "precip_07"]
    a2 = a1 + ["kdd_07", "edd_07", "gdd_07"]
    a3 = a2 + ["spei90_07", "spei90_08", "spi90_07", "spi90_08", "pdsi_08"]
    a4 = a3 + ["evi_06", "evi_07", "ndvi_07"]
    a5 = a4 + NEW_VEG_INDICES
    a6 = a5 + polaris_soil_features(data)
    a7 = a6 + ["lat", "lon", "elevation_m", "area_km2", "lgrip_irrigated_frac"]
    a8 = a7 + ["yield_lag_mean5", "yield_lag_std5", "yield_lag_trend"]
    a9 = a8 + NASS_CORE
    return [
        ("A0: Past-only trend", valid(data, a0)),
        ("A1: + Core weather", valid(data, a1)),
        ("A2: + Heat stress", valid(data, a2)),
        ("A3: + Drought (GRIDMET)", valid(data, a3)),
        ("A4: + Satellite vegetation", valid(data, a4)),
        ("A5: + Vegetation water indices", valid(data, a5)),
        ("A6: + Soil profile", valid(data, a6)),
        ("A7: + Geography & irrigation", valid(data, a7)),
        ("A8: + Historical yield", valid(data, a8)),
        ("A9: + USDA crop condition", valid(data, a9)),
    ]
