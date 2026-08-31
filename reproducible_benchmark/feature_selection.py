"""Nested, training-fold-only feature selection utilities.

The key rule in this module is simple: feature ranking must see only the
training rows from the current outer cross-validation fold. The held-out year or
state is not used for ranking, imputation, or model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from .config import PAST_TREND, RANDOM_SEED, TARGET
from .features import NASS_CORE, unique, valid

SELECTION_METHOD = "nested_rf_treeshap_top81_v1"
DEFAULT_MAX_FEATURES = 81
DEFAULT_SHAP_SAMPLE = 400

ALLOWED_YIELD_FEATURES = {
    PAST_TREND,
    "yield_lag_mean5",
    "yield_lag_std5",
    "yield_lag_trend",
}

FORBIDDEN_COLUMNS = {
    "FIPS",
    "year",
    "state_fips",
    TARGET,
    "yield_trend",
    "yield_anomaly",
    "anomaly_lag1",
    "anomaly_lag_std3",
}

WEATHER_PREFIXES = ("tmax_", "tmin_", "precip_", "vpd_", "srad_", "eto_")
HEAT_PREFIXES = ("gdd_", "kdd_", "edd_")
DROUGHT_PREFIXES = ("spei90_", "spi90_", "pdsi_")
SATELLITE_PREFIXES = (
    "ndvi_",
    "evi_",
    "ndwi_",
    "gci_",
    "lai_",
    "fpar_",
    "lst_day_",
    "lst_night_",
)
SOIL_PREFIXES = ("soil_",)
IRRIGATION_PREFIXES = ("lgrip_", "irrig_")
GEOGRAPHY_COLUMNS = ("lat", "lon", "area_km2", "elevation_m")
HISTORY_COLUMNS = (PAST_TREND, "yield_lag_mean5", "yield_lag_std5", "yield_lag_trend")


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for nested feature selection."""

    max_features: int = DEFAULT_MAX_FEATURES
    shap_sample_size: int = DEFAULT_SHAP_SAMPLE
    n_estimators: int = 25
    max_depth: int = 10
    min_samples_leaf: int = 20
    random_state: int = RANDOM_SEED


def clean_candidate_features(data: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> list[str]:
    """Return valid candidate predictors after removing forbidden columns."""
    cleaned: list[str] = []
    for feature in valid(data, list(candidates)):
        if is_forbidden_feature(feature):
            continue
        cleaned.append(feature)
    return unique(cleaned)


def is_forbidden_feature(feature: str) -> bool:
    """Return True if a column must never be used as a predictor."""
    if feature in FORBIDDEN_COLUMNS:
        return True
    if feature.startswith("anomaly_"):
        return True
    if feature.startswith("yield_") and feature not in ALLOWED_YIELD_FEATURES:
        return True
    return False


def select_features_train_only(
    train: pd.DataFrame,
    candidates: list[str],
    target: str = TARGET,
    config: SelectionConfig | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select top features using only rows from one training fold.

    If the candidate pool has at most ``max_features`` columns, all candidates are
    returned in their existing order and a ranking table is still emitted.
    """
    cfg = config or SelectionConfig()
    candidates = clean_candidate_features(train, candidates)
    if not candidates:
        raise ValueError("No candidate features remain after leakage filtering.")

    if len(candidates) <= cfg.max_features:
        ranking = pd.DataFrame(
            {
                "feature": candidates,
                "rank": range(1, len(candidates) + 1),
                "importance": np.nan,
                "selected": True,
                "selection_method": SELECTION_METHOD,
            }
        )
        return candidates, ranking

    x_train, y_train = _prepare_selection_arrays(train, candidates, target)
    model = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    model.fit(x_train, y_train)

    sample = _sample_rows(x_train, cfg.shap_sample_size, cfg.random_state)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample, check_additivity=False)
    importance = np.abs(shap_values).mean(axis=0)
    order = np.argsort(importance)[::-1]

    ranking = pd.DataFrame(
        {
            "feature": [candidates[i] for i in order],
            "rank": range(1, len(candidates) + 1),
            "importance": importance[order],
        }
    )
    ranking["selected"] = ranking["rank"] <= cfg.max_features
    ranking["selection_method"] = SELECTION_METHOD
    selected = ranking.loc[ranking["selected"], "feature"].tolist()
    return selected, ranking


def _prepare_selection_arrays(
    train: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build selection arrays with training-fold-only imputation."""
    x_train = train[features].to_numpy(dtype=float, copy=True)
    y_train = train[target].to_numpy(dtype=float, copy=True)
    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    for col in range(x_train.shape[1]):
        x_train[np.isnan(x_train[:, col]), col] = means[col]
    return x_train, y_train


def _sample_rows(x_train: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Return a deterministic training-only sample for SHAP."""
    if x_train.shape[0] <= sample_size:
        return x_train
    rng = np.random.default_rng(seed)
    idx = rng.choice(x_train.shape[0], size=sample_size, replace=False)
    return x_train[idx]


def end_of_season_candidate_features(data: pd.DataFrame, with_nass: bool = True) -> list[str]:
    """Return a broad end-of-season candidate pool for nested selection."""
    features = [c for c in data.columns if _is_model_feature(c)]
    if not with_nass:
        features = [c for c in features if not c.startswith("nass_")]
    return clean_candidate_features(data, features)


def ablation_candidate_steps(data: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Return cumulative broad source groups for nested ablation."""
    groups: list[tuple[str, list[str]]] = []
    groups.append(("A0: Past-only trend", [PAST_TREND]))
    groups.append(("A1: + Weather", _cols_by_prefix(data, WEATHER_PREFIXES)))
    groups.append(("A2: + Heat stress", _cols_by_prefix(data, HEAT_PREFIXES)))
    groups.append(("A3: + Drought (GRIDMET)", _cols_by_prefix(data, DROUGHT_PREFIXES)))
    groups.append(("A4: + Satellite vegetation", _cols_by_prefix(data, SATELLITE_PREFIXES)))
    groups.append(("A5: + Soil", _cols_by_prefix(data, SOIL_PREFIXES)))
    groups.append(("A6: + Geography and irrigation", list(GEOGRAPHY_COLUMNS) + _cols_by_prefix(data, IRRIGATION_PREFIXES)))
    groups.append(("A7: + Historical yield", ["yield_lag_mean5", "yield_lag_std5", "yield_lag_trend"]))
    groups.append(("A8: + USDA crop condition", NASS_CORE + ["nass_wk_plant90", "nass_cond_aug_gex", "nass_cond_min_gex"]))

    cumulative: list[str] = []
    steps: list[tuple[str, list[str]]] = []
    for label, columns in groups:
        cumulative = unique(cumulative + columns)
        steps.append((label, clean_candidate_features(data, cumulative)))
    return steps


def _is_model_feature(column: str) -> bool:
    if is_forbidden_feature(column):
        return False
    if _is_weekly_weather(column) or _is_after_harvest_month(column):
        return False
    return column.startswith(
        WEATHER_PREFIXES
        + HEAT_PREFIXES
        + DROUGHT_PREFIXES
        + SATELLITE_PREFIXES
        + SOIL_PREFIXES
        + IRRIGATION_PREFIXES
        + ("nass_",)
    ) or column in set(GEOGRAPHY_COLUMNS + HISTORY_COLUMNS)


def _cols_by_prefix(data: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [
        c
        for c in data.columns
        if c.startswith(prefixes) and not _is_weekly_weather(c) and not _is_after_harvest_month(c)
    ]


def _is_weekly_weather(column: str) -> bool:
    return len(column) >= 4 and column[-3] == "w" and column[-2:].isdigit()


def _is_after_harvest_month(column: str) -> bool:
    prefix_tuple = WEATHER_PREFIXES + HEAT_PREFIXES + DROUGHT_PREFIXES + SATELLITE_PREFIXES
    if not column.startswith(prefix_tuple):
        return False
    try:
        month = int(column.rsplit("_", 1)[-1])
    except ValueError:
        return False
    return month > 10
