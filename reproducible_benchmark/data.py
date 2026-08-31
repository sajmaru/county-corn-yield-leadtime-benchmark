"""Data loading and leak-safe yield-history features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANOMALY, INPUT_MATRIX, PAST_TREND, TARGET, YEAR_MAX, YEAR_MIN


ID_COLUMNS = ["FIPS", "year"]
REQUIRED_COLUMNS = ["FIPS", "year", TARGET, ANOMALY]


def load_benchmark_matrix(path=INPUT_MATRIX) -> pd.DataFrame:
    """Load the cached benchmark matrix used by all reproduction scripts.

    The matrix is the clean modeling input assembled from public data products.
    It contains raw NASS yield, the descriptive full-record trend/anomaly, and
    engineered weather, satellite, soil, crop-report, geography, irrigation, and
    history features.
    """
    data = pd.read_csv(path, dtype={"FIPS": str})
    data["FIPS"] = data["FIPS"].str.zfill(5)
    data = data[(data["year"] >= YEAR_MIN) & (data["year"] <= YEAR_MAX)].copy()
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()
    data = data.sort_values(["FIPS", "year"]).reset_index(drop=True)
    data["state_fips"] = data["FIPS"].str[:2]
    return add_past_only_trend(data)


def add_past_only_trend(data: pd.DataFrame, min_years: int = 3) -> pd.DataFrame:
    """Add a county trend estimated only from years before each row.

    The original exploratory notebooks include ``yield_trend`` estimated from the
    full county record. That is fine for descriptive anomaly analysis, but not as
    a forecast input. This function creates ``yield_trend_past`` so each row uses
    only prior yields from the same county.

    For the first few years of a county record, when fewer than ``min_years``
    past observations exist, the feature is left missing and later imputed from
    the training fold only.
    """
    data = data.copy()
    data[PAST_TREND] = np.nan

    for _, idx in data.groupby("FIPS").groups.items():
        g = data.loc[idx].sort_values("year")
        years = g["year"].to_numpy(dtype=float)
        yields = g[TARGET].to_numpy(dtype=float)
        trend_values = np.full(len(g), np.nan, dtype=float)

        for i, year in enumerate(years):
            prev_mask = years < year
            if prev_mask.sum() < min_years:
                continue
            x = years[prev_mask]
            y = yields[prev_mask]
            x_centered = x - x.mean()
            coef = np.polyfit(x_centered, y, deg=1)
            trend_values[i] = np.polyval(coef, year - x.mean())

        data.loc[g.index, PAST_TREND] = trend_values

    return data


def dataset_summary(data: pd.DataFrame) -> dict[str, float | int]:
    """Return the headline dataset counts used in the manuscript."""
    counts = data.groupby("FIPS").size()
    return {
        "rows": int(len(data)),
        "counties_or_reporting_units": int(data["FIPS"].nunique()),
        "years": int(data["year"].nunique()),
        "year_min": int(data["year"].min()),
        "year_max": int(data["year"].max()),
        "mean_yield": float(data[TARGET].mean()),
        "std_yield": float(data[TARGET].std()),
        "median_years_per_unit": float(counts.median()),
        "units_ge_10_years": int((counts >= 10).sum()),
        "units_lt_10_years": int((counts < 10).sum()),
    }


def state_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize observations, reporting units, mean yield, and SD by state."""
    return (
        data.assign(state=data["FIPS"].str[:2])
        .groupby("state")
        .agg(
            counties=("FIPS", "nunique"),
            obs=(TARGET, "size"),
            mean=(TARGET, "mean"),
            std=(TARGET, "std"),
        )
        .reset_index()
    )
