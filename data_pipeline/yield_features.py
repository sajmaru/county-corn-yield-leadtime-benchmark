"""Yield-table construction from USDA NASS records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig, get_config
from .nass import quickstats_get, state_fips_codes, value_to_float


def fetch_county_corn_yield(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Fetch county corn grain yield from NASS Quick Stats.

    The result keeps one row per NASS county/county-equivalent reporting unit and
    year. When multiple production-practice rows exist, the all-practice row is
    preferred so irrigated and non-irrigated records are not double counted.
    """
    cfg = config or get_config()
    frames: list[pd.DataFrame] = []
    base = {
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "COUNTY",
        "year__GE": str(cfg.start_year),
        "year__LE": str(cfg.end_year),
    }

    for state_fips in state_fips_codes():
        print(f"Fetching NASS county yield for state {state_fips}...")
        frame = quickstats_get(
            {**base, "state_fips_code": state_fips},
            cache_name=f"yield_county_{state_fips}_{cfg.start_year}_{cfg.end_year}",
            config=cfg,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No NASS yield rows returned.")

    raw = pd.concat(frames, ignore_index=True)
    keep = ["year", "state_fips_code", "county_code", "Value"]
    keep += [c for c in ("prac_desc", "class_desc") if c in raw.columns]
    data = raw[keep].copy()
    data = data.rename(
        columns={
            "state_fips_code": "sfips",
            "county_code": "cfips",
            "Value": "yield_bu_acre",
        }
    )
    data["FIPS"] = data["sfips"].astype(str).str.zfill(2) + data["cfips"].astype(str).str.zfill(3)
    data["year"] = data["year"].astype(int)
    data["yield_bu_acre"] = data["yield_bu_acre"].apply(value_to_float)
    data = data.dropna(subset=["yield_bu_acre"])

    if "prac_desc" in data.columns:
        data["_practice_rank"] = data["prac_desc"].map(_practice_rank)
        data = data.sort_values("_practice_rank")
        data = data.drop_duplicates(["FIPS", "year"], keep="first")
        data = data.drop(columns="_practice_rank")
    else:
        data = data.drop_duplicates(["FIPS", "year"], keep="first")

    return data.sort_values(["FIPS", "year"]).reset_index(drop=True)


def add_full_record_trend_and_anomaly(data: pd.DataFrame, min_years: int = 5) -> pd.DataFrame:
    """Add descriptive full-record trend and anomaly columns.

    These columns reproduce the historical dataset decomposition. They are useful
    for diagnostics and drought interpretation. Forecast experiments should use
    the past-only trend feature generated in `reproducible_benchmark.data`.
    """
    records: list[pd.DataFrame] = []
    for _, group in data.groupby("FIPS", sort=True):
        group = group.sort_values("year").copy()
        if len(group) < min_years:
            continue
        x = group["year"].to_numpy(dtype=float)
        y = group["yield_bu_acre"].to_numpy(dtype=float)
        centered = x - x.mean()
        coef = np.polyfit(centered, y, deg=1)
        trend = np.polyval(coef, centered)
        group["yield_trend"] = trend
        group["yield_anomaly"] = y - trend
        records.append(group)

    if not records:
        raise RuntimeError("No county records had enough years to detrend.")
    return pd.concat(records, ignore_index=True)


def build_historical_yield_features(yield_data: pd.DataFrame) -> pd.DataFrame:
    """Build lagged yield features using only years before each row."""
    data = yield_data.copy()
    data["FIPS"] = data["FIPS"].astype(str).str.zfill(5)
    data = data.sort_values(["FIPS", "year"])
    grouped = data.groupby("FIPS", group_keys=False)["yield_bu_acre"]
    data["yield_lag_mean5"] = grouped.apply(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean()
    )
    data["yield_lag_std5"] = grouped.apply(
        lambda s: s.shift(1).rolling(5, min_periods=3).std()
    )
    data["yield_lag_trend"] = grouped.apply(
        lambda s: s.shift(1).rolling(5, min_periods=3).apply(_linear_slope, raw=False)
    )

    if "yield_anomaly" in data.columns:
        anomaly_grouped = data.groupby("FIPS", group_keys=False)["yield_anomaly"]
        data["anomaly_lag1"] = anomaly_grouped.shift(1)
        data["anomaly_lag_std3"] = anomaly_grouped.apply(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )

    columns = [
        "FIPS",
        "year",
        "yield_lag_mean5",
        "yield_lag_std5",
        "yield_lag_trend",
        "anomaly_lag1",
        "anomaly_lag_std3",
    ]
    return data[[c for c in columns if c in data.columns]].copy()


def _practice_rank(value) -> int:
    text = str(value).upper()
    if text == "ALL PRODUCTION PRACTICES":
        return 0
    if text in {"ALL CLASSES", "ALL"}:
        return 1
    return 2


def _linear_slope(values: pd.Series) -> float:
    observed = values.dropna()
    if len(observed) < 3:
        return np.nan
    x = np.arange(len(values), dtype=float)
    mask = values.notna().to_numpy()
    return float(np.polyfit(x[mask], values.to_numpy(dtype=float)[mask], deg=1)[0])
