"""USDA NASS state-level corn progress and condition features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig, get_config
from .nass import STATE_FIPS_TO_ALPHA, quickstats_get, value_to_float


def fetch_progress_condition(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Fetch weekly state-level corn progress and condition rows."""
    cfg = config or get_config()
    frames: list[pd.DataFrame] = []
    for state_fips, state_alpha in STATE_FIPS_TO_ALPHA.items():
        for statistic in ("PROGRESS", "CONDITION"):
            print(f"Fetching NASS {statistic.lower()} for {state_alpha}...")
            frame = quickstats_get(
                {
                    "commodity_desc": "CORN",
                    "statisticcat_desc": statistic,
                    "agg_level_desc": "STATE",
                    "state_alpha": state_alpha,
                    "year__GE": str(cfg.start_year),
                    "year__LE": str(cfg.end_year),
                    "freq_desc": "WEEKLY",
                },
                cache_name=f"{statistic.lower()}_{state_fips}_{cfg.start_year}_{cfg.end_year}",
                config=cfg,
            )
            if not frame.empty:
                frame["state_fips"] = state_fips
                frames.append(frame)
    if not frames:
        raise RuntimeError("No NASS progress/condition rows returned.")
    return pd.concat(frames, ignore_index=True)


def aggregate_state_year_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert weekly NASS rows into one state-year feature row."""
    data = raw.copy()
    data["value_num"] = data["Value"].apply(value_to_float)
    data["year"] = data["year"].astype(int)
    data["week"] = data["reference_period_desc"].apply(_parse_week)

    rows: list[dict] = []
    for (state_fips, year), group in data.groupby(["state_fips", "year"], sort=True):
        row = {"state_fips": str(state_fips).zfill(2), "year": int(year)}
        row.update(_progress_features(group))
        row.update(_condition_features(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["state_fips", "year"]).reset_index(drop=True)


def _progress_features(group: pd.DataFrame) -> dict[str, float]:
    planted = _rows_matching(group, "PROGRESS, MEASURED IN PCT PLANTED").sort_values("week")
    silking = _rows_matching(group, "PROGRESS, MEASURED IN PCT SILKING").sort_values("week")
    return {
        "nass_wk_plant50": _first_week_at_or_above(planted, 50),
        "nass_wk_plant90": _first_week_at_or_above(planted, 90),
        "nass_wk_silk50": _first_week_at_or_above(silking, 50),
    }


def _condition_features(group: pd.DataFrame) -> dict[str, float]:
    good = _rows_matching(group, "CONDITION, MEASURED IN PCT GOOD")
    excellent = _rows_matching(group, "CONDITION, MEASURED IN PCT EXCELLENT")
    if good.empty or excellent.empty:
        return {}

    merged = good[["week", "value_num"]].rename(columns={"value_num": "good"}).merge(
        excellent[["week", "value_num"]].rename(columns={"value_num": "excellent"}),
        on="week",
        how="inner",
    )
    if merged.empty:
        return {}

    merged["gex"] = merged["good"] + merged["excellent"]
    july = merged[(merged["week"] >= 27) & (merged["week"] <= 30)]
    august = merged[(merged["week"] >= 31) & (merged["week"] <= 34)]
    ordered = merged.sort_values("week")
    return {
        "nass_cond_jul_gex": july["gex"].mean() if not july.empty else np.nan,
        "nass_cond_aug_gex": august["gex"].mean() if not august.empty else np.nan,
        "nass_cond_min_gex": merged["gex"].min(),
        "nass_cond_end_gex": ordered.iloc[-1]["gex"],
        "nass_cond_mean_gex": merged["gex"].mean(),
    }


def _rows_matching(group: pd.DataFrame, text: str) -> pd.DataFrame:
    return group[group["short_desc"].str.contains(text, regex=False, na=False)].copy()


def _first_week_at_or_above(data: pd.DataFrame, threshold: float) -> float:
    if data.empty:
        return np.nan
    hits = data.loc[data["value_num"] >= threshold, "week"].dropna()
    return float(hits.iloc[0]) if len(hits) else np.nan


def _parse_week(value) -> float:
    text = str(value)
    if "#" not in text:
        return np.nan
    try:
        return float(text.split("#")[-1].strip())
    except ValueError:
        return np.nan
