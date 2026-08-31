"""Small USDA NASS Quick Stats helpers.

These functions intentionally stay lightweight. They wrap the API, cache raw
responses, and convert NASS value strings to numbers. Product-specific scripts
handle the actual feature engineering.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import STATES, PipelineConfig, ensure_dirs, get_config
from .credentials import NASS_API_URL, require_nass_key

STATE_FIPS_TO_ALPHA = {
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "26": "MI",
    "27": "MN",
    "29": "MO",
    "31": "NE",
    "38": "ND",
    "39": "OH",
    "46": "SD",
    "55": "WI",
}

SUPPRESSED_VALUES = {"(D)", "(Z)", "(NA)", "(S)", "(X)", ""}


def value_to_float(value) -> float:
    """Convert NASS `Value` strings to floats, preserving suppressions as NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    if text in SUPPRESSED_VALUES:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def quickstats_get(
    params: dict,
    cache_name: str | None = None,
    config: PipelineConfig | None = None,
    max_retries: int = 3,
    sleep_seconds: float = 0.3,
) -> pd.DataFrame:
    """Call NASS Quick Stats and return the `data` array as a DataFrame.

    If `cache_name` is supplied, responses are cached under
    `data_pipeline/cache/nass/` so interrupted runs can resume.
    """
    cfg = config or get_config()
    ensure_dirs(cfg)
    cache_dir = cfg.cache_dir / "nass"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_name}.csv" if cache_name else None

    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path, dtype=str)

    api_key = require_nass_key(cfg)
    request_params = {**params, "key": api_key, "format": "JSON"}

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(NASS_API_URL, params=request_params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            rows = payload.get("data", [])
            out = pd.DataFrame(rows)
            if cache_path:
                out.to_csv(cache_path, index=False)
            time.sleep(sleep_seconds)
            return out
        except Exception as exc:  # noqa: BLE001 - API failures need retry context
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"NASS request failed after {max_retries} attempts") from last_error


def state_fips_codes() -> list[str]:
    """Return the study-region state FIPS codes in stable order."""
    return sorted(STATES)
