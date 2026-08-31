"""Credential checks for NASS Quick Stats and Google Earth Engine."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from .config import PipelineConfig, get_config

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"


class CredentialError(RuntimeError):
    """Raised when required credentials are missing or invalid."""


def require_nass_key(config: PipelineConfig | None = None) -> str:
    """Return the NASS API key or raise a clear setup error."""
    cfg = config or get_config()
    if not cfg.nass_api_key or cfg.nass_api_key.startswith("replace-"):
        raise CredentialError(
            "Missing NASS_API_KEY. Request a free key at "
            "https://quickstats.nass.usda.gov/api, then add it to .env."
        )
    return cfg.nass_api_key


def get_gee_project(config: PipelineConfig | None = None) -> str | None:
    """Return the optional Earth Engine Cloud project ID."""
    cfg = config or get_config()
    if not cfg.gee_project or cfg.gee_project.startswith("replace-"):
        return None
    return cfg.gee_project


def validate_nass_key(config: PipelineConfig | None = None, timeout: int = 30) -> bool:
    """Make a tiny Quick Stats request to confirm the NASS key works."""
    cfg = config or get_config()
    key = require_nass_key(cfg)
    params = {
        "key": key,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "agg_level_desc": "NATIONAL",
        "year": cfg.end_year,
        "format": "JSON",
    }
    url = f"{NASS_API_URL}?{urlencode(params)}"
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise CredentialError(f"NASS API returned an error: {payload['error']}")
    return True


def initialize_earth_engine(config: PipelineConfig | None = None):
    """Initialize Earth Engine if the `earthengine-api` package is installed.

    Users must authenticate once with:
        earthengine authenticate

    If `GEE_PROJECT` is set, it is passed to `ee.Initialize(project=...)`.
    """
    cfg = config or get_config()
    try:
        import ee
    except ImportError as exc:
        raise CredentialError(
            "earthengine-api is not installed. Install it with `pip install "
            "earthengine-api` or the project requirements file."
        ) from exc

    project = get_gee_project(cfg)
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee
