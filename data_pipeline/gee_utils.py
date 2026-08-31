"""Google Earth Engine helper functions for county-level extraction."""

from __future__ import annotations

from pathlib import Path

from .config import ROOT, STATES, PipelineConfig, get_config
from .credentials import initialize_earth_engine

COUNTY_SHP = ROOT / "geo" / "cb_2022_us_county_500k.shp"


def require_geo_files() -> None:
    """Raise a clear error if Census boundary files have not been downloaded."""
    if not COUNTY_SHP.exists():
        raise FileNotFoundError(
            "Missing Census county shapefile. Download it into geo/ using the "
            "commands in README.md before running GEE extraction scripts."
        )


def load_counties():
    """Load study-region county geometries as a GeoDataFrame."""
    require_geo_files()
    import geopandas as gpd

    counties = gpd.read_file(COUNTY_SHP)[["GEOID", "STATEFP", "NAME", "geometry"]]
    counties = counties[counties["STATEFP"].isin(STATES)].copy()
    counties = counties.rename(columns={"GEOID": "FIPS"})
    counties["FIPS"] = counties["FIPS"].astype(str).str.zfill(5)
    return counties.reset_index(drop=True)


def initialize_gee(config: PipelineConfig | None = None):
    """Initialize Earth Engine and return the imported `ee` module."""
    return initialize_earth_engine(config or get_config())


def county_batches(counties, ee, state_fips: str, batch_size: int = 40):
    """Convert one state's counties into small Earth Engine FeatureCollections."""
    from shapely.geometry import mapping

    state_rows = counties[counties["STATEFP"] == state_fips]
    batches = []
    for start in range(0, len(state_rows), batch_size):
        chunk = state_rows.iloc[start : start + batch_size]
        features = []
        for _, row in chunk.iterrows():
            geom = ee.Geometry(mapping(row.geometry))
            features.append(ee.Feature(geom, {"FIPS": row["FIPS"]}))
        batches.append(ee.FeatureCollection(features))
    return batches


def dry_run_summary() -> dict[str, object]:
    """Return a lightweight summary without requiring Earth Engine initialization."""
    geo_present = COUNTY_SHP.exists()
    return {
        "geo_present": geo_present,
        "county_shapefile": str(COUNTY_SHP),
        "states": len(STATES),
    }
