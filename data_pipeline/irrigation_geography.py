"""Irrigation and static geography feature extraction.

Outputs are intentionally small, static, and mergeable by FIPS. The satellite
irrigation and elevation steps require Earth Engine; Census irrigation requires
NASS credentials. Local centroid/area features only require Census shapefiles.
"""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig, ensure_dirs, get_config
from .gee_utils import initialize_gee, load_counties
from .nass import STATE_FIPS_TO_ALPHA, quickstats_get, value_to_float

LGRIP_CANDIDATES = [
    "projects/sat-io/open-datasets/GFSAD/LGRIP30",
    "projects/sat-io/open-datasets/LGRIP30",
    "USGS/GFSAD1000_V1",
]
CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]


def build_local_geography(config: PipelineConfig | None = None) -> Path:
    """Write centroid latitude/longitude and county area from local geometry."""
    import pandas as pd

    cfg = config or get_config()
    ensure_dirs(cfg)
    counties = load_counties()
    projected = counties.to_crs("EPSG:5070")
    centroids = projected.geometry.centroid.to_crs("EPSG:4326")
    frame = pd.DataFrame(
        {
            "FIPS": counties["FIPS"].astype(str).str.zfill(5),
            "lat": centroids.y,
            "lon": centroids.x,
            "area_km2": projected.geometry.area / 1_000_000,
        }
    )
    output = cfg.data_dir / "geographic_features.csv"
    frame.to_csv(output, index=False)
    return output


def add_srtm_elevation(
    geography_path: Path | None = None,
    config: PipelineConfig | None = None,
    scale: int = 500,
) -> Path:
    """Add mean SRTM elevation to `geographic_features.csv`."""
    import pandas as pd
    from shapely.geometry import mapping

    cfg = config or get_config()
    ensure_dirs(cfg)
    path = geography_path or cfg.data_dir / "geographic_features.csv"
    if not path.exists():
        path = build_local_geography(cfg)

    ee = initialize_gee(cfg)
    counties = load_counties()
    srtm = ee.Image("USGS/SRTMGL1_003")
    rows = []
    for index, row in counties.iterrows():
        if index == 0 or (index + 1) % 50 == 0:
            print(f"Elevation {index + 1}/{len(counties)}")
        try:
            stats = srtm.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry(mapping(row.geometry)),
                scale=scale,
                maxPixels=1e10,
                bestEffort=True,
            ).getInfo()
            elevation = stats.get("elevation")
        except Exception as exc:  # noqa: BLE001 - continue checkpoint-friendly extraction
            print(f"  elevation failed for {row['FIPS']}: {exc}")
            elevation = None
        rows.append({"FIPS": row["FIPS"], "elevation_m": elevation})

    geo = pd.read_csv(path, dtype={"FIPS": str})
    elev = pd.DataFrame(rows)
    elev["FIPS"] = elev["FIPS"].astype(str).str.zfill(5)
    geo = geo.drop(columns=["elevation_m"], errors="ignore").merge(elev, on="FIPS", how="left")
    geo.to_csv(path, index=False)
    return path


def fetch_lgrip30_irrigation(
    config: PipelineConfig | None = None,
    scale_override: int | None = None,
) -> Path:
    """Fetch county-level satellite irrigated/cropland fractions from GEE."""
    import numpy as np
    import pandas as pd
    from shapely.geometry import mapping

    cfg = config or get_config()
    ensure_dirs(cfg)
    output = cfg.data_dir / "lgrip30_irrigation.csv"
    if output.exists():
        return output

    ee = initialize_gee(cfg)
    counties = load_counties()
    image, asset = _first_accessible_irrigation_image(ee)
    band = image.bandNames().getInfo()[0]
    if "GFSAD1000" in asset:
        irrigated = image.select(band).gte(1).And(image.select(band).lte(2)).rename("irrigated")
        cropland = image.select(band).gte(1).And(image.select(band).lte(5)).rename("cropland")
        scale = scale_override or 1000
    else:
        irrigated = image.select(band).gte(2).rename("irrigated")
        cropland = image.select(band).gte(1).rename("cropland")
        scale = scale_override or 240

    rows = []
    for index, row in counties.iterrows():
        if index == 0 or (index + 1) % 50 == 0:
            print(f"Irrigation {index + 1}/{len(counties)}")
        try:
            stats = irrigated.addBands(cropland).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee.Geometry(mapping(row.geometry)),
                scale=scale,
                maxPixels=1e10,
                bestEffort=True,
            ).getInfo()
            irrigated_frac = stats.get("irrigated")
            cropland_frac = stats.get("cropland")
        except Exception as exc:  # noqa: BLE001 - keep going by county
            print(f"  irrigation failed for {row['FIPS']}: {exc}")
            irrigated_frac = None
            cropland_frac = None
        rows.append(
            {
                "FIPS": row["FIPS"],
                "lgrip_irrigated_frac": irrigated_frac,
                "lgrip_cropland_frac": cropland_frac,
            }
        )

    frame = pd.DataFrame(rows)
    frame["FIPS"] = frame["FIPS"].astype(str).str.zfill(5)
    frame["lgrip_irrig_of_crop"] = frame["lgrip_irrigated_frac"] / frame[
        "lgrip_cropland_frac"
    ].replace(0, np.nan)
    frame.to_csv(output, index=False)
    return output


def fetch_usda_census_irrigation(config: PipelineConfig | None = None) -> Path:
    """Fetch Census irrigated/cropland acres and compute county fractions."""
    import numpy as np
    import pandas as pd

    cfg = config or get_config()
    ensure_dirs(cfg)
    output = cfg.data_dir / "usda_census_irrigation.csv"
    if output.exists():
        return output

    rows = []
    for year in CENSUS_YEARS:
        for state_fips, state_alpha in STATE_FIPS_TO_ALPHA.items():
            irrigated = _census_ag_land(
                year, state_alpha, "AG LAND, IRRIGATED - ACRES", f"census_irr_{state_fips}_{year}", cfg
            )
            cropland = _census_ag_land(
                year, state_alpha, "AG LAND, CROPLAND - ACRES", f"census_crop_{state_fips}_{year}", cfg
            )
            if irrigated.empty:
                continue
            irrigated = _county_value_frame(irrigated, "irrig_ac")
            cropland = _county_value_frame(cropland, "crop_ac") if not cropland.empty else None
            if cropland is not None:
                merged = irrigated.merge(cropland, on=["state_fips_code", "county_code"], how="left")
            else:
                merged = irrigated.copy()
                merged["crop_ac"] = np.nan
            merged["year"] = year
            rows.append(merged)

    if not rows:
        raise RuntimeError("NASS Census irrigation returned no rows.")

    frame = pd.concat(rows, ignore_index=True)
    frame["FIPS"] = frame["state_fips_code"].str.zfill(2) + frame["county_code"].str.zfill(3)
    frame["irrig_frac_census"] = frame["irrig_ac"] / frame["crop_ac"].replace(0, np.nan)
    frame.to_csv(output, index=False)
    return output


def build_irrigation_features(config: PipelineConfig | None = None) -> Path:
    """Combine satellite and Census irrigation into one static county file."""
    import pandas as pd

    cfg = config or get_config()
    lgrip_path = cfg.data_dir / "lgrip30_irrigation.csv"
    census_path = cfg.data_dir / "usda_census_irrigation.csv"
    if not lgrip_path.exists() and not census_path.exists():
        raise FileNotFoundError("Run LGRIP30 and/or USDA Census irrigation extraction first.")

    frames = []
    if lgrip_path.exists():
        frames.append(pd.read_csv(lgrip_path, dtype={"FIPS": str}).drop_duplicates("FIPS"))
    if census_path.exists():
        census = pd.read_csv(census_path, dtype={"FIPS": str})
        census_latest = census.sort_values("year").drop_duplicates("FIPS", keep="last")
        keep = ["FIPS", "year", "irrig_ac", "crop_ac", "irrig_frac_census"]
        frames.append(census_latest[[col for col in keep if col in census_latest.columns]])

    output = cfg.data_dir / "irrigation_features.csv"
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="FIPS", how="outer")
    combined["FIPS"] = combined["FIPS"].astype(str).str.zfill(5)
    combined.to_csv(output, index=False)
    return output


def _first_accessible_irrigation_image(ee):
    last_error = None
    for asset in LGRIP_CANDIDATES:
        try:
            image = ee.Image(asset).select("landcover") if "GFSAD1000" in asset else ee.ImageCollection(asset).mosaic()
            image.bandNames().getInfo()
            print(f"Using irrigation asset: {asset}")
            return image, asset
        except Exception as exc:  # noqa: BLE001 - try fallback assets
            last_error = exc
            print(f"  asset unavailable: {asset}: {str(exc)[:120]}")
    raise RuntimeError("No irrigation asset was accessible.") from last_error


def _census_ag_land(year: int, state_alpha: str, short_desc: str, cache_name: str, cfg):
    return quickstats_get(
        {
            "source_desc": "CENSUS",
            "commodity_desc": "AG LAND",
            "short_desc": short_desc,
            "agg_level_desc": "COUNTY",
            "state_alpha": state_alpha,
            "year": str(year),
        },
        cache_name=cache_name,
        config=cfg,
    )


def _county_value_frame(frame, value_name: str):
    out = frame[["state_fips_code", "county_code", "Value"]].copy()
    out[value_name] = out["Value"].apply(value_to_float)
    out = out.drop(columns=["Value"])
    out["state_fips_code"] = out["state_fips_code"].astype(str).str.zfill(2)
    out["county_code"] = out["county_code"].astype(str).str.zfill(3)
    return out
