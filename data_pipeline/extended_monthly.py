"""Monthly Earth Engine features and wide-matrix construction.

This module extracts monthly county-level weather, drought, water-balance, and
vegetation features. It is checkpointed by state-year so long Earth Engine runs
can be resumed without restarting the world. Because apparently APIs enjoy
being dramatic.
"""

from __future__ import annotations

from pathlib import Path

from .config import STATES, PipelineConfig, ensure_dirs, get_config
from .gee_utils import county_batches, initialize_gee, load_counties

ALL_MONTHS = range(1, 13)
GROWING_MONTHS = range(4, 11)
GDD_BASE = 10.0
GDD_CAP = 30.0
KDD_THRESHOLD = 29.0
EDD_THRESHOLD = 34.0
MONTHLY_FEATURES = [
    "vpd",
    "tmin",
    "tmax",
    "precip",
    "srad",
    "eto",
    "gdd",
    "kdd",
    "edd",
    "pdsi",
    "spei90",
    "spi90",
    "ndvi",
    "evi",
    "lai",
    "fpar",
    "lst_day",
    "lst_night",
]
SOIL_FEATURES = [
    "soil_organic_carbon",
    "soil_clay_pct",
    "soil_sand_pct",
    "soil_awc",
    "soil_bulk_density",
]


def compute_daily_heat_bands(ee, daily_image):
    """Add daily corn heat-stress bands to one GRIDMET image."""
    tmax_c = daily_image.select("tmmx").subtract(273.15)
    tmin_c = daily_image.select("tmmn").subtract(273.15)
    gdd = (
        tmax_c.min(GDD_CAP)
        .add(tmin_c.max(GDD_BASE))
        .divide(2)
        .subtract(GDD_BASE)
        .max(0)
        .rename("gdd")
    )
    kdd = tmax_c.subtract(KDD_THRESHOLD).max(0).rename("kdd")
    edd = tmax_c.subtract(EDD_THRESHOLD).max(0).rename("edd")
    return daily_image.addBands([gdd, kdd, edd])


def build_monthly_image(ee, year: int, corn_mask):
    """Build a multi-band monthly feature image for one year."""
    import calendar

    bands = []
    for month in ALL_MONTHS:
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
        tag = f"{month:02d}"

        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(start, end)
        vpd = gridmet.select("vpd").mean().rename(f"vpd_{tag}")
        tmin = gridmet.select("tmmn").map(lambda img: img.subtract(273.15)).mean().rename(f"tmin_{tag}")
        tmax = gridmet.select("tmmx").map(lambda img: img.subtract(273.15)).mean().rename(f"tmax_{tag}")
        precip = gridmet.select("pr").sum().rename(f"precip_{tag}")
        srad = gridmet.select("srad").mean().rename(f"srad_{tag}")
        eto = gridmet.select("eto").sum().rename(f"eto_{tag}")
        heat = gridmet.map(lambda img: compute_daily_heat_bands(ee, img))
        gdd = heat.select("gdd").sum().rename(f"gdd_{tag}")
        kdd = heat.select("kdd").sum().rename(f"kdd_{tag}")
        edd = heat.select("edd").sum().rename(f"edd_{tag}")
        bands.extend([vpd, tmin, tmax, precip, srad, eto, gdd, kdd, edd])

        drought = ee.ImageCollection("GRIDMET/DROUGHT").filterDate(start, end)
        bands.extend(
            [
                drought.select("pdsi").mean().rename(f"pdsi_{tag}"),
                drought.select("spei90d").mean().rename(f"spei90_{tag}"),
                drought.select("spi90d").mean().rename(f"spi90_{tag}"),
            ]
        )

        if month in GROWING_MONTHS:
            bands.extend(_vegetation_bands(ee, start, end, tag, corn_mask))

    return ee.Image.cat(bands)


def fetch_monthly_features(
    config: PipelineConfig | None = None,
    batch_size: int = 40,
) -> Path:
    """Fetch monthly GEE features and return `gee_ext_monthly_all.csv`."""
    import pandas as pd

    try:
        from tqdm import tqdm
    except ImportError:  # keep the pipeline runnable without tqdm
        tqdm = None

    cfg = config or get_config()
    ensure_dirs(cfg)
    print(f"[monthly extract] resolved years {cfg.start_year}-{cfg.end_year} "
          f"| data_dir={cfg.data_dir}")
    ee = initialize_gee(cfg)
    counties = load_counties()
    chunk_dir = cfg.data_dir / "ext_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    years = list(range(cfg.start_year, cfg.end_year + 1))
    total = len(STATES) * len(years)
    bar = tqdm(total=total, desc="Monthly GEE extract", unit="state-yr") if tqdm else None

    for state_fips, state_name in STATES.items():
        batches = county_batches(counties, ee, state_fips, batch_size=batch_size)
        for year in years:
            chunk_path = chunk_dir / f"ext_{state_fips}_{year}.csv"
            if chunk_path.exists():
                frames.append(pd.read_csv(chunk_path, dtype={"FIPS": str}))
                status = "cached"
            else:
                frame = _fetch_state_year(ee, batches, year)
                frame.to_csv(chunk_path, index=False)
                frames.append(frame)
                status = f"{len(frame)} rows"
            if bar is not None:
                bar.set_postfix_str(f"{state_name} {year} ({status})")
                bar.update(1)
            else:
                print(f"  {state_fips} {year}: {status}")

    if bar is not None:
        bar.close()

    combined = pd.concat(frames, ignore_index=True)
    combined["FIPS"] = combined["FIPS"].astype(str).str.zfill(5)
    combined = combined.sort_values(["FIPS", "year", "month"]).reset_index(drop=True)
    output = cfg.data_dir / "gee_ext_monthly_all.csv"
    combined.to_csv(output, index=False)
    return output


def fetch_openlandmap_soil(
    config: PipelineConfig | None = None,
    batch_size: int = 80,
) -> Path:
    """Fetch static 0-30 cm OpenLandMap soil features by county."""
    import pandas as pd

    cfg = config or get_config()
    ensure_dirs(cfg)
    output = cfg.data_dir / "soil_properties.csv"
    if output.exists():
        return output

    ee = initialize_gee(cfg)
    counties = load_counties()
    soil_image = _openlandmap_soil_image(ee)
    rows = []
    for state_fips, state_name in STATES.items():
        print(f"{state_name} soil...")
        for batch in county_batches(counties, ee, state_fips, batch_size=batch_size):
            stats = soil_image.reduceRegions(batch, ee.Reducer.mean(), scale=1000)
            for feature in stats.getInfo()["features"]:
                rows.append(feature["properties"])

    frame = pd.DataFrame(rows)
    frame["FIPS"] = frame["FIPS"].astype(str).str.zfill(5)
    frame = frame[["FIPS", *[col for col in SOIL_FEATURES if col in frame.columns]]]
    frame.to_csv(output, index=False)
    return output


def build_wide_extended_features(
    config: PipelineConfig | None = None,
    monthly_path: Path | None = None,
    soil_path: Path | None = None,
) -> Path:
    """Pivot monthly features wide and merge optional static soil features."""
    import pandas as pd

    cfg = config or get_config()
    monthly = monthly_path or cfg.data_dir / "gee_ext_monthly_all.csv"
    if not monthly.exists():
        raise FileNotFoundError(f"Missing {monthly}. Run monthly fetch first.")

    data = pd.read_csv(monthly, dtype={"FIPS": str})
    data["FIPS"] = data["FIPS"].str.zfill(5)
    data["year"] = data["year"].astype(int)
    data["month"] = data["month"].astype(int)

    wide_parts = []
    for feature in MONTHLY_FEATURES:
        if feature not in data.columns:
            continue
        months = GROWING_MONTHS if feature in {"ndvi", "evi", "lai", "fpar", "lst_day", "lst_night"} else ALL_MONTHS
        subset = data[data["month"].isin(months)]
        pivot = subset.pivot_table(
            index=["FIPS", "year"], columns="month", values=feature, aggfunc="mean"
        ).rename(columns=lambda month: f"{feature}_{month:02d}")
        wide_parts.append(pivot)

    if not wide_parts:
        raise ValueError("No known monthly feature columns found.")
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.join(part, how="left")
    wide = wide.reset_index()
    wide = _add_seasonal_summaries(wide)

    soil = soil_path or cfg.data_dir / "soil_properties.csv"
    if soil.exists():
        soil_df = pd.read_csv(soil, dtype={"FIPS": str})
        soil_df["FIPS"] = soil_df["FIPS"].str.zfill(5)
        wide = wide.merge(soil_df, on="FIPS", how="left")

    output = cfg.data_dir / "wide_extended_features.csv"
    wide.sort_values(["FIPS", "year"]).to_csv(output, index=False)
    return output


def _fetch_state_year(ee, batches, year: int):
    import pandas as pd

    corn_mask = _corn_mask_for_year(ee, year)
    image = build_monthly_image(ee, year, corn_mask)
    rows = []
    for batch in batches:
        stats = image.reduceRegions(batch, ee.Reducer.mean(), scale=4000)
        for feature in stats.getInfo()["features"]:
            props = feature["properties"]
            fips = str(props.get("FIPS")).zfill(5)
            for month in ALL_MONTHS:
                row = {"FIPS": fips, "year": year, "month": month}
                tag = f"{month:02d}"
                for name in MONTHLY_FEATURES:
                    row[name] = props.get(f"{name}_{tag}")
                rows.append(row)
    return pd.DataFrame(rows)


def _corn_mask_for_year(ee, year: int):
    """Build a corn crop mask from LEAK-SAFE prior-year CDL (D3).

    For a forecast year Y the previous year's CDL (Y-1) is public before the
    season starts, so it never leaks the forecast year's own land use while
    still tracking the evolving corn footprint. GEE `USDA/NASS/CDL` only has
    reliable national coverage from ~2008, so years <= 2008 use the 2008 layer.
    An empty-image guard also falls back to 2008 if a given year is missing.
    """
    mask_year = 2008 if year <= 2008 else year - 1
    cdl = ee.ImageCollection("USDA/NASS/CDL").filter(
        ee.Filter.calendarRange(mask_year, mask_year, "year")
    )
    fallback = ee.ImageCollection("USDA/NASS/CDL").filter(
        ee.Filter.calendarRange(2008, 2008, "year")
    )
    cdl_image = ee.Image(
        ee.Algorithms.If(cdl.size().gt(0), cdl.first(), fallback.first())
    )
    return cdl_image.select("cropland").eq(1)


def _modis_within_month(ee, asset: str, start: str, end: str):
    """MODIS composites whose acquisition window ENDS on/before month-end (D5).

    `filterDate(start, end)` keeps composites intersecting the month; we then
    drop any whose `system:time_end` spills past the month boundary so a
    late-July composite extending into early August is not counted as July.
    `system:time_end` is a standard MODIS property; if a composite lacked it,
    the lte filter would exclude it (safe, conservative default).
    """
    month_end = ee.Date(end).advance(1, "day")
    return (
        ee.ImageCollection(asset)
        .filterDate(start, end)
        .filter(ee.Filter.lte("system:time_end", month_end.millis()))
    )


def _vegetation_bands(ee, start: str, end: str, tag: str, corn_mask):
    mod13 = _modis_within_month(ee, "MODIS/061/MOD13Q1", start, end)
    ndvi = mod13.select("NDVI").mean().multiply(0.0001).updateMask(corn_mask).rename(f"ndvi_{tag}")
    evi = mod13.select("EVI").mean().multiply(0.0001).updateMask(corn_mask).rename(f"evi_{tag}")
    lai_collection = _modis_within_month(ee, "MODIS/061/MOD15A2H", start, end)
    lai = lai_collection.select("Lai_500m").mean().multiply(0.1).updateMask(corn_mask).rename(f"lai_{tag}")
    fpar = lai_collection.select("Fpar_500m").mean().multiply(0.01).updateMask(corn_mask).rename(f"fpar_{tag}")
    lst_collection = _modis_within_month(ee, "MODIS/061/MOD11A2", start, end)
    lst_day = lst_collection.select("LST_Day_1km").mean().multiply(0.02).subtract(273.15).rename(f"lst_day_{tag}")
    lst_night = lst_collection.select("LST_Night_1km").mean().multiply(0.02).subtract(273.15).rename(f"lst_night_{tag}")
    return [ndvi, evi, lai, fpar, lst_day, lst_night]


def _openlandmap_soil_image(ee):
    soc = _mean_depths(ee, "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", "soil_organic_carbon")
    clay = _mean_depths(ee, "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02", "soil_clay_pct")
    sand = _mean_depths(ee, "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02", "soil_sand_pct")
    awc = _mean_depths(ee, "OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01", "soil_awc")
    bd = _mean_depths(ee, "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02", "soil_bulk_density").multiply(10)
    return ee.Image.cat([soc, clay, sand, awc, bd])


def _mean_depths(ee, asset: str, name: str):
    image = ee.Image(asset)
    return image.select("b0").add(image.select("b10")).add(image.select("b30")).divide(3).rename(name)


def _add_seasonal_summaries(wide):
    for name, cols, op in [
        ("vpd_jja_mean", ["vpd_06", "vpd_07", "vpd_08"], "mean"),
        ("kdd_jja_total", ["kdd_06", "kdd_07", "kdd_08"], "sum"),
        ("edd_jja_total", ["edd_06", "edd_07", "edd_08"], "sum"),
        ("gdd_gs_total", [f"gdd_{month:02d}" for month in range(4, 10)], "sum"),
        ("pdsi_jja_mean", ["pdsi_06", "pdsi_07", "pdsi_08"], "mean"),
        ("evi_jja_mean", ["evi_06", "evi_07", "evi_08"], "mean"),
        ("lai_jja_mean", ["lai_06", "lai_07", "lai_08"], "mean"),
        ("lst_day_jja_mean", ["lst_day_06", "lst_day_07", "lst_day_08"], "mean"),
        ("srad_jja_mean", ["srad_06", "srad_07", "srad_08"], "mean"),
    ]:
        present = [col for col in cols if col in wide.columns]
        if len(present) == len(cols):
            wide[name] = wide[present].mean(axis=1) if op == "mean" else wide[present].sum(axis=1)
    return wide
