"""Fetch MODIS-derived NDWI and GCI county features from Earth Engine."""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

from data_pipeline.config import STATES, ensure_dirs, get_config
from data_pipeline.gee_utils import county_batches, dry_run_summary, initialize_gee, load_counties


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check setup without calling Earth Engine.")
    parser.add_argument("--batch-size", type=int, default=40, help="Counties per reduceRegions batch.")
    return parser.parse_args()


def build_veg_image(ee, year: int):
    """Build monthly NDWI and GCI bands for April-October.

    MODIS 8-day (MOD09A1) composites are kept only when their acquisition
    window ends on/before month-end (D5), so a late-July composite spilling
    into early August is not counted as July. NDWI/GCI stay UNMASKED (D4).
    """
    bands = []
    for month in range(4, 11):
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{last_day}"
        tag = f"{month:02d}"
        month_end = ee.Date(end).advance(1, "day")
        modis = (
            ee.ImageCollection("MODIS/061/MOD09A1")
            .filterDate(start, end)
            .filter(ee.Filter.lte("system:time_end", month_end.millis()))
            .select(["sur_refl_b01", "sur_refl_b02", "sur_refl_b04", "sur_refl_b06"])
        )
        composite = modis.median()
        red = composite.select("sur_refl_b01")
        nir = composite.select("sur_refl_b02")
        green = composite.select("sur_refl_b04")
        swir = composite.select("sur_refl_b06")
        ndwi = nir.subtract(swir).divide(nir.add(swir)).rename(f"ndwi_{tag}")
        gci = nir.divide(green.max(1)).subtract(1).rename(f"gci_{tag}")
        bands.extend([ndwi, gci])
    return ee.Image.cat(bands)


def fetch(batch_size: int = 40) -> Path:
    """Run the extraction and return the combined output path."""
    import pandas as pd
    try:
        from tqdm import tqdm
    except ImportError:  # keep the pipeline runnable without tqdm
        tqdm = None
    cfg = get_config()
    ensure_dirs(cfg)
    print(f"[veg NDWI/GCI] resolved years {cfg.start_year}-{cfg.end_year} "
          f"| data_dir={cfg.data_dir}")
    ee = initialize_gee(cfg)
    counties = load_counties()
    output = cfg.data_dir / "gee_veg_indices.csv"

    years = list(range(cfg.start_year, cfg.end_year + 1))
    total = len(STATES) * len(years)
    bar = tqdm(total=total, desc="Veg NDWI/GCI extract", unit="state-yr") if tqdm else None

    rows = []
    for state_fips, state_name in STATES.items():
        batches = county_batches(counties, ee, state_fips, batch_size=batch_size)
        for year in years:
            chunk_path = cfg.data_dir / f"veg_{state_fips}_{year}.csv"
            if chunk_path.exists():
                rows.append(pd.read_csv(chunk_path, dtype={"FIPS": str}))
                status = "cached"
            else:
                image = build_veg_image(ee, year)
                year_rows = []
                for batch in batches:
                    reduced = image.reduceRegions(batch, ee.Reducer.mean(), scale=500)
                    for feature in reduced.getInfo()["features"]:
                        props = feature["properties"]
                        props["year"] = year
                        year_rows.append(props)
                frame = pd.DataFrame(year_rows)
                frame.to_csv(chunk_path, index=False)
                rows.append(frame)
                status = f"{len(frame)} rows"
            if bar:
                bar.set_postfix_str(f"{state_name} {year}: {status}")
                bar.update(1)
            else:
                print(f"  {state_fips} {year}: {status}")
    if bar:
        bar.close()

    combined = pd.concat(rows, ignore_index=True)
    combined["FIPS"] = combined["FIPS"].astype(str).str.zfill(5)
    combined.to_csv(output, index=False)
    return output


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(dry_run_summary())
        return
    output = fetch(batch_size=args.batch_size)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
