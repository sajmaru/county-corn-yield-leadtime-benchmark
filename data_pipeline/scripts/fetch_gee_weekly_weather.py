"""Fetch weekly GRIDMET county weather features from Earth Engine."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from data_pipeline.config import STATES, ensure_dirs, get_config
from data_pipeline.gee_utils import county_batches, dry_run_summary, initialize_gee, load_counties

WEEKLY_VARS = {
    "tmax": ("tmmx", "mean", -273.15),
    "tmin": ("tmmn", "mean", -273.15),
    "precip": ("pr", "sum", 0.0),
    "srad": ("srad", "mean", 0.0),
    "vpd": ("vpd", "mean", 0.0),
    "eto": ("eto", "mean", 0.0),
}
QUARTERS = [(1, 13, "Q1"), (14, 26, "Q2"), (27, 39, "Q3"), (40, 52, "Q4")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check setup without calling Earth Engine.")
    parser.add_argument("--batch-size", type=int, default=35, help="Counties per reduceRegions batch.")
    return parser.parse_args()


def week_ranges(year: int) -> list[tuple[int, str, str]]:
    """Return 52 simple 7-day windows for a calendar year."""
    start = datetime(year, 1, 1)
    weeks = []
    for week in range(1, 53):
        week_start = start + timedelta(days=(week - 1) * 7)
        week_end = min(week_start + timedelta(days=6), datetime(year, 12, 31))
        weeks.append((week, week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")))
    return weeks


def quarter_image(ee, year: int, q_start: int, q_end: int):
    """Build one quarter of weekly weather bands."""
    bands = []
    for week, start, end in week_ranges(year):
        if week < q_start or week > q_end:
            continue
        collection = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(start, end)
        for name, (band, reducer, offset) in WEEKLY_VARS.items():
            image = collection.select(band).mean() if reducer == "mean" else collection.select(band).sum()
            if offset:
                image = image.add(offset)
            bands.append(image.rename(f"{name}_w{week:02d}"))
    return ee.Image.cat(bands)


def fetch(batch_size: int = 35) -> Path:
    import pandas as pd
    try:
        from tqdm import tqdm
    except ImportError:  # keep the pipeline runnable without tqdm
        tqdm = None

    cfg = get_config()
    ensure_dirs(cfg)
    print(f"[weekly weather] resolved years {cfg.start_year}-{cfg.end_year} "
          f"| data_dir={cfg.data_dir}")
    ee = initialize_gee(cfg)
    counties = load_counties()
    output = cfg.data_dir / "gee_weekly_weather.csv"
    rows = []

    years = list(range(cfg.start_year, cfg.end_year + 1))
    total = len(STATES) * len(years)
    bar = tqdm(total=total, desc="Weekly GEE extract", unit="state-yr") if tqdm else None

    for state_fips, state_name in STATES.items():
        batches = county_batches(counties, ee, state_fips, batch_size=batch_size)
        for year in years:
            chunk_path = cfg.data_dir / f"weekly_{state_fips}_{year}.csv"
            if chunk_path.exists():
                rows.append(pd.read_csv(chunk_path, dtype={"FIPS": str}))
                status = "cached"
            else:
                quarter_frames = []
                for q_start, q_end, label in QUARTERS:
                    image = quarter_image(ee, year, q_start, q_end)
                    q_rows = []
                    for batch in batches:
                        reduced = image.reduceRegions(batch, ee.Reducer.mean(), scale=4000)
                        for feature in reduced.getInfo()["features"]:
                            q_rows.append(feature["properties"])
                    quarter_frames.append(pd.DataFrame(q_rows))
                merged = quarter_frames[0]
                for frame in quarter_frames[1:]:
                    merged = merged.merge(frame, on="FIPS", how="outer")
                merged["year"] = year
                merged.to_csv(chunk_path, index=False)
                rows.append(merged)
                status = f"{len(merged)} rows"
            if bar is not None:
                bar.set_postfix_str(f"{state_name} {year}: {status}")
                bar.update(1)
            else:
                print(f"  {state_fips} {year}: {status}")
    if bar is not None:
        bar.close()

    combined = pd.concat(rows, ignore_index=True)
    combined["FIPS"] = combined["FIPS"].astype(str).str.zfill(5)
    combined.to_csv(output, index=False)
    return output


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(dry_run_summary())
        print(f"Weekly variables: {list(WEEKLY_VARS)}")
        return
    output = fetch(batch_size=args.batch_size)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
