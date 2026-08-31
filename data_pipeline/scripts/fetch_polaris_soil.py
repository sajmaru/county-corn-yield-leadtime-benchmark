"""Fetch static POLARIS multi-depth soil features from Earth Engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.config import STATES, ensure_dirs, get_config
from data_pipeline.gee_utils import county_batches, dry_run_summary, initialize_gee, load_counties

POLARIS_PROPS = {
    "clay": "clay_mean",
    "sand": "sand_mean",
    "silt": "silt_mean",
    "om": "om_mean",
    "ph": "ph_mean",
    "bd": "bd_mean",
    "theta_s": "theta_s_mean",
}
POLARIS_DEPTHS = ["0_5", "5_15", "15_30", "30_60", "60_100", "100_200"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check setup without calling Earth Engine.")
    parser.add_argument("--batch-size", type=int, default=40, help="Counties per reduceRegions batch.")
    return parser.parse_args()


def build_soil_image(ee):
    """Build one multi-band POLARIS image."""
    bands = []
    for prop_name, collection_name in POLARIS_PROPS.items():
        collection = ee.ImageCollection(f"projects/sat-io/open-datasets/polaris/{collection_name}")
        image_list = collection.toList(collection.size())
        n_images = min(collection.size().getInfo(), len(POLARIS_DEPTHS))
        for idx in range(n_images):
            depth = POLARIS_DEPTHS[idx]
            bands.append(ee.Image(image_list.get(idx)).rename(f"soil_{prop_name}_{depth}"))
    if not bands:
        raise RuntimeError("No POLARIS bands were available.")
    return ee.Image.cat(bands)


def fetch(batch_size: int = 40) -> Path:
    import pandas as pd

    cfg = get_config()
    ensure_dirs(cfg)
    ee = initialize_gee(cfg)
    counties = load_counties()
    image = build_soil_image(ee)
    rows = []

    for state_fips, state_name in STATES.items():
        print(f"{state_name}...")
        for batch in county_batches(counties, ee, state_fips, batch_size=batch_size):
            reduced = image.reduceRegions(batch, ee.Reducer.mean(), scale=1000)
            for feature in reduced.getInfo()["features"]:
                rows.append(feature["properties"])

    output = cfg.data_dir / "polaris_soil_multidepth.csv"
    frame = pd.DataFrame(rows)
    frame["FIPS"] = frame["FIPS"].astype(str).str.zfill(5)
    frame.to_csv(output, index=False)
    return output


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(dry_run_summary())
        print(f"POLARIS properties: {list(POLARIS_PROPS)}")
        return
    output = fetch(batch_size=args.batch_size)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
