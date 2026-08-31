# Data Pipeline: Rebuilding the Corn Yield Benchmark Matrix

This folder contains the source-data pipeline for rebuilding the modeling table
used by the paper:

```text
USA_data/benchmark/benchmark_matrix.csv
```

The clean modeling benchmark in `reproducible_benchmark/` reproduces the paper
numbers from that matrix. This `data_pipeline/` folder documents and scripts how
the matrix is rebuilt from public data sources.

## Why this is separate from `reproducible_benchmark/`

The source-data pipeline depends on external services:

- USDA NASS Quick Stats API
- Google Earth Engine authentication
- Google Cloud / Earth Engine project access
- large public geospatial products
- long-running zonal summaries over thousands of county-year combinations

That means rebuilding the matrix from scratch is slower and more fragile than
rerunning the modeling benchmark. A reviewer should be able to reproduce the
paper tables from the archived `benchmark_matrix.csv` without rerunning every
API extraction. Users who want full provenance can run this pipeline.

## Credentials

Credentials are read from environment variables or a local `.env` file in the
repo root. The real `.env` file is gitignored and must never be committed.

A template is provided at:

```text
.env.example
```

Copy it once:

```bash
cp .env.example .env
```

Then edit `.env`.

### USDA NASS Quick Stats API key

1. Go to <https://quickstats.nass.usda.gov/api>
2. Request a free API key.
3. Add it to `.env`:

```bash
NASS_API_KEY=your-key-here
```

The pipeline uses NASS for:

- county corn grain yield
- state-level crop condition and progress reports
- historical yield context

### Google Earth Engine project ID

1. Create or select a Google Cloud project.
2. Enable Earth Engine access for that project.
3. Install the Earth Engine CLI/API.
4. Authenticate once:

```bash
earthengine authenticate
```

5. Add the project ID to `.env`:

```bash
GEE_PROJECT=your-google-cloud-project-id
```

The pipeline uses Earth Engine for crop-masked county summaries of:

- GRIDMET weather
- GRIDMET drought indicators
- MODIS NDVI/EVI/LAI/FPAR/LST products
- Cropland Data Layer corn masks
- SRTM elevation
- OpenLandMap / soil products where applicable
- LGRIP30 irrigation product where available

## Validate credentials

Run:

```bash
python -m data_pipeline.scripts.validate_credentials
```

The command prints whether credentials are present and tests NASS. It attempts to
initialize Earth Engine if `earthengine-api` is installed and the user has run
`earthengine authenticate`.

It does not print secret values.

## Recommended environment

For local development, install the project requirements from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline dependencies (`requests`, `earthengine-api`, `geopandas`,
`shapely`, `pyproj`) are already included in `requirements.txt`.

Alternatively, use `uv` for one-off commands without manually creating an
environment:

```bash
uv run \
  --with numpy --with pandas --with scipy --with scikit-learn \
  --with requests --with earthengine-api \
  python -m data_pipeline.scripts.validate_credentials
```

## Pipeline stages

The release is staged so each step writes a checkpoint. This makes long external
API/GEE extraction restartable.

Implemented now:

```bash
python -m data_pipeline.scripts.fetch_nass_yield
python -m data_pipeline.scripts.fetch_nass_condition
python -m data_pipeline.scripts.build_historical_yield
python -m data_pipeline.scripts.fetch_gee_veg_indices --dry-run
python -m data_pipeline.scripts.fetch_polaris_soil --dry-run
python -m data_pipeline.scripts.fetch_gee_weekly_weather --dry-run
python -m data_pipeline.scripts.fetch_gee_monthly_features --dry-run
python -m data_pipeline.scripts.fetch_irrigation_geography --dry-run
python -m data_pipeline.scripts.build_benchmark_matrix --check-only
python -m data_pipeline.scripts.write_data_dictionary
```

These write:

```text
USA_data/benchmark/nass_yield_detrended.csv
USA_data/benchmark/nass_progress_condition_raw.csv
USA_data/benchmark/nass_state_year_features.csv
USA_data/benchmark/historical_yield_features.csv
USA_data/benchmark/gee_veg_indices.csv
USA_data/benchmark/polaris_soil_multidepth.csv
USA_data/benchmark/gee_weekly_weather.csv
USA_data/benchmark/gee_ext_monthly_all.csv
USA_data/benchmark/soil_properties.csv
USA_data/benchmark/wide_extended_features.csv
USA_data/benchmark/geographic_features.csv
USA_data/benchmark/lgrip30_irrigation.csv
USA_data/benchmark/usda_census_irrigation.csv
USA_data/benchmark/irrigation_features.csv
USA_data/benchmark/benchmark_matrix_rebuilt.csv
USA_data/benchmark/benchmark_matrix_build_report.csv
USA_data/benchmark/benchmark_matrix_dictionary.csv
```

The implemented GEE scripts are intentionally checkpointed by state/year where
needed. Long-running steps can be resumed from existing chunk CSVs. For full
runs, drop the `--dry-run` flag after credentials, dependencies, and `geo/` are
available.

## Data products and provenance

The benchmark combines public data from these sources:

| Feature group | Product/API | Typical output |
|---|---|---|
| Yield | USDA NASS Quick Stats | `yield_bu_acre` |
| Crop condition/progress | USDA NASS Quick Stats | `nass_*` |
| Weather | GRIDMET | `tmax_*`, `tmin_*`, `precip_*`, `vpd_*`, `srad_*`, `eto_*` |
| Heat stress | GRIDMET daily | `gdd_*`, `kdd_*`, `edd_*` |
| Drought | GRIDMET drought | `spei90_*`, `spi90_*`, `pdsi_*` |
| Vegetation | MODIS | `ndvi_*`, `evi_*`, `lai_*`, `fpar_*`, `lst_*` |
| Water/greenness indices | MODIS-derived | `ndwi_*`, `gci_*` |
| Soil | OpenLandMap, POLARIS | `soil_*` |
| Irrigation | LGRIP30 | `lgrip_irrigated_frac` |
| Geography | Census, SRTM | `lat`, `lon`, `area_km2`, `elevation_m` |
| History | NASS yield, lagged only | `yield_lag_*` |

A machine-readable provenance table ships with this folder at
`data_pipeline/source_provenance.csv` (also included in the Zenodo data
release). It lists, per feature group:

- source name
- API or GEE asset ID
- native resolution
- aggregation method
- output feature names

## Important leakage rule

The main modeling target is raw NASS yield:

```text
yield_bu_acre
```

The clean reproduction code creates a past-only trend feature:

```text
yield_trend_past
```

This avoids using future years to define the trend feature in operational
forecasting. The older full-record `yield_trend` and `yield_anomaly` are useful
for descriptive analysis and drought interpretation, but should not be used as a
forecast input in the cleaned no-leakage benchmark.

## Security checklist before public release

Before publishing the repo:

- rotate any API key that was ever committed or stored in private working files
- remove hardcoded keys from scripts
- scrub secrets from git history if this repo becomes public
- keep `.env` gitignored
- keep exploratory notebooks outside the public release
- verify redistribution terms for derived data products

## Current status

Credential handling, NASS yield extraction, NASS progress/condition extraction,
lagged historical yield features, GEE extraction scripts, irrigation/geography
features, source provenance, benchmark-matrix rebuilding, and a data dictionary
generator are implemented.
