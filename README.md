# county-corn-yield-leadtime-benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22190344.svg)](https://doi.org/10.5281/zenodo.22190344)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=sajmaru.county-corn-yield-leadtime-benchmark)](https://github.com/sajmaru/county-corn-yield-leadtime-benchmark)

**County-Level US Corn Yield Lead-Time Forecasting Benchmark**

**Authors:** Saj Maru, Parth Maniar, Radhika Kotecha

Reproducible code for a county-level, in-season corn yield forecasting benchmark
across the US Corn Belt (2000–2025). The benchmark measures how forecast skill
grows from April through harvest using **only public data** — weather,
satellite, soil, USDA crop reports, irrigation, geography, and historical yield —
with leak-safe trend features and shared leave-one-year-out, walk-forward, and
leave-one-state-out cross-validation. Pairs with the archived modeling matrix on
Zenodo.

> **The research question:** *How early can a county-level corn yield forecast
> be trusted?*

This repository is **code only**. The assembled modeling matrix is large and is
distributed separately as an archived data release (see
[Getting the data](#getting-the-data)). You can either **download** that matrix
and reproduce the paper numbers in minutes, or **rebuild it from scratch** from
the public sources using the pipeline in [`data_pipeline/`](data_pipeline).

---

## Contents

| Path | What it is |
|---|---|
| [`reproducible_benchmark/`](reproducible_benchmark) | Clean, script-based code that reproduces the paper's tables and figures from the assembled matrix. **Start here.** |
| [`data_pipeline/`](data_pipeline) | Source-data pipeline that rebuilds the matrix from public APIs and Google Earth Engine. Optional — for full provenance. |
| `USA_data/benchmark/` | Local data folder. **Not committed.** You create it and drop the downloaded matrix here (see below). |
| `.env.example` | Template for the credentials needed only by the data pipeline. |
| `requirements.txt` | Core + pipeline dependencies. |

Each subfolder has its own README with the full detail:
[`reproducible_benchmark/README.md`](reproducible_benchmark/README.md) and
[`data_pipeline/README.md`](data_pipeline/README.md).

---

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That installs everything needed to reproduce the paper tables and to run the
data pipeline.

---

## Getting the data

The assembled county-year modeling matrix is published as an archived data
release on Zenodo under a CC BY 4.0 license:

> **Dataset DOI:** https://doi.org/10.5281/zenodo.22133245

### 1. Download

From the Zenodo record, download the modeling matrix. Two equivalent formats are
provided — pick one:

| File | Notes |
|---|---|
| `benchmark_matrix.parquet` | Preferred: preserves dtypes, faster to load. |
| `benchmark_matrix.csv` | Plain CSV for tools without Parquet support. |

The record also contains a data dictionary
(`benchmark_matrix_dictionary.csv`), a build report
(`benchmark_matrix_build_report.csv`), source provenance
(`source_provenance.csv`), and a `MANIFEST.txt` with SHA-256 checksums.

### 2. Place it where the code expects it

The reproduction scripts read a **plain CSV** at exactly this path
(relative to the repo root):

```text
USA_data/benchmark/benchmark_matrix.csv
```

Create the folder and put the file there. If you downloaded the plain CSV,
just move it into place:

```bash
mkdir -p USA_data/benchmark
mv benchmark_matrix.csv USA_data/benchmark/benchmark_matrix.csv
```

If you downloaded the Parquet instead, convert it once:

```bash
mkdir -p USA_data/benchmark
python -c "import pandas as pd; pd.read_parquet('benchmark_matrix.parquet').to_csv('USA_data/benchmark/benchmark_matrix.csv', index=False)"
```

### 3. (Optional) Verify integrity

Checksums for every archived file are in `MANIFEST.txt` on the Zenodo record:

```bash
sha256sum benchmark_matrix.csv    # compare against MANIFEST.txt
```

> **Note on scope.** The released matrix (22,797 rows × 594 columns, 2000–2025)
> matches the paper's analysis sample exactly: one row per county-year with a
> reported corn yield (22,797 county-years across 1,009 reporting units). The
> loader in `reproducible_benchmark/data.py` reads it directly; no extra row
> filtering is required.

---

## Quick start: reproduce the paper numbers

With the matrix in place, from the repo root:

```bash
python -m reproducible_benchmark.scripts.validate_inputs        # sanity-check the matrix + print dataset facts
python -m reproducible_benchmark.scripts.run_paper_tables       # model comparison + feature ablation
python -m reproducible_benchmark.scripts.run_leadtime           # compact lead-time tables
python -m reproducible_benchmark.scripts.run_leadtime_paper     # full lead-time tables + figures
python -m reproducible_benchmark.scripts.run_significance_tests # paired Wilcoxon/t-test model comparisons
python -m reproducible_benchmark.scripts.write_results_latex    # paste-ready LaTeX table snippets
python -m reproducible_benchmark.scripts.audit_no_leakage       # verify no forbidden/future feature was ever selected
```

All outputs are written under:

```text
reproducible_benchmark/outputs/
├── tables/
├── figures/
└── logs/
```

See [`reproducible_benchmark/README.md`](reproducible_benchmark/README.md) for
the methodology and the leak-safe trend feature.

---

## Rebuilding the matrix from public sources (optional)

If you want full provenance rather than the archived matrix, the
[`data_pipeline/`](data_pipeline) folder rebuilds every feature from public
sources (USDA NASS, GRIDMET, MODIS, POLARIS, OpenLandMap, LGRIP30,
US Census, SRTM).

This path needs API credentials and Google Earth Engine access. Copy the
template and fill it in:

```bash
cp .env.example .env      # then edit .env
```

```bash
NASS_API_KEY=your-nass-api-key          # free key: https://quickstats.nass.usda.gov/api
GEE_PROJECT=your-earth-engine-project   # optional; enables GEE extraction
```

Validate and inspect what's available:

```bash
python -m data_pipeline.scripts.validate_credentials
python -m data_pipeline.scripts.build_benchmark_matrix --check-only
```

Full instructions, pipeline stages, and the list of produced checkpoint files
are in [`data_pipeline/README.md`](data_pipeline/README.md).

### Census boundary files (needed only for the pipeline rebuild)

The Earth Engine extraction steps aggregate rasters over county polygons, so they
need the public US Census 2022 cartographic boundary shapefiles in a local
`geo/` folder:

```bash
mkdir -p geo
curl -L -o geo/cb_county.zip \
  https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip
curl -L -o geo/cb_state.zip \
  https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip
unzip -o geo/cb_county.zip -d geo
unzip -o geo/cb_state.zip -d geo
```

You do **not** need `geo/` to reproduce the paper tables from the downloaded
matrix — only for the from-scratch pipeline rebuild.

---

## Methodology at a glance

- **Target:** raw USDA NASS county corn grain yield, `yield_bu_acre` (bu/acre).
- **Leak-safe trend:** the code builds `yield_trend_past`, a county trend fit
  **only on prior years** for each row — never using future years as a forecast
  input. The descriptive full-record `yield_trend` / `yield_anomaly` columns are
  kept for diagnostics only and are **not** used as predictors.
- **Evaluation:** leave-one-year-out, walk-forward, and leave-one-state-out
  cross-validation, shared across all experiments.

---

## Citation

If you use this repository, cite the software release. If you also use the
assembled modeling matrix, cite the dataset separately.

**Software (version 1.0.0):** https://doi.org/10.5281/zenodo.22190345

```bibtex
@software{maru_corn_yield_leadtime_2026,
  author    = {Maru, Saj and Kotecha, Radhika and Maniar, Parth},
  title     = {{County-Level US Corn Yield Lead-Time Forecasting: Framework, Experiments, and Reproducible Benchmark}},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22190345},
  url       = {https://doi.org/10.5281/zenodo.22190345}
}
```

**Dataset:** https://doi.org/10.5281/zenodo.22133245

```bibtex
@misc{cornbenchmark_data_2026,
  author    = {Maru, Saj and Kotecha, Radhika and Maniar, Parth},
  title     = {{County-Level US Corn Yield Lead-Time Forecasting Benchmark (Corn Belt, 2000--2025)}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22133245},
  url       = {https://doi.org/10.5281/zenodo.22133245},
  note      = {Data set}
}
```

---

## License

The code in this repository is released under the **MIT License** — see
[`LICENSE`](LICENSE). The published dataset is licensed **CC BY 4.0**. The
underlying source products (USDA NASS, GRIDMET, MODIS, POLARIS,
OpenLandMap, LGRIP30, US Census, SRTM) are public; consult each provider for
their individual terms.
