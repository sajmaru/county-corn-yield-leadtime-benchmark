"""Build the benchmark matrix from available source-data checkpoints.

The full raw-data pipeline is intentionally checkpointed. Some checkpoints come
from slow external services such as Google Earth Engine. This builder merges the
checkpoints that are present and reports any missing components instead of
silently pretending the matrix is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PipelineConfig, get_config


@dataclass(frozen=True)
class Component:
    """A source-data checkpoint that can be merged into the matrix."""

    filename: str
    keys: tuple[str, ...]
    required: bool
    description: str


# Columns that flow in from checkpoint components as intermediates or legacy
# duplicates but are NOT part of the canonical benchmark_matrix.csv contract.
# Dropped after assembly so the pipeline matrix matches the released matrix's
# column set (see the data dictionary in data_pipeline/README.md).
EXCLUDE_COLUMNS = (
    # Census-irrigation intermediates (build_irrigation_features keeps the raw
    # acre counts + the census fraction; the released matrix keeps only the
    # LGRIP30 satellite fractions).
    "crop_ac",
    "irrig_ac",
    "irrig_frac_census",
    # Legacy duplicate one-year yield lag carried in wide_extended_features.csv;
    # superseded by historical_yield_features.csv (yield_lag_* / anomaly_lag*).
    "lag1_yield",
    # NASS Quick Stats identifier/descriptor columns that ride along on
    # nass_yield_detrended.csv: sfips/cfips duplicate state_fips/FIPS and
    # class_desc is a constant ("ALL CLASSES"). Not part of the canonical
    # 594-column contract (see the data dictionary in data_pipeline/README.md).
    "sfips",
    "cfips",
    "class_desc",
)


COMPONENTS = [
    Component(
        "wide_extended_features.csv",
        ("FIPS", "year"),
        False,
        "monthly weather, drought, water-balance, vegetation, and base soil features",
    ),
    Component(
        "gee_veg_indices.csv",
        ("FIPS", "year"),
        False,
        "MODIS-derived NDWI and GCI features",
    ),
    Component(
        "gee_weekly_weather.csv",
        ("FIPS", "year"),
        False,
        "weekly GRIDMET weather features",
    ),
    Component(
        "soil_properties.csv",
        ("FIPS",),
        False,
        "static OpenLandMap 0-30 cm soil features",
    ),
    Component(
        "polaris_soil_multidepth.csv",
        ("FIPS",),
        False,
        "static POLARIS multi-depth soil profile features",
    ),
    Component(
        "geographic_features.csv",
        ("FIPS",),
        False,
        "static county centroid, area, and optional elevation features",
    ),
    Component(
        "irrigation_features.csv",
        ("FIPS",),
        False,
        "static satellite and Census irrigation features",
    ),
    Component(
        "nass_state_year_features.csv",
        ("state_fips", "year"),
        False,
        "state-level NASS crop progress and condition features",
    ),
    Component(
        "historical_yield_features.csv",
        ("FIPS", "year"),
        False,
        "past-year historical yield features",
    ),
]


def build_matrix(
    config: PipelineConfig | None = None,
    require_all_components: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge available checkpoints and return `(matrix, report)`.

    The base table is `nass_yield_detrended.csv`, which supplies the target and
    diagnostic trend/anomaly columns. Other components are left-joined.
    """
    cfg = config or get_config()
    base_path = cfg.data_dir / "nass_yield_detrended.csv"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing {base_path}. Run `python -m data_pipeline.scripts.fetch_nass_yield` first."
        )

    matrix = pd.read_csv(base_path, dtype={"FIPS": str})
    matrix["FIPS"] = matrix["FIPS"].str.zfill(5)
    matrix["year"] = matrix["year"].astype(int)
    matrix["state_fips"] = matrix["FIPS"].str[:2]
    matrix = matrix[(matrix["year"] >= cfg.start_year) & (matrix["year"] <= cfg.end_year)].copy()

    report_rows = []
    for component in COMPONENTS:
        path = cfg.data_dir / component.filename
        if not path.exists():
            report_rows.append(_report_row(component, "missing", 0, 0, "file not found"))
            if require_all_components or component.required:
                raise FileNotFoundError(f"Missing required component: {path}")
            continue

        before_cols = matrix.shape[1]
        data = _read_component(path, component)
        matrix = _merge_component(matrix, data, component)
        added_cols = matrix.shape[1] - before_cols
        report_rows.append(
            _report_row(component, "merged", len(data), added_cols, str(path))
        )

    drop_extras = [col for col in EXCLUDE_COLUMNS if col in matrix.columns]
    if drop_extras:
        matrix = matrix.drop(columns=drop_extras)

    matrix = matrix.sort_values(["FIPS", "year"]).reset_index(drop=True)
    report = pd.DataFrame(report_rows)
    return matrix, report


def write_matrix(
    output_path: Path | None = None,
    report_path: Path | None = None,
    require_all_components: bool = False,
    config: PipelineConfig | None = None,
) -> tuple[Path, Path]:
    """Build and write the matrix plus a component report."""
    cfg = config or get_config()
    matrix, report = build_matrix(cfg, require_all_components=require_all_components)
    output = output_path or cfg.data_dir / "benchmark_matrix_rebuilt.csv"
    report_output = report_path or cfg.data_dir / "benchmark_matrix_build_report.csv"
    matrix.to_csv(output, index=False)
    report.to_csv(report_output, index=False)
    return output, report_output


def _read_component(path: Path, component: Component) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"FIPS": str, "state_fips": str})
    if "FIPS" in data.columns:
        data["FIPS"] = data["FIPS"].str.zfill(5)
    if "state_fips" in data.columns:
        data["state_fips"] = data["state_fips"].str.zfill(2)
    if "year" in component.keys:
        data["year"] = data["year"].astype(int)

    missing_keys = [key for key in component.keys if key not in data.columns]
    if missing_keys:
        raise ValueError(f"{path} is missing key columns: {missing_keys}")

    return data.drop_duplicates(list(component.keys), keep="first")


def _merge_component(matrix: pd.DataFrame, data: pd.DataFrame, component: Component) -> pd.DataFrame:
    keys = list(component.keys)
    overlap = [col for col in data.columns if col in matrix.columns and col not in keys]
    if overlap:
        data = data.drop(columns=overlap)
    return matrix.merge(data, on=keys, how="left")


def _report_row(
    component: Component,
    status: str,
    rows: int,
    added_columns: int,
    message: str,
) -> dict[str, object]:
    return {
        "component": component.filename,
        "status": status,
        "rows": rows,
        "added_columns": added_columns,
        "required": component.required,
        "description": component.description,
        "message": message,
    }
