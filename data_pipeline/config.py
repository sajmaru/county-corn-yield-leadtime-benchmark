"""Configuration and environment loading for the source-data pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
DEFAULT_DATA_DIR = ROOT / "USA_data" / "benchmark"
CACHE_DIR = ROOT / "data_pipeline" / "cache"
OUTPUT_DIR = ROOT / "data_pipeline" / "outputs"

STATES = {
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "26": "Michigan",
    "27": "Minnesota",
    "29": "Missouri",
    "31": "Nebraska",
    "38": "North Dakota",
    "39": "Ohio",
    "46": "South Dakota",
    "55": "Wisconsin",
}


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency.

    Existing environment variables win over values in the file.
    """
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved runtime configuration."""

    nass_api_key: str | None
    gee_project: str | None
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    start_year: int
    end_year: int


def get_config() -> PipelineConfig:
    """Return configuration from environment variables and .env."""
    load_dotenv()
    data_dir = Path(os.environ.get("BENCHMARK_DATA_DIR", DEFAULT_DATA_DIR)).expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    start_year = int(os.environ.get("BENCHMARK_START_YEAR", "2000"))
    end_year = int(os.environ.get("BENCHMARK_END_YEAR", "2025"))
    return PipelineConfig(
        nass_api_key=os.environ.get("NASS_API_KEY"),
        gee_project=os.environ.get("GEE_PROJECT"),
        data_dir=data_dir,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        start_year=start_year,
        end_year=end_year,
    )


def ensure_dirs(config: PipelineConfig | None = None) -> None:
    """Create cache/output directories."""
    cfg = config or get_config()
    for path in (cfg.cache_dir, cfg.output_dir, cfg.data_dir):
        path.mkdir(parents=True, exist_ok=True)
