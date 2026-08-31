"""Project paths and constants for the clean benchmark."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "USA_data" / "benchmark"
INPUT_MATRIX = DATA_DIR / "benchmark_matrix.csv"

OUT_DIR = ROOT / "reproducible_benchmark" / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"

YEAR_MIN = 2000
YEAR_MAX = 2025
TARGET = "yield_bu_acre"
PAST_TREND = "yield_trend_past"
DESCRIPTIVE_TREND = "yield_trend"
ANOMALY = "yield_anomaly"

STATE_NAMES = {
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "26": "MI",
    "27": "MN",
    "29": "MO",
    "31": "NE",
    "38": "ND",
    "39": "OH",
    "46": "SD",
    "55": "WI",
}

RANDOM_SEED = 42


def ensure_output_dirs() -> None:
    """Create output folders used by reproduction scripts."""
    for path in (OUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
