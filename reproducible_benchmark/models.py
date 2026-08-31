"""Model builders used by the reproduction scripts."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from .config import RANDOM_SEED


def build_model(model_name: str):
    """Return an unfitted model by short name."""
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=8,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
