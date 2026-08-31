"""Shared cross-validation utilities for the clean benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .config import TARGET
from .feature_selection import SelectionConfig, select_features_train_only
from .models import build_model


def prepare_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
):
    """Build train/test arrays with training-fold-only imputation.

    Ridge receives standardized predictors. Tree models keep original scales.
    """
    x_train = train[features].to_numpy(dtype=float, copy=True)
    y_train = train[target].to_numpy(dtype=float, copy=True)
    x_test = test[features].to_numpy(dtype=float, copy=True)
    y_test = test[target].to_numpy(dtype=float, copy=True)

    means = np.nanmean(x_train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    for col in range(x_train.shape[1]):
        x_train[np.isnan(x_train[:, col]), col] = means[col]
        x_test[np.isnan(x_test[:, col]), col] = means[col]

    if model_name == "ridge":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def metrics(y_true, y_pred) -> dict[str, float]:
    """Compute pooled forecast metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r_value = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r": float(r_value),
        "mbe": float(np.mean(y_pred - y_true)),
        "rrmse": float(rmse / np.mean(y_true) * 100),
    }


def _fit_predict(train, test, features, target, model_name):
    x_train, y_train, x_test, y_test = prepare_xy(
        train, test, features, target=target, model_name=model_name
    )
    model = build_model(model_name)
    model.fit(x_train, y_train)
    return y_test, model.predict(x_test)


def leave_one_year_out(
    data: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    return_folds: bool = False,
) -> dict:
    """Leave one calendar year out and pool predictions across folds."""
    all_true: list[float] = []
    all_pred: list[float] = []
    fold_rows: list[dict] = []

    for year in sorted(data["year"].unique()):
        train = data[data["year"] != year]
        test = data[data["year"] == year]
        y_test, pred = _fit_predict(train, test, features, target, model_name)
        all_true.extend(y_test)
        all_pred.extend(pred)
        fold_rows.append({"fold": int(year), "n": len(y_test), "r2": r2_score(y_test, pred)})

    result = metrics(all_true, all_pred)
    if return_folds:
        result["folds"] = fold_rows
    return result


def walk_forward(
    data: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    min_train_years: int = 5,
    return_folds: bool = False,
) -> dict:
    """Train on past years only and test the next year."""
    years = sorted(data["year"].unique())
    all_true: list[float] = []
    all_pred: list[float] = []
    fold_rows: list[dict] = []

    for i, year in enumerate(years):
        if i < min_train_years:
            continue
        train = data[data["year"] < year]
        test = data[data["year"] == year]
        if len(train) == 0 or len(test) == 0:
            continue
        y_test, pred = _fit_predict(train, test, features, target, model_name)
        all_true.extend(y_test)
        all_pred.extend(pred)
        fold_rows.append({"fold": int(year), "n": len(y_test), "r2": r2_score(y_test, pred)})

    result = metrics(all_true, all_pred)
    if return_folds:
        result["folds"] = fold_rows
    return result


def leave_one_state_out(
    data: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    return_folds: bool = False,
) -> dict:
    """Hold out each state FIPS code and pool predictions."""
    all_true: list[float] = []
    all_pred: list[float] = []
    fold_rows: list[dict] = []

    for state in sorted(data["state_fips"].unique()):
        train = data[data["state_fips"] != state]
        test = data[data["state_fips"] == state]
        y_test, pred = _fit_predict(train, test, features, target, model_name)
        all_true.extend(y_test)
        all_pred.extend(pred)
        fold_rows.append({"fold": str(state), "n": len(y_test), "r2": r2_score(y_test, pred)})

    result = metrics(all_true, all_pred)
    if return_folds:
        result["folds"] = fold_rows
    return result


def evaluate_all_protocols(
    data: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    return_folds: bool = False,
) -> dict[str, dict]:
    """Run LOYO, Walk-Forward, and LOSO using one feature set."""
    return {
        "loyo": leave_one_year_out(data, features, target, model_name, return_folds),
        "wf": walk_forward(data, features, target, model_name, return_folds=return_folds),
        "loso": leave_one_state_out(data, features, target, model_name, return_folds),
    }


def _fit_predict_with_selected(train, test, candidates, target, model_name, selection_config):
    selected, ranking = select_features_train_only(
        train,
        candidates,
        target=target,
        config=selection_config,
    )
    y_test, pred = _fit_predict(train, test, selected, target, model_name)
    return y_test, pred, selected, ranking


def _ranking_rows(ranking, metadata: dict) -> list[dict]:
    rows: list[dict] = []
    for _, row in ranking.iterrows():
        item = dict(metadata)
        item.update(
            {
                "feature": row["feature"],
                "rank": int(row["rank"]),
                "importance": row["importance"],
                "selected": bool(row["selected"]),
                "selection_method": row["selection_method"],
            }
        )
        rows.append(item)
    return rows


def leave_one_year_out_nested_selection(
    data: pd.DataFrame,
    candidates: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    selection_config: SelectionConfig | None = None,
    metadata: dict | None = None,
) -> dict:
    """LOYO with feature selection nested inside each held-out-year fold."""
    all_true: list[float] = []
    all_pred: list[float] = []
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    base_meta = metadata or {}

    for year in sorted(data["year"].unique()):
        train = data[data["year"] != year]
        test = data[data["year"] == year]
        y_test, pred, selected, ranking = _fit_predict_with_selected(
            train, test, candidates, target, model_name, selection_config
        )
        all_true.extend(y_test)
        all_pred.extend(pred)
        fold_rows.append(
            {
                "fold": int(year),
                "n": len(y_test),
                "r2": r2_score(y_test, pred),
                "n_candidates": len(candidates),
                "n_selected": len(selected),
            }
        )
        selection_rows.extend(
            _ranking_rows(
                ranking,
                {
                    **base_meta,
                    "protocol": "loyo",
                    "fold": int(year),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_candidates": len(candidates),
                    "n_selected": len(selected),
                },
            )
        )

    result = metrics(all_true, all_pred)
    result["folds"] = fold_rows
    result["selection_rows"] = selection_rows
    return result


def walk_forward_nested_selection(
    data: pd.DataFrame,
    candidates: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    min_train_years: int = 5,
    selection_config: SelectionConfig | None = None,
    metadata: dict | None = None,
) -> dict:
    """Walk-forward CV with train-only nested feature selection."""
    years = sorted(data["year"].unique())
    all_true: list[float] = []
    all_pred: list[float] = []
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    base_meta = metadata or {}

    for i, year in enumerate(years):
        if i < min_train_years:
            continue
        train = data[data["year"] < year]
        test = data[data["year"] == year]
        if len(train) == 0 or len(test) == 0:
            continue
        y_test, pred, selected, ranking = _fit_predict_with_selected(
            train, test, candidates, target, model_name, selection_config
        )
        all_true.extend(y_test)
        all_pred.extend(pred)
        fold_rows.append(
            {
                "fold": int(year),
                "n": len(y_test),
                "r2": r2_score(y_test, pred),
                "n_candidates": len(candidates),
                "n_selected": len(selected),
            }
        )
        selection_rows.extend(
            _ranking_rows(
                ranking,
                {
                    **base_meta,
                    "protocol": "wf",
                    "fold": int(year),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_candidates": len(candidates),
                    "n_selected": len(selected),
                },
            )
        )

    result = metrics(all_true, all_pred)
    result["folds"] = fold_rows
    result["selection_rows"] = selection_rows
    return result


def leave_one_state_out_nested_selection(
    data: pd.DataFrame,
    candidates: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    selection_config: SelectionConfig | None = None,
    metadata: dict | None = None,
) -> dict:
    """LOSO with feature selection nested inside each held-out-state fold."""
    all_true: list[float] = []
    all_pred: list[float] = []
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    base_meta = metadata or {}

    for state in sorted(data["state_fips"].unique()):
        train = data[data["state_fips"] != state]
        test = data[data["state_fips"] == state]
        y_test, pred, selected, ranking = _fit_predict_with_selected(
            train, test, candidates, target, model_name, selection_config
        )
        all_true.extend(y_test)
        all_pred.extend(pred)
        fold_rows.append(
            {
                "fold": str(state),
                "n": len(y_test),
                "r2": r2_score(y_test, pred),
                "n_candidates": len(candidates),
                "n_selected": len(selected),
            }
        )
        selection_rows.extend(
            _ranking_rows(
                ranking,
                {
                    **base_meta,
                    "protocol": "loso",
                    "fold": str(state),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_candidates": len(candidates),
                    "n_selected": len(selected),
                },
            )
        )

    result = metrics(all_true, all_pred)
    result["folds"] = fold_rows
    result["selection_rows"] = selection_rows
    return result


def evaluate_all_protocols_nested_selection(
    data: pd.DataFrame,
    candidates: list[str],
    target: str = TARGET,
    model_name: str = "hgb",
    selection_config: SelectionConfig | None = None,
    metadata: dict | None = None,
) -> dict[str, dict]:
    """Run all protocols with train-only nested feature selection."""
    return {
        "loyo": leave_one_year_out_nested_selection(
            data, candidates, target, model_name, selection_config, metadata
        ),
        "wf": walk_forward_nested_selection(
            data,
            candidates,
            target,
            model_name,
            selection_config=selection_config,
            metadata=metadata,
        ),
        "loso": leave_one_state_out_nested_selection(
            data, candidates, target, model_name, selection_config, metadata
        ),
    }
