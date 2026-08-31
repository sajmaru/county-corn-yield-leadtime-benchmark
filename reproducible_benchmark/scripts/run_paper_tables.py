"""Reproduce paper model-comparison and feature-ablation tables.

All feature selection is nested inside the outer CV fold. The held-out fold is
never used for feature ranking or feature choice.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import r2_score

from reproducible_benchmark.config import TABLE_DIR, TARGET, ensure_output_dirs
from reproducible_benchmark.cv import metrics, prepare_xy
from reproducible_benchmark.data import load_benchmark_matrix
from reproducible_benchmark.feature_selection import (
    SelectionConfig,
    ablation_candidate_steps,
    end_of_season_candidate_features,
    select_features_train_only,
)
from reproducible_benchmark.models import build_model

MODEL_NAMES = ("ridge", "rf", "hgb")
PROTOCOLS = ("loyo", "wf", "loso")


def _count_folds(data: pd.DataFrame) -> int:
    """Total (protocol, fold) pairs, for progress bars."""
    return sum(1 for protocol in PROTOCOLS for _ in protocol_splits(data, protocol))


def iter_protocol_folds(data: pd.DataFrame, desc: str):
    """Yield (protocol, fold, train, test) across all protocols with a progress bar.

    Falls back to a plain iterator if tqdm is unavailable. The bar is written to
    stderr, so redirecting stdout to a log keeps the bar visible on screen.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # keep runnable without tqdm
        tqdm = None
    bar = tqdm(total=_count_folds(data), desc=desc, unit="fold") if tqdm else None
    for protocol in PROTOCOLS:
        for fold, train, test in protocol_splits(data, protocol):
            yield protocol, fold, train, test
            if bar is not None:
                bar.set_postfix_str(f"{protocol} {fold}")
                bar.update(1)
    if bar is not None:
        bar.close()


def flatten_result(prefix: str, result: dict) -> dict[str, float]:
    """Flatten one protocol result into table columns."""
    return {
        f"{prefix}_r2": result["r2"],
        f"{prefix}_rmse": result["rmse"],
        f"{prefix}_r": result["r"],
        f"{prefix}_rrmse": result["rrmse"],
        f"{prefix}_mbe": result["mbe"],
    }


def protocol_splits(data: pd.DataFrame, protocol: str):
    """Yield fold label, train data, and test data for one protocol."""
    if protocol == "loyo":
        for year in sorted(data["year"].unique()):
            yield int(year), data[data["year"] != year], data[data["year"] == year]
    elif protocol == "wf":
        years = sorted(data["year"].unique())
        for idx, year in enumerate(years):
            if idx < 5:
                continue
            train = data[data["year"] < year]
            test = data[data["year"] == year]
            if len(train) and len(test):
                yield int(year), train, test
    elif protocol == "loso":
        for state in sorted(data["state_fips"].unique()):
            yield str(state), data[data["state_fips"] != state], data[data["state_fips"] == state]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def fit_predict(train, test, features, model_name):
    """Fit one model on a split and return truth and predictions."""
    x_train, y_train, x_test, y_test = prepare_xy(
        train, test, features, target=TARGET, model_name=model_name
    )
    model = build_model(model_name)
    model.fit(x_train, y_train)
    return y_test, model.predict(x_test)


def evaluate_models_with_nested_selection(
    data: pd.DataFrame,
    candidates: list[str],
    experiment: str,
    selection_config: SelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate Ridge, RF, and HGB using one nested selected set per fold."""
    predictions = {model: {protocol: {"true": [], "pred": []} for protocol in PROTOCOLS} for model in MODEL_NAMES}
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []

    for protocol, fold, train, test in iter_protocol_folds(data, desc=experiment):
        selected, ranking = select_features_train_only(
            train, candidates, target=TARGET, config=selection_config
        )
        for _, row in ranking.iterrows():
            selection_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_candidates": len(candidates),
                    "n_selected": len(selected),
                    "feature": row["feature"],
                    "rank": int(row["rank"]),
                    "importance": row["importance"],
                    "selected": bool(row["selected"]),
                    "selection_method": row["selection_method"],
                }
            )

        for model_name in MODEL_NAMES:
            y_test, pred = fit_predict(train, test, selected, model_name)
            predictions[model_name][protocol]["true"].extend(y_test)
            predictions[model_name][protocol]["pred"].extend(pred)
            fold_rows.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "fold": str(fold),
                    "r2": r2_score(y_test, pred),
                    "n": len(y_test),
                    "n_candidates": len(candidates),
                    "n_selected": len(selected),
                }
            )

    table_rows = []
    for model_name in MODEL_NAMES:
        row = {
            "model": model_name,
            "n_features": selection_config.max_features,
            "n_candidates": len(candidates),
        }
        for protocol in PROTOCOLS:
            result = metrics(
                predictions[model_name][protocol]["true"],
                predictions[model_name][protocol]["pred"],
            )
            row.update(flatten_result(protocol, result))
        table_rows.append(row)

    return pd.DataFrame(table_rows), pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def run_ablation(data: pd.DataFrame, selection_config: SelectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cumulative source-group ablation with nested selection."""
    rows = []
    selections = []
    previous_r2 = None
    for step, candidates in ablation_candidate_steps(data):
        print(f"Ablation: {step} ({len(candidates)} candidates)", flush=True)
        table, _, selection = evaluate_single_model_nested(
            data, candidates, "hgb", f"ablation_{step}", selection_config
        )
        row = table.iloc[0].to_dict()
        row["step"] = step
        row["n_candidates"] = len(candidates)
        row["n_features"] = min(len(candidates), selection_config.max_features)
        row["loyo_delta_r2"] = None if previous_r2 is None else row["loyo_r2"] - previous_r2
        previous_r2 = row["loyo_r2"]
        rows.append(row)
        selections.append(selection)
    return pd.DataFrame(rows), pd.concat(selections, ignore_index=True)


def evaluate_single_model_nested(
    data: pd.DataFrame,
    candidates: list[str],
    model_name: str,
    experiment: str,
    selection_config: SelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one model with nested feature selection."""
    pred_by_protocol = {protocol: {"true": [], "pred": []} for protocol in PROTOCOLS}
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []

    for protocol, fold, train, test in iter_protocol_folds(data, desc=experiment):
        selected, ranking = select_features_train_only(
            train, candidates, target=TARGET, config=selection_config
        )
        for _, row in ranking.iterrows():
            selection_rows.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_candidates": len(candidates),
                    "n_selected": len(selected),
                    "feature": row["feature"],
                    "rank": int(row["rank"]),
                    "importance": row["importance"],
                    "selected": bool(row["selected"]),
                    "selection_method": row["selection_method"],
                }
            )
        y_test, pred = fit_predict(train, test, selected, model_name)
        pred_by_protocol[protocol]["true"].extend(y_test)
        pred_by_protocol[protocol]["pred"].extend(pred)
        fold_rows.append(
            {
                "model": model_name,
                "protocol": protocol,
                "fold": str(fold),
                "r2": r2_score(y_test, pred),
                "n": len(y_test),
                "n_candidates": len(candidates),
                "n_selected": len(selected),
            }
        )

    row = {"model": model_name}
    for protocol in PROTOCOLS:
        row.update(flatten_result(protocol, metrics(pred_by_protocol[protocol]["true"], pred_by_protocol[protocol]["pred"])))
    return pd.DataFrame([row]), pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def write_model_folds(folds: pd.DataFrame) -> None:
    """Write combined and per-model fold files used by significance tests."""
    combined = TABLE_DIR / "model_comparison_per_fold_clean.csv"
    folds.to_csv(combined, index=False)
    print(f"Wrote {combined}", flush=True)
    for model_name in MODEL_NAMES:
        path = TABLE_DIR / f"model_comparison_{model_name}_per_fold_clean.csv"
        folds[folds["model"] == model_name].to_csv(path, index=False)
        print(f"Wrote {path}", flush=True)


def main() -> None:
    ensure_output_dirs()
    data = load_benchmark_matrix()
    selection_config = SelectionConfig()

    candidates = end_of_season_candidate_features(data, with_nass=True)
    model_df, folds_df, model_selection = evaluate_models_with_nested_selection(
        data, candidates, "model_comparison", selection_config
    )
    model_path = TABLE_DIR / "model_comparison_clean.csv"
    model_df.to_csv(model_path, index=False)
    print(f"Wrote {model_path}", flush=True)
    write_model_folds(folds_df)
    selection_path = TABLE_DIR / "model_comparison_feature_selection.csv"
    model_selection.to_csv(selection_path, index=False)
    print(f"Wrote {selection_path}", flush=True)

    ablation_df, ablation_selection = run_ablation(data, selection_config)
    ablation_path = TABLE_DIR / "feature_ablation_clean.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Wrote {ablation_path}", flush=True)
    ablation_selection_path = TABLE_DIR / "feature_ablation_feature_selection.csv"
    ablation_selection.to_csv(ablation_selection_path, index=False)
    print(f"Wrote {ablation_selection_path}", flush=True)


if __name__ == "__main__":
    main()
