"""Detailed lead-time analysis used by the paper.

This preserves the important logic from the original top-level
``leadtime_analysis.py`` while using the cleaned benchmark modules for data,
features, and cross-validation. Outputs are written under
``reproducible_benchmark/outputs`` instead of the data directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FIGURE_DIR, TABLE_DIR, TARGET
from .cv import evaluate_all_protocols_nested_selection
from .feature_selection import SelectionConfig
from .features import FORECAST_DATES, leadtime_features


def run_detailed_leadtime(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run six forecast dates × with/without NASS × three nested CV protocols."""
    rows: list[dict] = []
    per_fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    selection_config = SelectionConfig()

    try:
        from tqdm import tqdm
    except ImportError:  # keep runnable without tqdm
        tqdm = None
    combos = [
        (date_order, forecast_date, variant, with_nass)
        for date_order, forecast_date in enumerate(FORECAST_DATES)
        for variant, with_nass in (("with_nass", True), ("no_nass", False))
    ]
    bar = tqdm(total=len(combos), desc="Lead-time (date×NASS)", unit="combo") if tqdm else None

    for date_order, forecast_date, variant, with_nass in combos:
        candidates = leadtime_features(data, forecast_date, with_nass=with_nass)
        if bar is not None:
            bar.set_postfix_str(f"{forecast_date.label} {variant} ({len(candidates)} cand)")
        else:
            print(
                f"Lead time: {forecast_date.label} | {variant} | "
                f"{len(candidates)} candidates | nested top {selection_config.max_features}"
            )
        result = evaluate_all_protocols_nested_selection(
            data,
            candidates,
            TARGET,
            "hgb",
            selection_config=selection_config,
            metadata={
                "experiment": "leadtime",
                "date_label": forecast_date.label,
                "date_order": date_order,
                "variant": variant,
            },
        )
        row = {
            "date_label": forecast_date.label,
            "date_order": date_order,
            "lead_months": forecast_date.lead_months,
            "variant": variant,
            "n_candidates": len(candidates),
            "n_features": min(len(candidates), selection_config.max_features),
            "description": forecast_date.description,
        }
        for protocol, metrics in result.items():
            row.update(_flatten(protocol, metrics))
            for fold in metrics.get("folds", []):
                per_fold_rows.append(
                    {
                        "date_label": forecast_date.label,
                        "date_order": date_order,
                        "variant": variant,
                        "protocol": protocol,
                        "fold": fold["fold"],
                        "n": fold["n"],
                        "r2": fold["r2"],
                        "n_candidates": fold["n_candidates"],
                        "n_selected": fold["n_selected"],
                    }
                )
            selection_rows.extend(metrics.get("selection_rows", []))
        rows.append(row)
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    results = pd.DataFrame(rows).sort_values(["date_order", "variant"])
    per_fold = pd.DataFrame(per_fold_rows).sort_values(
        ["date_order", "variant", "protocol", "fold"]
    )
    selection = pd.DataFrame(selection_rows)
    gap = build_nass_gap(results, selection)
    return results, per_fold, gap, selection


def build_nass_gap(results: pd.DataFrame, selection: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute marginal contribution of NASS features by forecast date."""
    with_nass = results[results["variant"] == "with_nass"]
    no_nass = results[results["variant"] == "no_nass"]
    gap = with_nass.merge(
        no_nass,
        on=["date_label", "date_order", "lead_months"],
        suffixes=("_with", "_without"),
    )
    gap["loyo_delta_r2"] = gap["loyo_r2_with"] - gap["loyo_r2_without"]
    gap["wf_delta_r2"] = gap["wf_r2_with"] - gap["wf_r2_without"]
    gap["loyo_delta_rmse"] = gap["loyo_rmse_without"] - gap["loyo_rmse_with"]
    if selection is not None and not selection.empty:
        nass_counts = _selected_nass_counts(selection)
        gap = gap.merge(nass_counts, on=["date_label", "date_order"], how="left")
        gap["n_nass_feats"] = gap["n_nass_feats"].fillna(0.0)
    else:
        gap["n_nass_feats"] = gap["n_features_with"] - gap["n_features_without"]
    return gap.sort_values("date_order")


def _selected_nass_counts(selection: pd.DataFrame) -> pd.DataFrame:
    """Return average selected NASS features per outer fold for With NASS runs."""
    selected = selection[selection["selected"].astype(bool)].copy()
    selected = selected[selected["variant"] == "with_nass"]
    selected["is_nass"] = selected["feature"].str.startswith("nass_", na=False)
    per_fold = (
        selected.groupby(["date_label", "date_order", "protocol", "fold"], dropna=False)["is_nass"]
        .sum()
        .reset_index(name="n_nass_selected")
    )
    return (
        per_fold.groupby(["date_label", "date_order"])["n_nass_selected"]
        .mean()
        .reset_index(name="n_nass_feats")
    )


def write_outputs(
    results: pd.DataFrame,
    per_fold: pd.DataFrame,
    gap: pd.DataFrame,
    selection: pd.DataFrame,
    table_dir: Path = TABLE_DIR,
) -> dict[str, Path]:
    """Write detailed lead-time CSVs and LaTeX table."""
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "main": table_dir / "leadtime_main_results.csv",
        "per_fold": table_dir / "leadtime_per_fold.csv",
        "gap": table_dir / "leadtime_nass_gap.csv",
        "latex": table_dir / "leadtime_latex_table.tex",
        "selection": table_dir / "leadtime_feature_selection.csv",
    }
    results.to_csv(paths["main"], index=False)
    per_fold.to_csv(paths["per_fold"], index=False)
    gap.to_csv(paths["gap"], index=False)
    paths["latex"].write_text(build_latex_table(results, gap))
    selection.to_csv(paths["selection"], index=False)
    return paths


def build_latex_table(results: pd.DataFrame, gap: pd.DataFrame) -> str:
    """Build a paper-ready LaTeX table for lead-time accuracy."""
    with_nass = results[results["variant"] == "with_nass"].sort_values("date_order")
    no_nass = results[results["variant"] == "no_nass"].sort_values("date_order")
    best_order = with_nass.loc[with_nass["loyo_r2"].idxmax(), "date_order"]
    no_lookup = no_nass.set_index("date_order")
    gap_lookup = gap.set_index("date_order")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{In-season forecast accuracy by lead time. For each forecast date, the model is trained using only features available on that date. $\Delta R^2_\mathrm{NASS}$ is the marginal gain from USDA crop-condition reports. RMSE is in bu/acre.}",
        r"\label{tab:leadtime}",
        r"\begin{tabular}{l r cccc cc c}",
        r"\toprule",
        r"Forecast date & Lead & \multicolumn{4}{c}{With NASS} & \multicolumn{2}{c}{Without NASS} & $\Delta R^2_\mathrm{NASS}$ \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-8}",
        r"& (mo) & $n_\mathrm{feat}$ & LOYO $R^2$ & WF $R^2$ & LOSO $R^2$ & LOYO $R^2$ & WF $R^2$ & (LOYO) \\",
        r"\midrule",
    ]
    for _, row in with_nass.iterrows():
        order = row["date_order"]
        no_row = no_lookup.loc[order]
        delta = gap_lookup.loc[order, "loyo_delta_r2"]
        is_best = order == best_order
        lines.append(
            f"{row['date_label']} & {int(row['lead_months'])} & {int(row['n_features'])} & "
            f"{_fmt(row['loyo_r2'], is_best)} & {_fmt(row['wf_r2'], is_best)} & "
            f"{_fmt(row['loso_r2'], is_best)} & {_fmt(no_row['loyo_r2'], is_best)} & "
            f"{_fmt(no_row['wf_r2'], is_best)} & {delta:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def print_key_numbers(results: pd.DataFrame, gap: pd.DataFrame) -> None:
    """Print the lead-time headline numbers used in the paper narrative."""
    with_nass = results[results["variant"] == "with_nass"].set_index("date_order")
    gap_by_order = gap.set_index("date_order")
    print("\nKEY NUMBERS FOR PAPER — LEAD-TIME SECTION")
    print("=" * 72)
    for order, label in enumerate([date.label for date in FORECAST_DATES]):
        row = with_nass.loc[order]
        print(
            f"{label:<14} lead={int(row['lead_months'])}mo  "
            f"LOYO R2={row['loyo_r2']:.4f}  RMSE={row['loyo_rmse']:.2f}  "
            f"WF R2={row['wf_r2']:.4f}  LOSO R2={row['loso_r2']:.4f}"
        )
    print("\nKey deltas")
    print(f"  Apr 1 -> Aug 1 gain: {with_nass.loc[3, 'loyo_r2'] - with_nass.loc[0, 'loyo_r2']:+.4f}")
    print(f"  Jul 1 -> Aug 1 gain: {with_nass.loc[3, 'loyo_r2'] - with_nass.loc[2, 'loyo_r2']:+.4f}")
    print(f"  Aug 1 -> End gain:   {with_nass.loc[5, 'loyo_r2'] - with_nass.loc[3, 'loyo_r2']:+.4f}")
    print("\nNASS contribution")
    for order, row in gap_by_order.iterrows():
        print(
            f"  {row['date_label']:<14} dR2={row['loyo_delta_r2']:+.4f}  "
            f"dRMSE={row['loyo_delta_rmse']:+.2f}  n_nass={int(row['n_nass_feats'])}"
        )
    print(f"\nTables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")


def _flatten(prefix: str, result: dict) -> dict[str, float]:
    return {
        f"{prefix}_r2": result["r2"],
        f"{prefix}_rmse": result["rmse"],
        f"{prefix}_mae": result["mae"],
        f"{prefix}_r": result["r"],
        f"{prefix}_mbe": result["mbe"],
        f"{prefix}_rrmse": result["rrmse"],
    }


def _fmt(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if bold else text
