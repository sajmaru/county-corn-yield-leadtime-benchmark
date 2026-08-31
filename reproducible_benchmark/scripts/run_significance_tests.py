"""Run paired significance tests from nested model-comparison folds."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from reproducible_benchmark.config import TABLE_DIR, ensure_output_dirs
from reproducible_benchmark.feature_selection import SELECTION_METHOD

REFERENCE_MODEL = "hgb"
PROTOCOL_LABELS = {"loyo": "LOYO", "wf": "Walk-Forward", "loso": "LOSO"}
MODEL_LABELS = {"ridge": "Ridge", "rf": "Random Forest", "hgb": "HistGradientBoosting"}


def load_nested_folds() -> pd.DataFrame:
    """Load fold outputs produced by run_paper_tables."""
    path = TABLE_DIR / "model_comparison_per_fold_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_paper_tables first.")
    folds = pd.read_csv(path, dtype={"fold": str})
    required = {"model", "protocol", "fold", "r2", "n", "n_selected"}
    missing = required - set(folds.columns)
    if missing:
        raise ValueError(f"Fold file is missing nested-selection columns: {sorted(missing)}")
    return folds


def paired_test_rows(folds: pd.DataFrame) -> pd.DataFrame:
    """Compute paired tests against HistGradientBoosting."""
    rows: list[dict[str, object]] = []
    for protocol in ("loyo", "wf", "loso"):
        sub = folds[folds["protocol"] == protocol]
        ref = sub[sub["model"] == REFERENCE_MODEL][["fold", "r2"]].rename(columns={"r2": "reference_r2"})
        if ref.empty:
            continue
        for model_name in sorted(m for m in sub["model"].unique() if m != REFERENCE_MODEL):
            other = sub[sub["model"] == model_name][["fold", "r2"]].rename(columns={"r2": "model_r2"})
            paired = ref.merge(other, on="fold", how="inner").dropna()
            if len(paired) < 3:
                continue
            delta = paired["reference_r2"].to_numpy() - paired["model_r2"].to_numpy()
            try:
                w_stat, w_p = wilcoxon(paired["reference_r2"], paired["model_r2"])
            except ValueError:
                w_stat, w_p = math.nan, math.nan
            t_stat, t_p = ttest_rel(paired["reference_r2"], paired["model_r2"])
            rows.append(
                {
                    "protocol": protocol,
                    "comparison": f"HGB vs {MODEL_LABELS.get(model_name, model_name)}",
                    "reference_model": REFERENCE_MODEL,
                    "model": model_name,
                    "n_folds": len(paired),
                    "mean_delta_r2": float(np.mean(delta)),
                    "median_delta_r2": float(np.median(delta)),
                    "wilcoxon_w": float(w_stat) if not math.isnan(w_stat) else math.nan,
                    "wilcoxon_p": float(w_p) if not math.isnan(w_p) else math.nan,
                    "paired_t": float(t_stat),
                    "paired_t_p": float(t_p),
                    "selection_method": SELECTION_METHOD,
                }
            )
    return pd.DataFrame(rows)


def format_p(value: float) -> str:
    """Format p-values for manuscript tables."""
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write_latex_table(results: pd.DataFrame) -> None:
    """Write compact LaTeX appendix table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired significance tests for model differences using per-fold $R^2$. Positive $\Delta R^2$ means HistGradientBoosting performed better than the comparison model. Feature selection is nested within each outer fold.}",
        r"\label{tab:appendixsignificance}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Protocol & Comparison & $n$ & $\Delta R^2$ & Wilcoxon $p$ & Paired $t$ $p$ \\",
        r"\midrule",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"{PROTOCOL_LABELS[row['protocol']]} & {row['comparison']} & "
            f"{int(row['n_folds'])} & {row['mean_delta_r2']:.3f} & "
            f"{format_p(row['wilcoxon_p'])} & {format_p(row['paired_t_p'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path = TABLE_DIR / "model_significance_latex_table.tex"
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def main() -> None:
    ensure_output_dirs()
    folds = load_nested_folds()
    all_path = TABLE_DIR / "model_comparison_all_per_fold_clean.csv"
    folds.to_csv(all_path, index=False)
    print(f"Wrote {all_path}")

    results = paired_test_rows(folds)
    result_path = TABLE_DIR / "model_significance_clean.csv"
    results.to_csv(result_path, index=False)
    print(f"Wrote {result_path}")
    write_latex_table(results)


if __name__ == "__main__":
    main()
