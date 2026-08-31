"""Write LaTeX snippets for the model-comparison and feature-ablation tables.

Results-packaging helper (NOT a manuscript edit): reads the authoritative
``*_clean.csv`` tables produced by ``run_paper_tables`` and emits paste-ready
``.tex`` snippets alongside them. Re-run after regenerating
the tables.

    venv/bin/python -m reproducible_benchmark.scripts.write_results_latex
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"

PRETTY_MODEL = {
    "ridge": "Ridge",
    "rf": "Random forest",
    "hgb": "HGB (ours)",
}


def _fmt(x: float, bold: bool = False, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write_model_comparison() -> None:
    df = pd.read_csv(TABLES / "model_comparison_clean.csv").set_index("model")
    order = ["ridge", "rf", "hgb"]

    # best (max) per protocol across all models shown, for bolding
    def val(model, col):
        return float(df.loc[model, col])

    best = {c: max(val(m, c) for m in order) for c in ("loyo_r2", "wf_r2", "loso_r2")}

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{End-of-season model comparison under three cross-validation "
        "protocols (leave-one-year-out, walk-forward, leave-one-state-out). "
        "Features are selected inside each training fold (top~81 by TreeSHAP); "
        "$n_\\mathrm{feat}$ is the number of selected features "
        "($n_\\mathrm{cand}=251$ candidates). RMSE is LOYO, in bu/acre.}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{l r ccc r}",
        "\\toprule",
        "Model & $n_\\mathrm{feat}$ & LOYO $R^2$ & WF $R^2$ & LOSO $R^2$ "
        "& LOYO RMSE \\\\",
        "\\midrule",
    ]
    for m in order:
        nfeat = int(val(m, "n_features"))
        row = (
            f"{PRETTY_MODEL[m]} & {nfeat} & "
            f"{_fmt(val(m, 'loyo_r2'), val(m, 'loyo_r2') == best['loyo_r2'])} & "
            f"{_fmt(val(m, 'wf_r2'), val(m, 'wf_r2') == best['wf_r2'])} & "
            f"{_fmt(val(m, 'loso_r2'), val(m, 'loso_r2') == best['loso_r2'])} & "
            f"{val(m, 'loyo_rmse'):.2f} \\\\"
        )
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    out = TABLES / "model_comparison_latex_table.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def write_ablation() -> None:
    df = pd.read_csv(TABLES / "feature_ablation_clean.csv")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cumulative feature-group ablation (HGB, leave-one-year-out). "
        "Groups are added in order; $n_\\mathrm{cand}$ is the candidate pool and "
        "$n_\\mathrm{feat}=\\min(81, n_\\mathrm{cand})$ features are selected "
        "inside each training fold. $\\Delta R^2$ is the change from the "
        "previous row.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{l r r c c}",
        "\\toprule",
        "Feature groups & $n_\\mathrm{cand}$ & $n_\\mathrm{feat}$ "
        "& LOYO $R^2$ & $\\Delta R^2$ \\\\",
        "\\midrule",
    ]
    last = None
    n = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        step = str(r["step"]).split(":", 1)[-1].strip()
        r2 = float(r["loyo_r2"])
        d = "" if last is None else f"{r2 - last:+.3f}"
        r2_str = _fmt(r2, bold=(i == n - 1))
        lines.append(
            f"{step} & {int(r['n_candidates'])} & {int(r['n_features'])} & "
            f"{r2_str} & {d} \\\\"
        )
        last = r2
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    out = TABLES / "feature_ablation_latex_table.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> None:
    write_model_comparison()
    write_ablation()


if __name__ == "__main__":
    main()
