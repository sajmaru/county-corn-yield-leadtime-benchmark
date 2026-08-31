"""Paper figures for the detailed lead-time analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLORS = {
    "with_nass": "#1565C0",
    "without_nass": "#E65100",
    "loyo": "#1565C0",
    "wf": "#E65100",
    "loso": "#2E7D32",
    "nass_gap": "#6A1B9A",
}


def write_leadtime_figures(
    results: pd.DataFrame,
    per_fold: pd.DataFrame,
    gap: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    """Write the five lead-time figures used by the paper."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _set_style(plt)
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written += [_fig_main_curve(results, figure_dir, plt, mticker)]
    written += [_fig_cv_protocols(results, figure_dir, plt)]
    written += [_fig_nass_gap(results, gap, figure_dir, plt)]
    written += [_fig_per_year_heatmap(per_fold, figure_dir, plt)]
    written += [_fig_rmse_curve(results, figure_dir, plt)]
    return written


def _fig_main_curve(results, figure_dir, plt, mticker) -> Path:
    with_nass, no_nass, labels, labels_multi, x = _series(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, with_nass["loyo_r2"], "o-", color=COLORS["with_nass"], lw=2.5, ms=10, label="With NASS")
    ax.plot(x, no_nass["loyo_r2"], "s--", color=COLORS["without_nass"], lw=2, ms=9, label="Without NASS")
    for i, value in enumerate(with_nass["loyo_r2"]):
        ax.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 14 if i % 2 == 0 else -20), ha="center", fontsize=10, fontweight="bold", color=COLORS["with_nass"])
    ax.fill_between(x, no_nass["loyo_r2"], with_nass["loyo_r2"], alpha=0.12, color=COLORS["nass_gap"], label="NASS information gain")
    ax.axvline(2.5, color="gray", lw=1, ls=":", alpha=0.7)
    ax.annotate("July data\nbecomes available\n(pollination window)", xy=(2.5, with_nass["loyo_r2"].iloc[3]), xytext=(1.2, with_nass["loyo_r2"].iloc[3] - 0.04), fontsize=9, color="gray", arrowprops={"arrowstyle": "->", "color": "gray"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels_multi)
    ax.set_ylabel("$R^2$ (LOYO cross-validation)")
    ax.set_xlabel("Forecast date")
    ax.set_title("In-Season Forecast Accuracy: How Early Can We Predict Corn Yield?", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(v)} mo" for v in with_nass["lead_months"]], fontsize=9)
    ax2.set_xlabel("Lead time before harvest")
    return _save(fig, figure_dir / "fig_lt1_main_curve", plt)


def _fig_cv_protocols(results, figure_dir, plt) -> Path:
    with_nass, _, _, labels_multi, x = _series(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, with_nass["loyo_r2"], "o-", color=COLORS["loyo"], lw=2.5, ms=9, label="Leave-One-Year-Out")
    ax.plot(x, with_nass["wf_r2"], "s--", color=COLORS["wf"], lw=2, ms=8, label="Walk-Forward")
    ax.plot(x, with_nass["loso_r2"], "^:", color=COLORS["loso"], lw=2, ms=8, label="Leave-One-State-Out")
    for i, value in enumerate(with_nass["loyo_r2"]):
        ax.annotate(f"{value:.3f}", (i, value), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=COLORS["loyo"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels_multi)
    ax.set_ylabel("$R^2$")
    ax.set_xlabel("Forecast date")
    ax.set_title("Forecast Accuracy by Cross-Validation Protocol", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, figure_dir / "fig_lt2_cv_protocols", plt)


def _fig_nass_gap(results, gap, figure_dir, plt) -> Path:
    with_nass, no_nass, _, labels_multi, x = _series(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    delta_pp = gap["loyo_delta_r2"] * 100
    bars = ax1.bar(x, delta_pp, color=COLORS["nass_gap"], edgecolor="white", width=0.6)
    for bar, value in zip(bars, delta_pp):
        offset = 0.10 if value >= 0 else -0.08
        va = "bottom" if value >= 0 else "top"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:+.1f} pp",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_multi, fontsize=9)
    ax1.set_ylabel("Change in LOYO accuracy (percentage points)")
    ax1.set_title("(a) Gain from NASS reports")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_ylim(min(-0.35, float(delta_pp.min()) - 0.25), float(delta_pp.max()) + 0.35)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(x, with_nass["loyo_r2"] * 100, "o-", color=COLORS["with_nass"], lw=2.5, ms=9, label="With NASS")
    ax2.plot(x, no_nass["loyo_r2"] * 100, "s--", color=COLORS["without_nass"], lw=2, ms=8, label="Without NASS")
    ax2.fill_between(x, no_nass["loyo_r2"] * 100, with_nass["loyo_r2"] * 100, alpha=0.15, color=COLORS["nass_gap"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_multi, fontsize=9)
    ax2.set_ylabel("LOYO accuracy (%)")
    ax2.set_title("(b) Forecast accuracy by source")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Value of USDA Crop-Condition Reports", fontweight="bold")
    return _save(fig, figure_dir / "fig_lt3_nass_gap", plt)


def _fig_per_year_heatmap(per_fold, figure_dir, plt) -> Path:
    data = per_fold[(per_fold["variant"] == "with_nass") & (per_fold["protocol"] == "loyo")]
    years = sorted(data["fold"].unique())
    orders = sorted(data["date_order"].unique())
    labels = [data[data["date_order"] == order]["date_label"].iloc[0] for order in orders]
    matrix = np.full((len(orders), len(years)), np.nan)
    for row_i, order in enumerate(orders):
        for col_i, year in enumerate(years):
            row = data[(data["date_order"] == order) & (data["fold"] == year)]
            if not row.empty:
                matrix[row_i, col_i] = row.iloc[0]["r2"]
    fig, ax = plt.subplots(figsize=(16, 5))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0, interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] < 0.65 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=color, fontweight="bold")
    ax.set_yticks(range(len(orders)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Year held out in LOYO")
    ax.set_ylabel("Forecast date")
    ax.set_title("Per-Year LOYO $R^2$ by Forecast Date", fontweight="bold")
    fig.colorbar(image, ax=ax, pad=0.01, fraction=0.02).set_label("$R^2$")
    if 2012 in years:
        drought_col = years.index(2012)
        for row_i in range(len(orders)):
            ax.add_patch(plt.Rectangle((drought_col - 0.5, row_i - 0.5), 1, 1, fill=False, edgecolor="navy", lw=2.5))
    return _save(fig, figure_dir / "fig_lt4_peryear_heatmap", plt)


def _fig_rmse_curve(results, figure_dir, plt) -> Path:
    with_nass, no_nass, _, labels_multi, x = _series(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_metric(ax1, x, with_nass, no_nass, "loyo_rmse", "RMSE (bu/acre)", "Prediction Error")
    _plot_metric(ax2, x, with_nass, no_nass, "loyo_rrmse", "RRMSE (%)", "Relative RMSE")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels_multi, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("Error Metrics Across Forecast Lead Times", fontweight="bold")
    return _save(fig, figure_dir / "fig_lt5_rmse_curve", plt)


def _plot_metric(ax, x, with_nass, no_nass, column, ylabel, title):
    ax.plot(x, with_nass[column], "o-", color=COLORS["with_nass"], lw=2.5, ms=9, label="With NASS")
    ax.plot(x, no_nass[column], "s--", color=COLORS["without_nass"], lw=2, ms=8, label="Without NASS")
    for i, value in enumerate(with_nass[column]):
        suffix = "%" if "rrmse" in column else ""
        ax.annotate(f"{value:.1f}{suffix}", (i, value), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=COLORS["with_nass"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _series(results):
    with_nass = results[results["variant"] == "with_nass"].sort_values("date_order").reset_index(drop=True)
    no_nass = results[results["variant"] == "no_nass"].sort_values("date_order").reset_index(drop=True)
    labels = with_nass["date_label"].tolist()
    labels_multi = [label.replace(" ", "\n", 1) if label != "End of season" else "End of\nseason" for label in labels]
    x = np.arange(len(labels))
    return with_nass, no_nass, labels, labels_multi, x


def _set_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def _save(fig, stem: Path, plt) -> Path:
    fig.tight_layout()
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf} and {png}")
    return pdf
