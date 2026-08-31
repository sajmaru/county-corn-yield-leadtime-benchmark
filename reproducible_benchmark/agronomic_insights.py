"""Mine agronomic insight candidates for the paper discussion.

This module is intentionally descriptive. It does not train models. It summarizes
observed county-year relationships that can support Section 4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - optional local dependency
    smf = None


STATE_ABBR = {
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    26: "MI",
    27: "MN",
    29: "MO",
    31: "NE",
    38: "ND",
    39: "OH",
    46: "SD",
    55: "WI",
}


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]

    @property
    def matrix(self) -> Path:
        return self.root / "USA_data" / "benchmark" / "benchmark_matrix.csv"

    @property
    def county_shape(self) -> Path:
        return self.root / "geo" / "cb_2022_us_county_500k.shp"

    @property
    def output_dir(self) -> Path:
        return self.root / "reproducible_benchmark" / "outputs" / "agronomic_insights"


def _needed_columns() -> list[str]:
    base = [
        "FIPS",
        "year",
        "yield_bu_acre",
        "yield_trend",
        "yield_anomaly",
        "state_fips",
        "lat",
        "lon",
        "lgrip_irrig_of_crop",
        "soil_awc",
        "soil_sand_pct",
        "soil_clay_pct",
        "soil_organic_carbon",
        "soil_theta_s_0_5",
        "soil_theta_s_30_60",
    ]
    monthly = []
    for var in ["tmax", "precip", "kdd", "edd", "vpd", "spei90", "spi90", "pdsi"]:
        monthly.extend([f"{var}_{m:02d}" for m in range(1, 13)])
    veg = []
    for var in ["evi", "ndvi", "ndwi", "lai", "fpar", "lst_day"]:
        veg.extend([f"{var}_{m:02d}" for m in range(4, 11)])
    extras = ["vpd_jja_mean", "kdd_jja_total", "edd_jja_total", "pdsi_jja_mean", "evi_jja_mean"]
    return base + monthly + veg + extras


def load_data(paths: Paths) -> pd.DataFrame:
    cols = pd.read_csv(paths.matrix, nrows=0).columns
    usecols = [c for c in _needed_columns() if c in cols]
    df = pd.read_csv(paths.matrix, usecols=usecols)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)].copy()
    df["state"] = df["state_fips"].map(STATE_ABBR)
    df["county_fips"] = df["FIPS"].astype(int).astype(str).str.zfill(5)
    return add_county_names(df, paths.county_shape)


def add_county_names(df: pd.DataFrame, shp_path: Path) -> pd.DataFrame:
    try:
        import geopandas as gpd

        counties = gpd.read_file(shp_path)[["GEOID", "NAME", "STUSPS"]]
        counties = counties.rename(columns={"GEOID": "county_fips", "NAME": "county", "STUSPS": "shape_state"})
        return df.merge(counties, on="county_fips", how="left")
    except Exception:
        df["county"] = pd.NA
        df["shape_state"] = pd.NA
        return df


def contrast(df: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, float | str | int]:
    a = df.loc[mask, "yield_anomaly"].dropna()
    b = df.loc[~mask, "yield_anomaly"].dropna()
    return {
        "contrast": label,
        "n_true": int(a.size),
        "n_false": int(b.size),
        "mean_true": float(a.mean()),
        "mean_false": float(b.mean()),
        "difference_true_minus_false": float(a.mean() - b.mean()),
    }


def quartile_contrast(df: pd.DataFrame, col: str, label: str, high_good: bool = True) -> dict[str, float | str | int]:
    q1, q4 = df[col].quantile([0.25, 0.75])
    low = df[df[col] <= q1]["yield_anomaly"].dropna()
    high = df[df[col] >= q4]["yield_anomaly"].dropna()
    diff = high.mean() - low.mean() if high_good else low.mean() - high.mean()
    return {
        "contrast": label,
        "variable": col,
        "q25": float(q1),
        "q75": float(q4),
        "n_low": int(low.size),
        "n_high": int(high.size),
        "mean_low": float(low.mean()),
        "mean_high": float(high.mean()),
        "difference_high_minus_low": float(high.mean() - low.mean()),
        "reported_direction_difference": float(diff),
    }


def monthly_correlations(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        if col in {"yield_bu_acre", "yield_trend", "yield_anomaly"}:
            continue
        if any(col.startswith(prefix) for prefix in ("tmax_", "precip_", "kdd_", "edd_", "vpd_", "spei90_", "spi90_", "pdsi_", "evi_", "ndvi_", "ndwi_", "lai_", "fpar_", "lst_day_")):
            pair = df[[col, "yield_anomaly"]].dropna()
            if len(pair) > 50 and pair[col].std() > 0:
                r, p = stats.pearsonr(pair[col], pair["yield_anomaly"])
                records.append({"feature": col, "r": r, "abs_r": abs(r), "p": p, "n": len(pair)})
    return pd.DataFrame(records).sort_values("abs_r", ascending=False)


def stress_by_state(df: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    records = []
    for state, sub in df.groupby("state"):
        pair = sub[[stress_col, "yield_anomaly"]].dropna()
        if len(pair) < 100 or pair[stress_col].std() == 0:
            continue
        slope, intercept, r, p, se = stats.linregress(pair[stress_col], pair["yield_anomaly"])
        records.append({"state": state, "n": len(pair), "slope_bu_per_unit": slope, "r": r, "p": p, "mean_anomaly": pair["yield_anomaly"].mean()})
    return pd.DataFrame(records).sort_values("slope_bu_per_unit")


def interaction_contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Global dry/wet contrasts. Useful, but can be geographically confounded."""
    drought = df["spei90_07"] <= df["spei90_07"].quantile(0.20)
    wet = df["spei90_07"] >= df["spei90_07"].quantile(0.80)
    low_awc = df["soil_awc"] <= df["soil_awc"].quantile(0.33)
    high_awc = df["soil_awc"] >= df["soil_awc"].quantile(0.67)
    low_irrig = df["lgrip_irrig_of_crop"].fillna(0) <= df["lgrip_irrig_of_crop"].fillna(0).quantile(0.33)
    high_irrig = df["lgrip_irrig_of_crop"].fillna(0) >= df["lgrip_irrig_of_crop"].fillna(0).quantile(0.67)
    groups = {
        "dry_low_awc": drought & low_awc,
        "dry_high_awc": drought & high_awc,
        "dry_low_irrig": drought & low_irrig,
        "dry_high_irrig": drought & high_irrig,
        "wet_low_awc": wet & low_awc,
        "wet_high_awc": wet & high_awc,
    }
    records = []
    for name, mask in groups.items():
        vals = df.loc[mask, "yield_anomaly"].dropna()
        records.append({"group": name, "n": len(vals), "mean_yield_anomaly": vals.mean(), "median_yield_anomaly": vals.median()})
    return pd.DataFrame(records)


def within_state_dry_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Compare high/low irrigation and soil AWC inside each state during dry Julys."""
    records = []
    for state, sub in df.dropna(subset=["spei90_07", "yield_anomaly"]).groupby("state"):
        if len(sub) < 100:
            continue
        dry_cut = sub["spei90_07"].quantile(0.20)
        dry = sub[sub["spei90_07"] <= dry_cut].copy()
        if len(dry) < 40:
            continue
        for var, label in [("lgrip_irrig_of_crop", "irrigation"), ("soil_awc", "soil_awc")]:
            x = dry[var].fillna(0) if var == "lgrip_irrig_of_crop" else dry[var]
            q_low, q_high = x.quantile([0.33, 0.67])
            low = dry.loc[x <= q_low, "yield_anomaly"].dropna()
            high = dry.loc[x >= q_high, "yield_anomaly"].dropna()
            if len(low) < 10 or len(high) < 10:
                continue
            records.append(
                {
                    "state": state,
                    "factor": label,
                    "dry_cut_spei07": dry_cut,
                    "n_low": len(low),
                    "n_high": len(high),
                    "mean_low": low.mean(),
                    "mean_high": high.mean(),
                    "high_minus_low": high.mean() - low.mean(),
                }
            )
    return pd.DataFrame(records).sort_values(["factor", "high_minus_low"])


def fixed_effect_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """State and year fixed-effect checks for drought buffering.

    Drought intensity is -SPEI, so larger values mean drier July. A positive
    drought_intensity coefficient means drier Julys lower yield anomaly if the
    dependent variable is multiplied by -1? To keep signs intuitive, we model
    yield_anomaly directly: negative drought coefficients are bad. A positive
    interaction with irrigation/AWC means the negative drought slope is reduced.
    """
    if smf is None:
        return pd.DataFrame([{"model": "statsmodels_missing"}])
    cols = ["yield_anomaly", "spei90_07", "lgrip_irrig_of_crop", "soil_awc", "state", "year"]
    d = df[cols].dropna().copy()
    d["drought_intensity"] = -d["spei90_07"]
    for col in ["drought_intensity", "lgrip_irrig_of_crop", "soil_awc"]:
        std = d[col].std()
        d[col + "_z"] = (d[col] - d[col].mean()) / std if std else 0
    formulas = {
        "state_year_fe_irrig": "yield_anomaly ~ drought_intensity_z * lgrip_irrig_of_crop_z + C(state) + C(year)",
        "state_year_fe_awc": "yield_anomaly ~ drought_intensity_z * soil_awc_z + C(state) + C(year)",
        "state_year_fe_both": "yield_anomaly ~ drought_intensity_z * lgrip_irrig_of_crop_z + drought_intensity_z * soil_awc_z + C(state) + C(year)",
    }
    rows = []
    for name, formula in formulas.items():
        model = smf.ols(formula, data=d).fit(cov_type="HC3")
        for term in [
            "drought_intensity_z",
            "lgrip_irrig_of_crop_z",
            "soil_awc_z",
            "drought_intensity_z:lgrip_irrig_of_crop_z",
            "drought_intensity_z:soil_awc_z",
        ]:
            if term in model.params:
                rows.append(
                    {
                        "model": name,
                        "term": term,
                        "coef_bu": model.params[term],
                        "p_value": model.pvalues[term],
                        "n": int(model.nobs),
                        "r2": model.rsquared,
                    }
                )
    return pd.DataFrame(rows)


def year_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("year").agg(
        n=("yield_anomaly", "size"),
        mean_yield=("yield_bu_acre", "mean"),
        mean_anomaly=("yield_anomaly", "mean"),
        sd_anomaly=("yield_anomaly", "std"),
        p10_anomaly=("yield_anomaly", lambda s: s.quantile(0.10)),
        p90_anomaly=("yield_anomaly", lambda s: s.quantile(0.90)),
        tmax_07=("tmax_07", "mean"),
        precip_07=("precip_07", "mean"),
        spei90_07=("spei90_07", "mean"),
        evi_07=("evi_07", "mean"),
    ).reset_index()
    return out.sort_values("mean_anomaly")


def county_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    threshold = df["yield_anomaly"].quantile(0.10)
    sub = df.assign(bottom_decile=df["yield_anomaly"] <= threshold)
    out = sub.groupby(["county_fips", "state", "county"], dropna=False).agg(
        years=("year", "nunique"),
        bad_years=("bottom_decile", "sum"),
        mean_anomaly=("yield_anomaly", "mean"),
        sd_anomaly=("yield_anomaly", "std"),
        mean_spei07=("spei90_07", "mean"),
        soil_awc=("soil_awc", "mean"),
        irrig=("lgrip_irrig_of_crop", "mean"),
    ).reset_index()
    out["bad_year_share"] = out["bad_years"] / out["years"]
    return out[out["years"] >= 20].sort_values(["bad_year_share", "mean_anomaly"], ascending=[False, True])


def run() -> None:
    paths = Paths()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(paths)

    contrasts = [
        contrast(df, df["tmax_07"] > 35, "July tmax > 35 C vs all other county-years"),
        contrast(df, df["spei90_07"] < -1.5, "July SPEI-90 < -1.5 vs all other county-years"),
        contrast(df, df["kdd_07"] >= df["kdd_07"].quantile(0.90), "Top decile July KDD vs rest"),
        contrast(df, df["vpd_07"] >= df["vpd_07"].quantile(0.90), "Top decile July VPD vs rest"),
        quartile_contrast(df, "precip_07", "July precipitation wettest vs driest quartile"),
        quartile_contrast(df, "evi_07", "July EVI greenest vs weakest quartile"),
        quartile_contrast(df, "pdsi_07", "July PDSI (drought) wettest vs driest quartile"),
    ]
    pd.DataFrame(contrasts).to_csv(paths.output_dir / "threshold_contrasts.csv", index=False)
    monthly_correlations(df).to_csv(paths.output_dir / "monthly_correlations.csv", index=False)
    stress_by_state(df, "spei90_07").to_csv(paths.output_dir / "state_spei07_sensitivity.csv", index=False)
    stress_by_state(df, "tmax_07").to_csv(paths.output_dir / "state_tmax07_sensitivity.csv", index=False)
    interaction_contrast(df).to_csv(paths.output_dir / "soil_irrigation_interactions.csv", index=False)
    within_state_dry_contrasts(df).to_csv(paths.output_dir / "within_state_dry_contrasts.csv", index=False)
    fixed_effect_interactions(df).to_csv(paths.output_dir / "fixed_effect_interactions.csv", index=False)
    year_summary(df).to_csv(paths.output_dir / "year_summary.csv", index=False)
    county_recurrence(df).head(50).to_csv(paths.output_dir / "county_recurrent_losses_top50.csv", index=False)

    print(f"Rows: {len(df):,}; counties: {df['FIPS'].nunique():,}; years: {df['year'].nunique()}")
    print("\nThreshold contrasts")
    print(pd.DataFrame(contrasts).to_string(index=False))
    print("\nTop correlations")
    print(monthly_correlations(df).head(20).to_string(index=False))
    print("\nWorst years")
    print(year_summary(df).head(8).to_string(index=False))
    print("\nGlobal soil/irrigation interactions")
    print(interaction_contrast(df).to_string(index=False))
    print("\nWithin-state dry July contrasts")
    print(within_state_dry_contrasts(df).to_string(index=False))
    print("\nFixed-effect interaction checks")
    print(fixed_effect_interactions(df).to_string(index=False))
    print("\nMost drought-sensitive states by July SPEI slope")
    print(stress_by_state(df, "spei90_07").to_string(index=False))
    print("\nRecurrent county losses")
    print(county_recurrence(df).head(15).to_string(index=False))
    print(f"\nWrote outputs to {paths.output_dir}")


if __name__ == "__main__":
    run()
