"""Audit nested-selection outputs for obvious leakage mistakes."""

from __future__ import annotations

import pandas as pd

from reproducible_benchmark.config import TABLE_DIR
from reproducible_benchmark.feature_selection import FORBIDDEN_COLUMNS, is_forbidden_feature

SELECTION_FILES = [
    "model_comparison_feature_selection.csv",
    "feature_ablation_feature_selection.csv",
    "leadtime_feature_selection.csv",
]


def assert_no_forbidden_features(frame: pd.DataFrame, path) -> None:
    selected = frame[frame["selected"].astype(bool)]
    bad = selected[selected["feature"].map(is_forbidden_feature)]
    if not bad.empty:
        cols = bad[["feature", "experiment", "protocol", "fold"]].head(20)
        raise AssertionError(f"Forbidden selected features in {path}:\n{cols}")
    explicit_bad = set(selected["feature"]) & FORBIDDEN_COLUMNS
    if explicit_bad:
        raise AssertionError(f"Forbidden columns selected in {path}: {sorted(explicit_bad)}")


def assert_leadtime_rules(frame: pd.DataFrame) -> None:
    selected = frame[frame["selected"].astype(bool)].copy()
    if "variant" in selected.columns:
        no_nass = selected[selected["variant"] == "no_nass"]
        bad_nass = no_nass[no_nass["feature"].str.startswith("nass_", na=False)]
        if not bad_nass.empty:
            raise AssertionError(f"NASS feature selected in no_nass lead-time variant:\n{bad_nass.head(20)}")

    month_limits = {0: 0, 1: 5, 2: 6, 3: 7, 4: 8, 5: 10}
    monthly_prefixes = (
        "tmax_", "tmin_", "precip_", "vpd_", "srad_", "eto_",
        "gdd_", "kdd_", "edd_", "spei90_", "spi90_", "pdsi_",
        "ndvi_", "evi_", "ndwi_",
        "gci_", "lai_", "fpar_", "lst_day_", "lst_night_",
    )
    for _, row in selected.iterrows():
        feature = str(row["feature"])
        if not feature.startswith(monthly_prefixes):
            continue
        try:
            month = int(feature.rsplit("_", 1)[-1])
        except ValueError:
            continue
        allowed = month_limits[int(row["date_order"])]
        if month > allowed:
            raise AssertionError(
                f"Future monthly feature selected for {row['date_label']}: {feature} > month {allowed}"
            )


def audit_file(path) -> None:
    frame = pd.read_csv(path)
    required = {"feature", "selected", "protocol", "fold", "n_train", "n_test"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"{path} missing required columns: {sorted(missing)}")
    assert_no_forbidden_features(frame, path)
    if path.name == "leadtime_feature_selection.csv":
        assert_leadtime_rules(frame)
    print(f"OK {path}")


def main() -> None:
    missing = []
    for name in SELECTION_FILES:
        path = TABLE_DIR / name
        if not path.exists():
            missing.append(path)
            continue
        audit_file(path)
    if missing:
        print("Selection files not present yet:")
        for path in missing:
            print(f"  {path}")
        print("Run experiments, then rerun this audit for full verification.")
    else:
        print("All nested-selection leakage audits passed.")


if __name__ == "__main__":
    main()
