"""Validate the benchmark input matrix and print manuscript data facts."""

from __future__ import annotations

from reproducible_benchmark.config import STATE_NAMES, TABLE_DIR, ensure_output_dirs
from reproducible_benchmark.data import dataset_summary, load_benchmark_matrix, state_summary


def main() -> None:
    ensure_output_dirs()
    data = load_benchmark_matrix()

    summary = dataset_summary(data)
    print("Dataset summary")
    print("===============")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    by_state = state_summary(data)
    by_state["abbr"] = by_state["state"].map(STATE_NAMES)
    by_state = by_state[["state", "abbr", "counties", "obs", "mean", "std"]]
    by_state.to_csv(TABLE_DIR / "dataset_summary_by_state.csv", index=False)

    print("\nState summary written to:")
    print(TABLE_DIR / "dataset_summary_by_state.csv")
    print("\nState summary")
    print(by_state.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
