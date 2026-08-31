# USAPA Corn Yield Lead-Time Benchmark

Clean, script-based reproduction code for the paper's county-level US corn yield
forecasting benchmark.

This folder is intended to replace the exploratory notebooks as the public,
reproducible code path. The original notebooks are retained only as provenance.

## Design goals

- One clear input matrix: `USA_data/benchmark/benchmark_matrix.csv`.
- No notebook-only logic needed to reproduce paper tables and figures.
- No target leakage from same-year yield or future-year trend features.
- Shared cross-validation code for all experiments.
- Tables, figures, and logs written under `reproducible_benchmark/outputs/`
  (`tables/`, `figures/`, `logs/`).

## Main commands

```bash
python -m reproducible_benchmark.scripts.validate_inputs
python -m reproducible_benchmark.scripts.run_paper_tables
python -m reproducible_benchmark.scripts.run_leadtime
python -m reproducible_benchmark.scripts.run_leadtime_paper
python -m reproducible_benchmark.scripts.run_significance_tests
python -m reproducible_benchmark.scripts.write_results_latex
python -m reproducible_benchmark.scripts.audit_no_leakage
```

- `run_significance_tests.py` runs paired Wilcoxon/t-tests between
  HistGradientBoosting and the other models using the per-fold results from
  `run_paper_tables.py`.
- `write_results_latex.py` turns the model-comparison and ablation CSVs into
  paste-ready `.tex` table snippets.
- `audit_no_leakage.py` re-checks every nested feature-selection output and
  fails loudly if a forbidden or future-dated column was ever selected.

## Lead-time analysis

`run_leadtime.py` is the compact clean reproduction path for lead-time tables.
`run_leadtime_paper.py` is the full paper version migrated from the original
`leadtime_analysis.py`; it writes the main lead-time table, NASS-gap table,
per-fold results, LaTeX table, and five paper figures.

## Important methodological correction

The exploratory code used `yield_trend` computed from the full county record as a
feature. That is useful for descriptive decomposition, but it can leak future
information in an operational forecast. The clean benchmark therefore creates
`yield_trend_past`, estimated only from years before the prediction year within
each county. All main forecast experiments use this past-only trend feature.

The prediction target remains raw NASS county yield in bushels per acre
(`yield_bu_acre`). Yield anomaly is retained for interpretation and diagnostics,
not as the target of the main benchmark.

## Note on `agronomic_insights.py`

`agronomic_insights.py` is a descriptive helper used to mine county-year
relationships for the paper discussion. It is not part of the main table/figure
reproduction path, but it can be run directly:

```bash
python -m reproducible_benchmark.agronomic_insights
```

It optionally reads Census county geometry from `geo/` to attach county names
(see the root README for the download commands); without `geo/` it still runs
and just leaves the county-name column empty.
