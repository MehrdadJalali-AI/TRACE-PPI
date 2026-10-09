# TRACE-PPI: evidence-aware overlap deconvolution

TRACE-PPI (Tiered Record of Assignment-level Channel Evidence) is the code for the
manuscript "Traceable, Evidence-Tiered Assignments for Overlapping Protein Module
Detection in Protein–Protein Interaction Networks".

TRACE-PPI treats an initial MCL partition as coarse candidate localisation. It
then detects heterogeneous MCL clusters, deconvolves them into candidate
subcomplexes, assigns overlapping protein memberships with separately exported
topology, semantic and GO support, applies bounded supplementation, and records
support tiers and provenance for every assignment.

## Reproduce the deconvolution revision

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_deconvolution_revision.py --dataset gavin
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_deconvolution_revision.py --dataset krogan
MPLCONFIGDIR="$(pwd)/results/matplotlib_cache" python3 scripts/plot_deconvolution_figures.py
```

The run is deterministic (`random_seed=42`) and saves its graph size, embedding
backend, parameters, runtime and Git revision when one is available in
`results/experiment_log.json`.

## Pipeline

1. MCL creates coarse local search regions.
2. Three empirical heterogeneity criteria identify likely merged regions:
   topology/cut separation, semantic dispersion and node-support variation.
3. Seed-and-expand and local spectral (Fiedler) deconvolution are compared by
   internal evidence quality; no gold complex is used for selection.
4. A protein receives one or more memberships using
   `0.45 topology + 0.35 semantic + 0.20 GO` support. Candidate-specific
   lower-quartile thresholds avoid a single global admission cutoff.
5. Roles (`core`, `boundary`, `shared`) are separated from evidence tiers
   (`high`, `moderate`, `weak`, `uncertain`).
6. Bounded supplementation runs only after structural deconvolution.

## Main outputs

`results/module_metrics.csv` contains module-list recovery, Omega and cover NMI.
`results/membership_metrics.csv` and `results/shared_membership_metrics.csv`
contain direct protein--complex and reference-shared-protein evaluation.
`results/candidate_coverage.csv`, `results/merged_cluster_cases.csv`,
`results/ablation_results.csv`, `results/threshold_sensitivity.csv`, and
`results/synthetic_overlap_test.csv` record candidate-stage diagnostics,
reviewer-targeted stress tests, staged ablations, sensitivity analyses and
controlled overlap tests. Assignment-level provenance is in
`results/final_deconvolution_memberships.csv`.

The same Krogan outputs are retained in `results/krogan_deconvolution/` so they
do not overwrite the Gavin manuscript tables. On Krogan, membership F1 rises
from 0.401 for the previous overlap path to 0.481 and shared-membership F1 from
0.004 to 0.249.

The Gavin run is a sensitivity analysis on the supplied CYC2008 reference. It
does not use reference complexes for split selection, membership admission or
threshold selection; it should not be described as a held-out optimisation.

## Input data

Raw third-party inputs (Gavin, STRING, BioGRID and GO files) are not redistributed.
Download them from their providers and place them under `data/` as listed in
`data_manifest/README.md`. The CYC2008 reference is included in
`data/gold_standards/cyc2008_yeast.csv`. Processed result tables are in `results/`
and `tables/`.
