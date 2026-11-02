# TRACE-PPI

**TRACE-PPI: Tiered Record of Assignment-level Channel Evidence** is the code for the
manuscript *"Traceable, Evidence-Tiered Assignments for Overlapping Protein Module
Detection in Protein–Protein Interaction Networks"*, submitted to
*Intelligent Data Technologies*.

TRACE-PPI is an assignment-auditing framework. It does not claim higher module recovery
than established detectors (it does not reach ClusterONE's module F1 on Gavin); its
contribution is that every protein–module membership is exported with separate
topology, semantic and GO-overlap scores, an evidence tier and a provenance record.

## Method in brief

1. **Localisation** – strict MCL (unweighted, inflation 2.0, no fallback) gives coarse regions.
2. **Heterogeneity diagnostics** – topology separation, semantic dispersion and support
   variation; a region of ≥6 proteins is flagged when ≥2 signals exceed the upper quartile.
3. **Decomposition** – a flagged region is kept, split by seed-and-expand, or split
   spectrally (Fiedler vector), choosing whichever has the best internal quality
   score. Reference complexes are never used.
4. **Adaptive assignment** – S = 0.45 τ + 0.35 σ + 0.20 g, with a per-candidate
   threshold at the 25th percentile of incumbent leave-one-out scores and an absolute
   guard (≥1 neighbour and max(τ, σ) ≥ 0.12). Annotation-only matches become *review
   suggestions*, never memberships.
5. **Bounded supplementation**, final channel recomputation, and
   tiers: Core (τ ≥ 0.35 and σ ≥ 0.25), Inner (S ≥ 0.25), Outer (S ≥ 0.12).
6. **Stability** under edge removal is reported as a diagnostic. It does not filter
   memberships, because the pre-declared gate rule was not met.

σ and g are both derived from GO annotations, so they are related annotation channels,
not independent biological validation. All parameters are listed in
[`configs/trace_ppi_frozen.yaml`](configs/trace_ppi_frozen.yaml), committed before any
reference-based evaluation (commit `6eeab3f`).

## Installation

```bash
python3 -m pip install -r requirements.txt   # exact versions used for the paper
python3 -m pip install -e .                  # provides the `trace-ppi` command
```

ClusterONE 1.0 (`tools/clusterone/cluster_one-1.0.jar`) needs Java ≥ 8.

## Data

See [`data_manifest/README.md`](data_manifest/README.md). The human inputs are
fetched with `bash data_manifest/download_data.sh`; the yeast PPI and GO files of the
yeast study must be placed in `data/raw/`. Checksums: `data_manifest/SHA256SUMS`.

## Reproduce the revision

```bash
export PYTHONPATH="$(pwd)" PYTHONHASHSEED=0 MPLCONFIGDIR=/tmp/mpl
# TRACE-PPI, baselines, ablations, bootstrap, stability, sensitivity (per dataset)
python3 scripts/run_experiments.py --dataset gavin
python3 scripts/run_experiments.py --dataset krogan
python3 scripts/run_experiments.py --dataset bioplex
# Runtime and memory (3 fresh runs per dataset)
python3 scripts/measure_runtime.py
# Rule-selected case studies
python3 scripts/case_studies.py
# Manuscript tables, figures and numeric macros
python3 scripts/make_manuscript_assets.py --out <manuscript_dir>/generated
python3 figures/src/make_fig1.py
```

Module construction alone (does not read reference complexes):

```bash
trace-ppi run --dataset gavin --out results/run_gavin
```

Outputs: `modules.csv`, `ledger.csv` (accepted memberships, review suggestions and
rejections, with τ, σ, g, S, threshold, guard status, tier and provenance),
`regions.csv`, `split_audit.csv` and `provenance.json`.

## Repository layout

| Path | Content |
|---|---|
| `trace_ppi/` | TRACE-PPI implementation, evaluation and baseline wrappers |
| `configs/trace_ppi_frozen.yaml` | frozen configuration |
| `scripts/` | experiment, runtime, case-study and manuscript-asset scripts |
| `results/revised/<dataset>/` | results (benchmark, bootstrap CIs, ledger, overmerging, stability, sensitivity) |
| `figures/` | Figure 1 (workflow) and its source (`figures/src/make_fig1.py`) |
| `logs/` | run logs |
| `tools/clusterone/` | ClusterONE 1.0 JAR (Nepusz et al. 2012) |

## Citation

If you use TRACE-PPI, please cite the manuscript (citation details will be added on
publication).
