#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${ROOT}/results/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"
python3 scripts/build_evidence_profiles.py
python3 scripts/build_semantic_embeddings.py
# Refresh BH-dependent caches so coefficient changes affect candidates/scores.
rm -f results/cores/black_hole_cores_gavin.csv
rm -f results/candidates/candidate_modules_gavin.csv
rm -f results/candidates/candidate_scores_gavin.csv
rm -f results/cores/black_hole_cores_krogan.csv
rm -f results/candidates/candidate_modules_krogan.csv
rm -f results/candidates/candidate_scores_krogan.csv
python3 scripts/evaluate_trace_ppi_final.py
python3 scripts/evaluate_krogan_benchmark.py
python3 scripts/plot_trace_ppi_figures.py
python3 scripts/plot_cross_dataset_figures.py
if [[ -f manuscript/trace_ppi_main.tex ]]; then
  (cd manuscript && pdflatex -interaction=nonstopmode trace_ppi_main.tex >/dev/null && \
   bibtex trace_ppi_main >/dev/null && \
   pdflatex -interaction=nonstopmode trace_ppi_main.tex >/dev/null && \
   pdflatex -interaction=nonstopmode trace_ppi_main.tex >/dev/null)
fi
echo "TRACE-PPI pipeline complete."
