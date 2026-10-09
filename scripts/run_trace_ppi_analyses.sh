#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MPLCONFIGDIR="${ROOT}/results/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR" "$ROOT/results/revision" "$ROOT/tables/revision" "$ROOT/figures/revision"
cd "$ROOT"

python3 scripts/plot_revision_pipeline_figure.py
python3 scripts/verify_go_leave_one_out.py
python3 scripts/regenerate_case_study.py
python3 scripts/profile_end_to_end.py
python3 scripts/sensitivity_oat.py
python3 scripts/revision_confidence_validation.py
python3 scripts/revision_label_threshold_sensitivity.py
python3 scripts/revision_refinement_impact.py
python3 scripts/revision_tuned_seed_analysis.py
python3 scripts/revision_multimembership_analysis.py

echo "TRACE-PPI analyses complete."
