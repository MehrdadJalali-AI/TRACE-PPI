#!/usr/bin/env bash
# Download the public human inputs and Complex Portal files, then verify checksums.
# The yeast PPI and GO files of the original study must be placed manually (see README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/human data/raw/complexportal
curl -fsSL -o data/raw/human/BioPlex_293T_Network_10K_Dec_2019.tsv \
  "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv"
curl -fsSL -o data/raw/human/goa_human.gaf.gz "https://current.geneontology.org/annotations/goa_human.gaf.gz"
curl -fsSL -o data/raw/complexportal/9606.tsv \
  "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv"
curl -fsSL -o data/raw/complexportal/559292.tsv \
  "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/559292.tsv"
date -u +"%Y-%m-%dT%H:%M:%SZ" > data/raw/download_date_utc.txt
shasum -a 256 -c data_manifest/SHA256SUMS || echo "WARNING: checksum differences (newer releases); results may differ."
