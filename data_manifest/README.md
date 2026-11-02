# Input data manifest

Third-party inputs are not redistributed (except the frozen yeast reference, which is
CC0). `data_manifest/download_data.sh` downloads the public files; `SHA256SUMS` lists
the exact files used for the revision (verified or downloaded 2026-11-02 UTC).
Releases change over time; if a checksum differs, results may differ.

| Local path | Content | Source | Licence |
|---|---|---|---|
| `data/raw/gavin2006_socioaffinities_rescaled.txt` | Gavin et al. 2006 socio-affinity network (rescaled) | author-supplied file from the Gavin 2006 socio-affinity release | publication supplement |
| `data/raw/biogrid_scerivise.tab3.txt` | BioGRID TAB3, *S. cerevisiae*; filtered to PubMed 16554755 (Krogan 2006), physical, yeast–yeast, systematic ORFs | thebiogrid.org (release used in the submitted study) | MIT |
| `data/raw/GO.txt` | SGD GO annotations (GAF 2.x, dated 2020-04-06) | yeastgenome.org (file used in the submitted study) | CC BY 4.0 |
| `data/raw/human/BioPlex_293T_Network_10K_Dec_2019.tsv` | BioPlex 3.0 HEK293T interaction network | https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv | BioPlex terms of use |
| `data/raw/human/goa_human.gaf.gz` | GOA human GAF (generated 2026-05-21) | https://current.geneontology.org/annotations/goa_human.gaf.gz | CC BY 4.0 |
| `data/raw/complexportal/9606.tsv` | EBI Complex Portal, *H. sapiens* (Last-Modified 2026-10-17) | https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv | CC0 |
| `data/raw/complexportal/559292.tsv` | EBI Complex Portal, *S. cerevisiae* (names and literature for case studies only) | https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/559292.tsv | CC0 |
| `data/gold_standards/complexportal_yeast_610.csv` | Frozen yeast reference used in the submitted and revised evaluation (610 Complex Portal complexes) | derived from Complex Portal; identical to `cyc2008_yeast.csv` | CC0 |
| `data/gold_standards/cyc2008_yeast.csv` | Same file under its original, incorrect name (kept for the original code path) | — | CC0 |

## Correction: the yeast reference is Complex Portal, not CYC2008

`cyc2008_yeast.csv` contains 610 complexes whose identifiers are Complex Portal
accessions (CPX-…); all 610 are present in the 2026-11-02 Complex Portal yeast release.
The submitted manuscript described it as CYC2008. The file is unchanged; only its
description was corrected.

## Leakage control

Before module construction, `trace_ppi.datasets.parse_gaf` removes GO annotations
assigned by `ComplexPortal` and annotations citing a `CPX-` identifier (in the reference,
with/from or annotation-extension columns), and annotations with a NOT qualifier. Counts per dataset are
recorded in `results/revised/<dataset>/run_log.json`.

## Identifier mapping (BioPlex)

Nodes are Entrez Gene IDs (GeneA/GeneB). Complex Portal members (UniProt, isoforms
collapsed, nested CPX complexes expanded, CHEBI/RNAcentral components removed) are
mapped to Entrez through the UniProt columns of BioPlex; unmapped members remain in the
reference as `uniprot:ACC` so that recall is measured against complete complexes.
