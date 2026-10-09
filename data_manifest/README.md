Parent repository data files used by COSMOS-PPI (paths relative to repo root).

| File | Role |
|------|------|
| gavin2006_socioaffinities_rescaled.txt | Gavin PPI |
| 4932.protein.links.detailed.v11.5.txt | STRING PPI |
| GO.txt | GO annotations |
| cache/4932.protein.aliases.v11.5.txt.gz | STRING→locus map |
| data/gold_standards/cyc2008_yeast.csv | Evaluation only |

## High-rank rerun dataset notes

The current TRACE-PPI workspace contains or symlinks the following local datasets:

| Dataset | Required local file(s) | Status |
|---|---|---|
| Gavin yeast | `data/gavin2006_socioaffinities_rescaled.txt` | available |
| STRING yeast | `data/4932.protein.links.detailed.v11.5.txt`, `data/cache/4932.protein.aliases.v11.5.txt.gz` | available via local symlink |
| BioGRID yeast | `data/biogrid_scerivise.tab3.txt` | available via local symlink |
| Krogan yeast | `data/krogan_yeast.tsv` | missing as a separate edge list |
| Human BioPlex/CORUM | `data/human_bioplex.tsv`, `data/gold_standards/corum_human.csv` | missing |

Mapping losses and gold-protein coverage are written to:

```bash
tables/high_rank_rerun/dataset_inventory_mapping_losses.csv
```

No network downloads were performed in the high-rank rerun. For a public submission, replace local symlinks with scripted downloads or archived processed files with checksums, version numbers, license notes, and exact identifier-mapping rules.

## Reproduction command

```bash
python3 scripts/high_rank_rerun.py
```

The result manifest is:

```bash
results/high_rank_rerun/result_manifest.json
```
