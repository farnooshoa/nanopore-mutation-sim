# Nanopore Mutation Simulator (Structure-aware) — v0

This repo is the start of a structure-aware framework to study how amino-acid mutations in a protein nanopore
change transport-relevant geometry (and later, electrostatics).

## What works now (v0)
- Downloads a PDB structure (example: 7AHL)
- Computes a rough radius profile along an inferred pore axis (geometry-only baseline)
- Saves:
  - `outputs/7AHL_radius_profile.csv`
  - `outputs/7AHL_radius_profile.png`

## Run
```bash
python src/download_pdb.py 7AHL
python src/radius_profile.py data/pdb/7AHL.pdb --out_prefix outputs/7AHL
