# DivIVA V8A Analysis Scripts

Analysis pipeline and scripts used to investigate the structural and 
membrane-interaction effects of the V8A mutation in the N-terminal 
membrane-binding domain of DivIVA, using AlphaFold structural models 
and GROMACS molecular dynamics simulations.

## Pipeline overview

1. **AlphaFold** — structural models of WT and V8A DivIVA (monomeric 
   and dimeric) were generated using AlphaFold.
2. **CHARMM-GUI** — AlphaFold-derived structures were used to build 
   protein-membrane systems (CHARMM36m force field, TIP3P water model) 
   using CHARMM-GUI Membrane Builder.
3. **GROMACS simulations** — energy minimization, equilibration, and 
   production MD were run on the ALICE HPC cluster using the Slurm 
   batch scripts in this repository (energy minimization is included 
   as the first step in `equilibration.sbatch`).
4. **Trajectory processing** — production trajectories were converted 
   to .xvg output using GROMACS tools.
5. **Residue selection preparation** — `3_index_prep_helix_chain.py`, 
   `3_index_prep_membrane.py`
6. **Structural stability analysis** — `backbone_RMSD.py`, 
   `backbone_RMSF.py`, `radius_gyration.py`, `SASA.py`
7. **Membrane interaction analysis** — `contacts_membrane.py`, 
   `distance_membrane.py`, `gangle.py`
8. **Helix-coiled-coil interaction analysis** — `contacts_protein.py`, 
   `distance_protein.py`

## Repository contents

| File | Description |
|---|---|
| `equilibration.sbatch` | Slurm batch script for energy minimization and equilibration steps |
| `production.sbatch` | Slurm batch script for production MD |
| `3_index_prep_helix_chain.py` | Residue selection for helix-chain comparisons |
| `3_index_prep_membrane.py` | Residue selection for membrane-associated systems |
| `backbone_RMSD.py` | Backbone RMSD analysis |
| `backbone_RMSF.py` | Backbone RMSF (per-residue flexibility) analysis |
| `radius_gyration.py` | Radius of gyration analysis |
| `SASA.py` | Solvent-accessible surface area analysis |
| `contacts_membrane.py` | Protein-lipid contact analysis |
| `distance_membrane.py` | Protein-membrane z-distance analysis |
| `gangle.py` | Protein tilt angle relative to membrane |
| `contacts_protein.py` | Helix-coiled-coil contact analysis |
| `distance_protein.py` | Helix-coiled-coil center-of-mass distance analysis |
| `main.py` | Main script for running the analysis pipeline |

## Requirements

Python 3.11, pandas, matplotlib, seaborn, scipy, statsmodels, openpyxl
