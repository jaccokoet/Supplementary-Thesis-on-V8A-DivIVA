# DivIVA V8A Analysis Scripts

## Pipeline overview
1. **AlphaFold** — structural models of WT and V8A DivIVA (monomeric and 
   dimeric) were generated using AlphaFold.
2. **CHARMM-GUI** — AlphaFold-derived structures were used to build 
   protein-membrane systems (CHARMM36m force field, TIP3P water model) 
   using CHARMM-GUI Membrane Builder.
3. **GROMACS simulations** — energy minimization, equilibration, and 
   production MD were run on the ALICE HPC cluster using the Slurm 
   batch scripts in this repository.
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

## Requirements
Python 3.11, pandas, matplotlib, seaborn, scipy, statsmodels, openpyxl
