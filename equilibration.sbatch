#!/bin/bash
#SBATCH --job-name=charmm_eq_repl_1
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=168:00:00

module load GROMACS

set -e

echo "Starting CHARMM-GUI GROMACS equilibration"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

########################################
# STEP 6.0 MINIMIZATION
########################################

gmx grompp \
    -f step6.0_minimization.mdp \
    -c step5_input.gro \
    -r step5_input.gro \
    -p topol.top \
    -n index.ndx \
    -o step6.0_minimization.tpr

gmx mdrun \
    -deffnm step6.0_minimization \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK

########################################
# STEP 6.1 EQUILIBRATION
########################################

gmx grompp \
    -f step6.1_equilibration.mdp \
    -c step6.0_minimization.gro \
    -r step6.0_minimization.gro \
    -p topol.top \
    -n index.ndx \
    -o step6.1_equilibration.tpr

gmx mdrun \
    -deffnm step6.1_equilibration \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu

########################################
# STEP 6.2 EQUILIBRATION
########################################

gmx grompp \
    -f step6.2_equilibration.mdp \
    -c step6.1_equilibration.gro \
    -r step6.1_equilibration.gro \
    -t step6.1_equilibration.cpt \
    -p topol.top \
    -n index.ndx \
    -o step6.2_equilibration.tpr

gmx mdrun \
    -deffnm step6.2_equilibration \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu

########################################
# STEP 6.3 EQUILIBRATION
########################################

gmx grompp \
    -f step6.3_equilibration.mdp \
    -c step6.2_equilibration.gro \
    -r step6.2_equilibration.gro \
    -t step6.2_equilibration.cpt \
    -p topol.top \
    -n index.ndx \
    -o step6.3_equilibration.tpr

gmx mdrun \
    -deffnm step6.3_equilibration \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu

########################################
# STEP 6.4 EQUILIBRATION
########################################

gmx grompp \
    -f step6.4_equilibration.mdp \
    -c step6.3_equilibration.gro \
    -r step6.3_equilibration.gro \
    -t step6.3_equilibration.cpt \
    -p topol.top \
    -n index.ndx \
    -o step6.4_equilibration.tpr

gmx mdrun \
    -deffnm step6.4_equilibration \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu

########################################
# STEP 6.5 EQUILIBRATION
########################################

gmx grompp \
    -f step6.5_equilibration.mdp \
    -c step6.4_equilibration.gro \
    -r step6.4_equilibration.gro \
    -t step6.4_equilibration.cpt \
    -p topol.top \
    -n index.ndx \
    -o step6.5_equilibration.tpr

gmx mdrun \
    -deffnm step6.5_equilibration \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu

########################################
# STEP 6.6 EQUILIBRATION
########################################

gmx grompp \
    -f step6.6_equilibration.mdp \
    -c step6.5_equilibration.gro \
    -r step6.5_equilibration.gro \
    -t step6.5_equilibration.cpt \
    -p topol.top \
    -n index.ndx \
    -o step6.6_equilibration.tpr

gmx mdrun \
    -deffnm step6.6_equilibration \
    -ntmpi 1 \
    -ntomp $SLURM_CPUS_PER_TASK \
    -nb gpu

echo "Equilibration completed successfully"
echo "Finished at: $(date)"
