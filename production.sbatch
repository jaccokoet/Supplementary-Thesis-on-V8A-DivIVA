#!/bin/bash
#SBATCH --job-name=charmm_prod_repl_1
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=gpu-2080ti-11g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

module load GROMACS

set -e

echo "Starting CHARMM-GUI GROMACS production"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cnt=1
cntmax=10

while [ $cnt -le $cntmax ]; do

    pcnt=$((cnt - 1))

    istep="step7_${cnt}"
    pstep="step7_${pcnt}"

    echo "Running production step ${cnt}"

    if [ $cnt -eq 1 ]; then
        pstep="step6.6_equilibration"
    fi

    gmx grompp \
        -f step7_production.mdp \
        -c ${pstep}.gro \
        -t ${pstep}.cpt \
        -p topol.top \
        -n index.ndx \
        -o ${istep}.tpr

    gmx mdrun \
        -deffnm ${istep} \
        -ntmpi 1 \
        -ntomp $SLURM_CPUS_PER_TASK \
        -nb gpu

    cnt=$((cnt + 1))

done

echo "Production completed successfully"
echo "Finished at: $(date)"
