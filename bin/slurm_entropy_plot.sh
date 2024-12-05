#!/bin/bash
#SBATCH --job-name="mkb"
#SBATCH -p milanq #ipuq #milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=1-10:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR

. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

cd ~/lab/minBandit/src
echo "$(pwd)"

srun python3 entropy_run.py --reps 128 --cores 64

echo "finished"


