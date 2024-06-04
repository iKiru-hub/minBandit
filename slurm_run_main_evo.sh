#!/bin/bash
#SBATCH --job-name="mkb"
#SBATCH -p milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-10:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR

. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

cd ~/lab/minBandit
echo "$(pwd)"

srun python3 evo_main.py --verbose
echo "finished"


