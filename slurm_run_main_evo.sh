#!/bin/bash
#sbatch --job-name="hsnn"
#sbatch -p milanq #ipuq #milanq #armq #fpgaq #milanq # partition (queue)
#sbatch -n 1 # number of nodes
#sbatch --ntasks=1
#sbatch --cpus-per-task=64
##sbatch --mem-per-cpu=1gb
#sbatch --time=0-03:00:00
#sbatch -o /home/daniekru/slurm.%j.%n.out # stdout
#sbatch -e /home/daniekru/slurm.%j.%n.err # stderr


. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

cd ~/lab/minBandit
echo "$(pwd)"

srun python3 evo_main.py
echo "finished"


