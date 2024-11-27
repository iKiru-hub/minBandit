#!/bin/bash #SBATCH --job-name="mkbwb"
#SBATCH -p ipuq #milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-10:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR

. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "%ecl1 activated"

cd ~/lab/minBandit/src
echo "%in $(pwd)"

srun large_run.py

git add .
git commit -m "large run from ex3"
git push

echo " "
echo "%finished%"


