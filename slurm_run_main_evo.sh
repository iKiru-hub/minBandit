#!/bin/bash
#SBATCH --job-name="mkb"
#SBATCH -p ipuq #milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=1-10:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR

. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

cd ~/lab/minBandit
echo "$(pwd)"

#srun python3 evo_main.py --verbose
# srun python3 src/make_plot.py --run "smooth" --save
#srun python3 main.py --verbose --trials 600 --rounds 1 --K 10 --reps 10 --plot --env "smooth" --load --save --idx 4 --multiple 2
srun python3 main.py --verbose --trials 2 --rounds 6000 --K 800 --reps 20 --plot --env "simple" --load --save --idx 4 --multiple 2

echo "finished"


