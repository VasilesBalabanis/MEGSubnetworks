#!/bin/bash
# your institution's slurm account
#SBATCH --account= [your account]
# how many nodes to use. If you are not using job-array, change it with this.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
# 3 days
#SBATCH --time=3-0:00:00
#SBATCH --job-name=FCsubnetworks

# This creates job arrays, so that if you are using a supercomputer, it will use these across multiple nodes. Average run-time for each is 6-12 hours.
#SBATCH --array=0-1
# sends notifications to your email
#SBATCH --mail-user= [your email]
#SBATCH --mail-type=ALL

# depending on your version and environment. My environment is called 'brains'
module load anaconda/2023.09
source activate brains

# First argument is the name of the Simulated Annealing file.
# Second argument is the number of runs for each node to do: 2500, which means 2500 instances of Simulated Annealing.
# Third argument is number of regions in the sub-network of the brain to consider.
# Fourth argument is the name of the numpy file to be generated.
python SimulatedAnnealing.py 2500 10 10regions30seconds_${SLURM_ARRAY_TASK_ID}

