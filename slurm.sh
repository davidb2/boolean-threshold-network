#!/bin/sh
#SBATCH -J BooleanNetwork-run-extraction
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dbrewster@g.harvard.edu
#SBATCH --mem-per-cpu=10G
#SBATCH -t 1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=./slurm/%j.out

cargo run-extraction
# cargo run-release
# python3.11 -m pip install matplotlib networkx numpy pandas seaborn tqdm scikit-learn
# python3.11 ./scripts/orthogonal-degree-node-selection-phase-transition.py
