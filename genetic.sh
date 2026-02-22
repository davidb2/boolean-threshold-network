#!/bin/sh
#SBATCH -J BooleanNetwork-genetic-1.8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dbrewster@g.harvard.edu
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=400MB
#SBATCH --output=./slurm/%j.out


# python3.11 -m pip install matplotlib networkx numpy pandas seaborn tqdm scikit-learn
# --states-file "data/drug-v0129d-N5k-gamma1.6-10drugs/derived/states-1769778787192.csv" \
source david-brewster-boolean-network-env/bin/activate
echo "host=$(hostname) procid=$SLURM_PROCID"
python3.11 ./scripts/genetic-algorithm-selection.py \
  --original-network-idx 0 \
  --states-file "data/drug-v0129d-N5k-gamma1.8-10drugs/derived/states-1769778862662.csv" \
  --output-dir "data/random-forests/retention-vs-accuracy-v0129d-N5k-gamma1.8-10drugs-genetic-shard"

