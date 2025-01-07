#!/bin/sh
#SBATCH -J Equilibrium
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dbrewster@g.harvard.edu
#SBATCH --mem-per-cpu=10G
#SBATCH -t 0-72:00:00
#SBATCH --ntasks=100
#SBATCH --output=./slurm/%j.out

# module load cmake/3.28.3-fasrc01
# cmake -Wno-dev -DCMAKE_BUILD_TYPE=Release
# cmake --build cmake-build-release --target run -- -j 8

OMP_NUM_THREADS=100 \
  ./cmake-build-release/apps/cmake-build-debug/run \
  --N=100 \
  --birth-mutation-rate=0.01 \
  --independent-mutation-rate=0.0 \
  --num-steps=10000000 \
  --graph-name=complete \
  --num-simulations=10000
