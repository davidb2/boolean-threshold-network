#!/usr/bin/env bash
set -ex
# source /n/home04/dbrewster/playground/boolean-threshold-network/david-brewster-boolean-network-env/bin/activate
# salloc -N 1 -c 4 --mem=32G -t 12:00:00
# srun --pty bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
