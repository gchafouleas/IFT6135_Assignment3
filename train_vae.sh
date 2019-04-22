#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=4000MB
#SBATCH --time=0-03:00
#SBATCH --account=def-jiayuan

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt

python vae_main_3.py --eval_mode=Train
