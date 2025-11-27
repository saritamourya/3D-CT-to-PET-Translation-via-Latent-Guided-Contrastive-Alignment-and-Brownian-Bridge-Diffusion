#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577 -p alvis
#SBATCH -N 1 --gpus-per-node=V100:1
#SBATCH -t 0-48:00:00
# Output files
#SBATCH --error=./error/job_%J.err
#SBATCH --output=./output/out_%J.out
# Mail me1SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1992720@campus.udg.edu

# Load modules
module purge
module load PyTorch-bundle/1.12.1-foss-2022a-CUDA-11.7.0
module load scikit-image/0.19.3-foss-2022a
module load scikit-learn/1.1.2-foss-2022a


# Activate venv
cd /mimer/NOBACKUP/groups/naiss2023-6-336/sarita/ct2pet_ven
source bin/activate

# Executes the code 
cd /mimer/NOBACKUP/groups/naiss2023-6-336/sarita/

# Train HERE YOU RUN YOUR PROGRAM # 565

python3 ./train.py --dataroot /mimer/NOBACKUP/groups/snic2022-5-277/rrestivo/Data/nifti_images

deactivate
