#!/bin/bash
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=first_CNN
#SBATCH --time=1-12:00:00              #Set the wall clock limit to 1 day 12 hour
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=8G                  #Request 8GB per node
#SBATCH --output=pnn.%j
#SBATCH --gres=gpu:2                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --account=122810042376             #Set billing account to 122810042376
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=pbhatt@tamu.edu    #Send all emails to email_address
ml purge
module load Anaconda
source activate my_keras-gpu-2.2.2
module load FFmpeg
python segment_video.py > terminal_output.txt
