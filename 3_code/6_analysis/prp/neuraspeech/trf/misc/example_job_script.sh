#!/bin/bash
#SBATCH --account=pXXXXX			                        ## Your allocation ID
#SBATCH --partition=short			                        ## Partition (buyin, short, normal, etc)
#SBATCH --nodes=1 					                        ## How many computers do you need. Normally just 1.
#SBATCH --ntasks-per-node=24 		                        ## How many cpus or processors do you need on each computer.
#SBATCH --time=04:00:00 			                        ## How long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=1G 			                        ## How much RAM do you need per CPU. Typically 1 GB should be enough. Also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=job_name    	                            ## Optional. When you run squeue -u NETID this is how you can identify the job
#SBATCH --output=/PATH/TO/YOUR/PROJECT/FOLDER/outlog        ## Standard out and standard error goes to this file
#SBATCH --mail-type=ALL 						            ## You can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=YOUR_EMAIL@ADDRESS.edu 		            ## Your email

# Reset loaded modules
module purge all

# Initialize the Conda installation that contains your neuraspeech env
# If you follow the instructions on the NeuraSpeech page, you'll likely have this Conda installation in your NETID folder under home/
# Replace the YOUR_NETID part
source /home/YOUR_NETID/miniforge3/etc/profile.d/conda.sh

# Activate your neuraspeech environment here
source activate neuraspeech

# Navigate to your project folder, which should contain the neuraspeech folder. For example, if you have the following folder structure: 
# /PATH/TO/YOUR/PROJECT/FOLDER
#     - neuraspeech
cd /PATH/TO/YOUR/PROJECT/FOLDER

# Run the boosting function to estimate TRFs. Assume that:
# 1. You've followed previous steps to generate the predictors and prepare EEG data.
# 2. You'll run the script using the default configuration file in neuraspeech/trf/conf/conf.yaml. If you want to
# use a different configuration file, do:
#    python -m neuraspeech.trf.run_boosting -c /FULL/PATH/TO/YOUR_CONFIG.yaml
python --version
python -m neuraspeech.trf.run_boosting
