#!/bin/bash -l
# Batch script to run an MPI parallel job with the upgraded software
# stack under SGE with Intel MPI.
# 1. Force bash as the executing shell.
#$ -S /bin/bash
# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=16:00:0
# 3. Request 1 gigabyte of RAM per process (must be an integer)
#$ -l mem=8G
# 4. Request 15 gigabyte of TMPDIR space per node (default is 10 GB)
#$ -l tmpfs=15G
# 5. Set the name of the job.
#$ -N batch_slow_fusion_mpi_job
# 6. Select the MPI parallel environment and 16 processes.
#$ -pe mpi 8
# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :
#$ -wd /home/zceef06/Scratch/output
# 8. loading required python models
module load python3/recommended
# 9. Run our MPI job.  GERun is a wrapper that launches MPI jobs on our clusters.
python /home/zceef06/msc-project/models/slowFusionBatch.py
