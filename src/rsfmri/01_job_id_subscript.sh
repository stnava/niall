#!/bin/bash
echo TASK ID is $SLURM_ARRAY_TASK_ID
python3 ~/coderepo/niall/src/rsfmri/02_job_script_snbf.py $SLURM_ARRAY_TASK_ID
# Rscript ~/coderepo/niall/src/rsfmri/02_job_script_dfn.R $SLURM_ARRAY_TASK_ID
