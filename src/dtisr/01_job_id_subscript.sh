#!/bin/bash
echo TASK ID is $SLURM_ARRAY_TASK_ID
python3 ~/coderepo/niall/src/dtisr/02_job_script.py $SLURM_ARRAY_TASK_ID
