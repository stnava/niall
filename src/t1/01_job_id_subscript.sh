#!/bin/bash
echo TASK ID is $SLURM_ARRAY_TASK_ID
echo $1
python3 ~/coderepo/niall/src/t1/02_job_script.py $SLURM_ARRAY_TASK_ID True $1
