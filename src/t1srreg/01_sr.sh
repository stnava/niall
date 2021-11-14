#!/bin/bash

NUMPARAMS=$#

echo TASK ID is $SLURM_ARRAY_TASK_ID
python3 srreg.py $SLURM_ARRAY_TASK_ID
