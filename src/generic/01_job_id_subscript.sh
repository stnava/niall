#!/bin/bash

NUMPARAMS=$#

echo TASK ID is $SLURM_ARRAY_TASK_ID
# python3 scriptname.py $SLURM_ARRAY_TASK_ID
# Rscript scriptname.R $SLURM_ARRAY_TASK_ID
