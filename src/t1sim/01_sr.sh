#!/bin/bash

NUMPARAMS=$#

echo TASK ID is $SLURM_ARRAY_TASK_ID
# python3 t1sim.py $SLURM_ARRAY_TASK_ID
python3 t1toNpy.py $SLURM_ARRAY_TASK_ID

