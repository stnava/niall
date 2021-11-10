#!/bin/bash
echo TASK ID is $SLURM_ARRAY_TASK_ID
python3 /mnt/cluster/src/dti/batch_sr_and_recon.py $SLURM_ARRAY_TASK_ID
