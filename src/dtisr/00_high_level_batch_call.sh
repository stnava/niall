# this shows how to export all env variables, controls cpus per task and defines an output location per job
# the array number is available with the target .sh script as seen below:
# cat echobatchnum.sh
# > #!/bin/bash
# > echo TASK ID is $SLURM_ARRAY_TASK_ID
#
# sbatch  --export=ALL --cpus-per-task 1  -o ./slurmout/output.%a.out  --array=1-10  ./echobatchnum.sh
#
# for batch dti - there are 194 of these images
sbatch  --export=ALL --cpus-per-task 24  -o ./slurmout/dtisr.%a.out  --array=1-194  ./batchdtisr.sh
