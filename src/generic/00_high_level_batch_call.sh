#!/bin/bash

NUMPARAMS=$#

if [ $NUMPARAMS -lt 4 ]
then
echo " USAGE ::  "
echo "  sh   $0 cpusPerTask numberOfJobs  imageType subscriptName  "
echo "subscriptName should point to src/generic/01_job_id_subscript.sh"
exit
fi
cpusPerTask=$1
n=$2
imageType=$3
subscriptName=$4
echo "RUNNING: sbatch  --export=ALL --cpus-per-task $cpusPerTask  -o ~/slurmout/${imageType}.%a.out    --array=0-${n}  $4 "
sbatch  --export=ALL --cpus-per-task $cpusPerTask  -o ~/slurmout/${imageType}.%a.out    --array=0-${n}  $4
