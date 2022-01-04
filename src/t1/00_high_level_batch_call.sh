sbatch  --export=ALL --cpus-per-task $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS  -o ~/slurmout/t1.%a.out  \
  --array=0-143  ~/coderepo/niall/src/t1/01_job_id_subscript.sh
