sbatch  --export=ALL --cpus-per-task 12  -o ~/slurmout/t1.%a.out  \
  --array=0-143  ~/coderepo/niall/src/t1/01_job_id_subscript.sh
