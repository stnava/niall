sbatch  --export=ALL --cpus-per-task 24  -o ~/slurmout/flair.%a.out  \
  --array=1-143  ~/coderepo/niall/src/flair/01_job_id_subscript.sh
