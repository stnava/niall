sbatch  --export=ALL --cpus-per-task 24  -o ~/slurmout/dtrecon.%a.out  \
  --array=0-82  ~/coderepo/niall/src/dtirecon/01_job_id_subscript.sh
