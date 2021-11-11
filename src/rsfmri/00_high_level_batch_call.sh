sbatch  --export=ALL --cpus-per-task 24  -o ~/slurmout/rsfmri.%a.out  \
  --array=1-143  ~/coderepo/niall/src/rsfmri/01_job_id_subscript.sh
