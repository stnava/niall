sbatch  --export=ALL --cpus-per-task 8  -o ~/slurmout/rsfmri.%a.out  \
  --array=1-281  ~/coderepo/niall/src/rsfmri/01_job_id_subscript.sh
