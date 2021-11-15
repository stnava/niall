sbatch  --export=ALL --cpus-per-task $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS  -o ~/slurmout/rsfmri%a.out  \
  --array=0-281  ~/coderepo/niall/src/rsfmri/01_job_id_subscript.sh
