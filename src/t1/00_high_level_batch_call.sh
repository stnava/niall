export SLURM_CONF=/opt/slurm/etc/slurm.conf
for offset in 0 1000 2000 ; do
  sbatch  --export=ALL --cpus-per-task $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS  -o ~/slurmout/PPMI_${offset}x.%a.out  \
    --array=0-1000 ./01_job_id_subscript.sh $offset
done
