# niall

notes on interaction and use of parallel cluster


## environment variables

in your head node `.bashrc` file :

```sh
export TF_NUM_INTEROP_THREADS=24
export TF_NUM_INTRAOP_THREADS=24
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=24
```

these control default number of threads but these can also be set within your
job script.  the values should be coordinated across the tasks.

## how to set the task cpus with respect to RAM and compute resource

the number of threads should be adjusted for what you are doing.

suppose we have an instance with 96 cores per node and around 800GB of
RAM.  then if you job takes 80GB of RAM (max), you might be able to squeeze
8 jobs in parallel per node.  That would mean each job would be 96/8=12 cores
per task.   this is relevant in the call to `sbatch` specifically ` --cpus-per-task`.
see the `src/00*` script.

ideally you would maximize the use of CPUs and RAM for each task.

## useful commands

* activate the virtual environment for your pcluster install :

```sh
source apc-ve/bin/activate
```

* login

```sh
pcluster ssh --cluster-name clustername -i ~/.aws/my.pem
```

* quick list of commands [https://srcc.stanford.edu/sge-slurm-conversion](https://srcc.stanford.edu/sge-slurm-conversion)

* run an interactive shell within the parallel cluster (useful for debugging):

```sh
srun --pty bash
```


## initial environment setup

sets up all the ants python tools - do once in your head node account

```sh
pip3 install --upgrade pip # very critical it turns out to get the correct tf
python3 -m pip install tensorflow --user
python3 -m pip install tensorflow_probability --user
python3 -m pip install keras --user

python3 -m pip install dipy --user
python3 -m pip install antspyx --user
python3 -m pip install antspyt1w --user
python3 -m pip install antspymm --user
python3 -m pip install antspynet --user
```

open python on the head node and do:

```python3
import ants
import antspyt1w
import antspymm
antspyt1w.get_data()
antspymm.get_data()
```

then you should be ready to run tasks in parallel.  

see the `src` directory for an example run (not a fast one).


## copying data to a local directory ...

there is a way to do it with rsync and ssh with a .pem file and
knowing the public ip address of the head node.   FIXME.

get the public ip addresss

```sh
pcluster describe-cluster --cluster-name bba
```

now you can rsync

```sh
rsync -av --progress -e 'ssh -i pathto.pem' ubuntu@00.00.00.00:/tmp/z*nii.gz /tmp
```

note: replace the 00.00.00.00 with the real IP address. same for the pem file.
