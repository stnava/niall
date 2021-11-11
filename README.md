# niall

for processing all neuroimages in parallel computing environments

notes on interaction with parallel cluster and usage with HPC for the ANTsX ecosystem

## how to read this repository

in the `src` directory, we put the organizing scripts for the calls to
each sub-process for each modality.  it's not fully generalized but provides a
framework that should allow minimal modifications for other projects.

in each `src/modality` directory, we have something like this:

* an overall call to `sbatch` typically prefixed by `00`

* a call to a subscript that passes the index to the sub-process, typically prefixed by `01`

* the actual processing script typically prefixed by `02`

this is fairly general for applications that are deployed on a HPC system.

the `02` script will need the most modification for each process taking care of:

* the nature/directory structure of the input

* the nature/directory structure of the output

* the specific tasks that need to be run

* across `00` `01` and `02`, the threads/tasks should be coordinated (if you
  care about efficient compute usage).

* the `02` script should be run either in an interactive session (on a node)
before being deployed at scale through `sbatch`.  this is to make sure it's
actually working before running lots of computation through.

Please open issues in the repository for any questions.

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


## things having to do with `R` and `C++`

* cmake

```sh
sudo apt remove --purge cmake
sudo snap install cmake --classic
```

* R (updated version) [https://cran.rstudio.com/bin/linux/ubuntu/](https://cran.rstudio.com/bin/linux/ubuntu/)
