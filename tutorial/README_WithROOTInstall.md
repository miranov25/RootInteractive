# THIS IS WORK in progress

## buildin container on alice server at GSI or on local laptop:
* assuming NOTES poits to directory with TPC software
* setting variables:
    ```
    export SINGULARITY_CACHEDIR=/data.local1/${USER}/
    export SINGULARITY_PREFIX=/data.local1/${USER}/JIRA/ATO-500
    export NOTES=/lustre/alice/users/miranov/NOTES/alice-tpc-notes2
    #
    export SINGULARITY_DOCKER_USERNAME=<your docker name>
    export SINGULARITY_DOCKER_PASSWORD=<redacted>
    e.g.
    export SINGULARITY_DOCKER_USERNAME=miranov25
    export SINGULARITY_DOCKER_PASSWORD= ???? - You should be registered to the DOCKER (usernmae/password) if you want to use it
    ```
* build container as sandbox
** keep this option for later updates
```sh
time singularity build --fakeroot --fix-perms --sandbox $SINGULARITY_PREFIX/tensorflow_latest-gpuCUDALight $NOTES/JIRA/ATO-500/machineLearning/tensorflow_latest-gpu_CUDA.def
```
* build container as a sif file
```sh
time singularity build --fakeroot --fix-perms  $SINGULARITY_PREFIX/tensorflow_latest-gpuCUDALight.sif $SINGULARITY_PREFIX/tensorflow_latest-gpuCUDALight
```

* Install and use virual environment in container itself
  * install
  ```sh
  singularity shell --writable --fakeroot -B /lustre -B /cvmfs   $SINGULARITY_PREFIX/tensorflow_latest-gpuCUDALight
  eval $(/cvmfs/alice.cern.ch/bin/alienv printenv O2sim/v20220220-1)
  python3 -m venv /venv/venv3
  source /venv/venv3/bin/activate
  pip install   -r  $RootInteractive/requirements.txt
  pip install  -r   $NOTES/JIRA/ATO-500/machineLearning/requirements.txt
  ```
  * use
  ```sh
  singularity shell -B /u/ -B /lustre -B /cvmfs   $SINGULARITY_PREFIX/tensorflow_latest-gpuCUDALight
  eval $(/cvmfs/alice.cern.ch/bin/alienv printenv O2sim/v20221128-1)
  source /venv/venv3/bin/activate
  cd $RootInteractive
  ```
  * make test:
  ```shell
  export PYTHONPATH=$RootInteractive/:$PYTHONPATH
  cd $RootInteractive/RootInteractive/InteractiveDrawing/bokeh/
  pytest test_ClientSideJoin.py
  jupyter notebook --no-browser --ip=127.0.0.1
  ```

# Container at lxplus  is .sif container not working

bug report submitted https://cern.service-now.com/service-portal?id=ticket&table=incident&n=INC3064146
```asm
 Description:
The Singularity Shell converts the sif file to the sandbox instead of starting. Later it fails.
The sif container was created on the GSI cluster. To create the container at CERn, it is not clear how to get the fakeroot criteria.

To reproduce problem:
[mivanov@lxplus8s07 singularity]$ singularity shell /eos/user/m/mivanov/www/tensorflow_latest-gpuCUDA.sif
INFO: Converting SIF file to temporary sandbox...
FATAL: while extracting /eos/user/m/mivanov/www/tensorflow_latest-gpuCUDA.sif: root filesystem extraction failed: extract command failed: ERROR : Failed to create user namespace: maximum number of user namespaces exceeded, check /proc/sys/user/max_user_namespaces
: exit status 1

As a follow up of the ticket, it was possible to install the contained from the "sandbox" but this is very slow from EOS.
This option was therefore abandomed.


```

# Installing only as virtual environment at lxplus8
* login
```shell
ssh -X lxplus8.cern.ch
# you might have to use your CERN user name for the login, i.e. ssh -X username@lxplus8.cern.ch
```

* global variables
```shell
export NOTES=/eos/user/m/mivanov/www/github/alice-tpc-notes
```

* user defined variables
```shell
export EOSuser=/path/to/your/EOS/directory
# examples:
# export EOSuser=/eos/user/e/ehellbar

export RootInteractive=/path/to/your/RootInteractive/directory
# examples:
# export RootInteractive=$EOSuser/RootInteractive
# export RootInteractive=/eos/user/m/mivanov/www/github/RootInteractive

export venvPrefix=/path/to/your/virtual/python/environments
# examples:
# export venvPrefix=$EOSuser/venv
# export venvPrefix=/eos/user/e/ehellbar/venv
# export venvPrefix=/eos/user/m/mivanov/www/venv
```

* clone RootInteractive
```shell
git clone https://github.com/miranov25/RootInteractive $RootInteractive
```

* install and use git version  - to be done once or if the O2/AliRoot version wil change (on top of version O2sim/v20220220-1) or ROOT
```shell
source  /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.26.04/x86_64-ubuntu20-gcc94-opt/bin/thisroot.sh 
or source ALICE code
eval $(/cvmfs/alice.cern.ch/bin/alienv printenv O2sim/v20220220-1)

python3 -m venv ${venvPrefix}/setup2
source ${venvPrefix}/setup2/bin/activate
# python3 -m pip install   -r  $RootInteractive/requirements.txt
python3 -m pip install   -e  $RootInteractive
# install devel packages to be able to test 
pip install -r  $RootInteractive/requirements_Devel.txt
```

* initialize environment
```shell
ssh -X lxplus8.cern.ch
# you might have to use your CERN user name for the login, i.e. ssh -X username@lxplus8.cern.ch
source  /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.26.04/x86_64-ubuntu20-gcc94-opt/bin/thisroot.sh 
or 
eval $(/cvmfs/alice.cern.ch/bin/alienv printenv O2sim/v20220220-1)
source ${venvPrefix}/setup2/bin/activate
export PYTHONPATH=$RootInteractive/:$PYTHONPATH
```

* make test:
```shell
cd $RootInteractive/RootInteractive/InteractiveDrawing/bokeh/
pytest test_*py
cd $RootInteractive/RootInteractive/tutorial/
jupyter notebook --no-browser --ip=127.0.0.1
```

* open jupyter notebook in the browser on your local laptop
  * Link to copy to your browser is provided by jupyter notebook shell output:
    ```shell
        Or copy and paste one of these URLs:
            http://127.0.0.1:8888/?token=48e7e052b37a01608c03f57fb678d05a0c4ca6e9fe5b28bc
    ```
  * Opening of tunnel required to view the notebook in your local browser on your laptop
    * Remote hostname and port are required
      * \<host\> can be found in the terminal command line on the remote host, e.g. -> **lxplus8s13.cern.ch**:
      ```shell
      (setup2) ehellbar@lxplus8s13:/eos/user/e/ehellbar/RootInteractive/RootInteractive/tutorial$
      ```
      * \<port\> is indicated in the jupyter notebook URL, e.g. -> **8888**
      ```shell
      http://127.0.0.1:8888/?token=48e7e052b37a01608c03f57fb678d05a0c4ca6e9fe5b28bc
      ```
    * open ssh tunnel in terminal on your local laptop / computer
    ```shell
    ssh -N -f -L <port>:localhost:<port> username@<host>
    # examples:
    # ssh -N -f -L 8888:localhost:8888 ehellbar@lxplus8s13.cern.ch
    ```

# Installing only as virtual environment on local machine
This should work in principle and was tested on a laptop with ubuntu 20 and a local O2 installation (to be installed locally before). Errors might still occur due to local package or version incompatibilities which can be individual to every machine or use case.

* user defined variables
```shell
export RootInteractive=/path/to/your/RootInteractive/directory
# examples:
#  export RootInteractive=$HOME/RootInteractive

export venvPrefix=/path/to/your/virtual/python/environments
# examples:
# export venvPrefix=$HOME/venv
```

* clone RootInteractive
```shell
git clone https://github.com/miranov25/RootInteractive $RootInteractive
```

* install and setup virtual environment
```shell
python3 -m venv ${venvPrefix}/setup2
source ${venvPrefix}/setup2/bin/activate
# python3 -m pip install   -r  $RootInteractive/requirements.txt
python3 -m pip install   -e  $RootInteractive
```

* initialize  ROOT or O2/AliPhsycis installation if needed, e.g: 
```shell
  source  /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.26.04/x86_64-ubuntu20-gcc94-opt/bin/thisroot.sh 
  or
   eval $(alienv printenv O2::latest)
```

* initialize virtual environment on top of installation
```shell
source ${venvPrefix}/setup2/bin/activate
export PYTHONPATH=$RootInteractive/:$PYTHONPATH
```

* make test:
```shell
cd $RootInteractive/RootInteractive/InteractiveDrawing/bokeh/
pytest test_*py
cd $RootInteractive/RootInteractive/tutorial/
jupyter notebook
```