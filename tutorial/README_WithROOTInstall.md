# THIS IS always WORK in progress, as dependencies are regularly evolved
The last review of a particular recipe is indicated in the subsections

## Building container on alice server at GSI or on local laptop:
For users working with ALICE TPC and reconstruction software
* assuming that NOTES the directory with the TPC software is
* set variables:
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

## Container at lxplus  is .sif container not working
Looks like supported on voluntary basis

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

## Installing only as virtual environment at lxplus8

Since we want to use the optional root or ALICE software of cvmfs, we should use the appropriate Python version, since the binaries ROOT 
(Python 3.8) and ALICE (Python3.9) depend on this version.  Configure the virtual environment and use the correct Python.  
This should work in principle and was tested on a laptop with Ubuntu 20 and a local O2 installation (which should be installed locally beforehand).

In order to be able to use different versions of the software, we usually create different versions of the compatible 
virtual environment to match root or AliRoot or O2 respectively.  Usually I also try to make virtual environment installation with Python version.

* login
```shell
ssh -X lxplus8.cern.ch
# you might have to use your CERN user name for the login, i.e. ssh -X username@lxplus9.cern.ch
```

* global variables
** NOTES needed only for the ALICE TPC users 
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
# export RootInteractive=/afs/cern.ch/user/m/mivanov/public/github/RootInteractive

export venvPrefix=/path/to/your/virtual/python/environments
# examples:
# export venvPrefix=$EOSuser/venv
# export venvPrefix=/eos/user/e/ehellbar/venv
# export venvPrefix=/eos/user/m/mivanov/www/venv
## export venvPrefix=/afs/cern.ch/user/m/mivanov/public/venv
```

* clone RootInteractive
```shell
git clone https://github.com/miranov25/RootInteractive $RootInteractive
```

### Select prefered Root, O2, AliRoot version from cvmfs

* install and use git version  - to be done once or if the O2/AliRoot version wil change (on top of version O2sim/v20220220-1) or ROOT
```shell
source  /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.26.04/x86_64-ubuntu20-gcc94-opt/bin/thisroot.sh 
eval $(/cvmfs/alice.cern.ch/bin/alienv printenv ROOT/v6-26-10-alice5-4)
or source ALICE code
eval $(/cvmfs/alice.cern.ch/bin/alienv printenv O2sim/v20220220-1)
```

### Install matching virtual environment for the Root resp. ALICE sowfare from the cvmfs
* the code must be installed with the newer Python version > 3.8 - should be the case on this computer
* to ensure consistency with the software of cvmfs, a suitable python version must be used
* To check the version use
   `root-config --python-version`
  * base on that please select the venv version to install
  * newest O2 use Python 3.9.12
  * /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.26.04/x86_64-ubuntu20-gcc94-opt/bin/thisroot.sh  Python 3.9.12
* it is strongly recomened to create virtual enironment explictly with python version and use special names to be able to change  

Example:
````  
python3.9 -m venv ${venvPrefix}/setup39
source ${venvPrefix}/setup39/bin/activate
``````
### Install RootInteractive dependencies
```
python3 -m pip install   -r  $RootInteractive/requirements.txt
python3 -m pip install   -e  $RootInteractive
# install devel packages to be able to test 
pip install -r  $RootInteractive/requirements_Devel.txt
```

### Run from sctrach, initialize environment
```shell
ssh -X lxplus.cern.ch
source ${venvPrefix}/setup39/bin/activate
# you might have to use your CERN user name for the login, i.e. ssh -X username@lxplus8.cern.ch
eval $(/cvmfs/alice.cern.ch/bin/alienv printenv O2sim/v20221111-1)
source ${venvPrefix}/setup39/bin/activate
export PYTHONPATH=$RootInteractive/:$PYTHONPATH
```

### Make tests
* RootInetractive Unit test
* iPython import ROOT test
* Jupyter notebook test

```shell
cd $RootInteractive/RootInteractive/InteractiveDrawing/bokeh/
pytest test_bokehClientHistogram.py 
# or full test pytest test_*.py 
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

## Installing only as virtual environment on local machine

Since we want to use the optional root or ALICE software of cvmfs, we should use the appropriate Python version, since the binaries ROOT 
(Python 3.8) and ALICE (Python3.9) depend on this version.  Configure the virtual environment and use the correct Python.  
This should work in principle and was tested on a laptop with Ubuntu 20 and a local O2 installation (which should be installed locally beforehand).

In order to be able to use different versions of the software, we usually create different versions of the compatible 
virtual environment to match root or AliRoot or O2 respectively.  Usually I also try to make virtual environment installation with Python version.


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
python3 -m venv ${venvPrefix}/setupX
source ${venvPrefix}/setupX/bin/activate
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
source ${venvPrefix}/setup39/bin/activate
export PYTHONPATH=$RootInteractive/:$PYTHONPATH
```

* make test:
```shell
cd $RootInteractive/RootInteractive/InteractiveDrawing/bokeh/
pytest test_*py
cd $RootInteractive/RootInteractive/tutorial/
jupyter notebook
```

## Node js installation at lxplus

NvM  (node version) installation following https://www.freecodecamp.org/news/node-version-manager-nvm-install-guide/
Nodejs at lxplus historical. Version >v14.0.0 needed as in the bokeh requirement
FAILED test_bokehClientHistogram.py::testBokehClientHistogramRowwiseTable - RuntimeError: node.js v14.0.0 or higher
Details could be find in the link above. Bellow is shorter version


### Step1 install nvm
```curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash```

### Step2 - list available version of the node.js
```nvm ls-remote```

### Step 3 install version of the nodejs from list above and make it default
```
nvm install v18.12.0
Downloading and installing node v18.12.0...
Downloading https://nodejs.org/dist/v18.12.0/node-v18.12.0-linux-x64.tar.xz...
####################################################################################################################################################################################################### 100,0%
Computing checksum with sha256sum
Checksums matched!
Now using node v18.12.0 (npm v8.19.2)
Creating default alias: default -> v18.12.0
```
### Step4 - switch between different versions of nodejs if needed
```nvm use vA.B.C```
