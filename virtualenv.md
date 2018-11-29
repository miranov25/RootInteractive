# Install RootInteractive package
* add line to your .bashr to define python path. E.g 
```
export PYTHONPATH=$PYTHONPATH:$HOME/github/RootInteractive
```
* WITHOUT defining path -  classes can not be used

### Install virtualenv:

I used to install virtualenvwrapper:

https://virtualenvwrapper.readthedocs.io/en/latest/

````
source /home/miranov/.local/bin/virtualenvwrapper.sh
mkvirtualenv  our_new_env
workon our_new_env
````

### Install root python dependencies
Following instruction in
https://alice.its.cern.ch/jira/browse/ATO-448

```
pip install pip -U
pip install aliBuild
pip install matplotlib numpy certifi ipython==5.1.0 ipywidgets ipykernel notebook metakernel pyyaml
```


### Install jupyter lab and interactive packages:
Following instructions in:
https://alice.its.cern.ch/jira/browse/PWGPP-485

**all done in virtual environment to do not mess wtih system unusable pip**

```
pip install jupyter jupyterlab

```

````
pip install sklearn
pip install scikit-garden

pip install beakerx pandas ipywidgets
pip install plot runtime
pip install plotly
pip install request
pip install bqplot
pip install qgrid
pip install ipympl
````



### Install lab extension
 jupyter labextension list

* first - node js had to be installed
```
# Using Ubuntu
curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
sudo apt-get install -y nodejs
```

* install extensions for jupyter-lab
````
jupyter labextension install @jupyter-widgets/jupyterlab-manager  #OK
jupyter labextension install bqplot   #OK
jupyter labextension install qgrid@1.0.1-beta.0
jupyter labextension install jupyterlab_bokeh      #OK
jupyter labextension install jupyter-matplotlib
````

* check list
```
jupyter labextension list
JupyterLab v0.33.12
Known labextensions:
   app dir: /home/miranov/.virtualenvs/our_new_env/share/jupyter/lab
@jupyter-widgets/jupyterlab-manager
        @jupyter-widgets/jupyterlab-manager v0.37.0  enabled  OK
bqplot
        bqplot v0.4.3  enabled  OK
jupyter-matplotlib
        jupyter-matplotlib v0.3.0  enabled  OK
jupyterlab_bokeh
        jupyterlab_bokeh v0.6.1  enabled  OK
qgrid
        qgrid v1.0.1-beta.0  enabled  OK

```