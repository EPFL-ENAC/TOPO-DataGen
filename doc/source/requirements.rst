Getting started
====================


Requirements
--------------------

The follwing elements are needed :

* OS: Ubuntu Desktop 18.04 + 
* Nvidia graphics processing unit (GPU) to run CUDA
* `firefox <https://www.mozilla.org/en-US/>`_ browser (headless node might not work)
* A `cesium Ion <https://cesium.com/platform/cesium-ion/>`_ account

- a reliable Internet connection is preferred

Other components are needed, check bellow for a full installation process. 


--------------


Installation guide
--------------------

The below instructions show how to install stable versions of TOPO-DataGen.

Git
**********************
Check if git is installed

.. code-block::

    git --version

If git is not installed

.. code-block::

    sudo apt install git

Docker
**********************
Check if docker is installed :

.. code-block::

    docker --version

If docker is not installed

.. code-block::

    sudo apt-get remove docker docker-engine docker.io
    sudo apt-get update
    sudo apt install docker.io
    sudo snap install docker

.. note::
   Some specifics rights must be added as follow
   
   .. code-block::

    sudo chmod 666 /var/run/docker.sock


Go 
**********************
Go must be install in order to use Cesium terrain server 

Check if Go is installed :

.. code-block::

    go version

If go is not installed

.. code-block::

    sudo apt install gccgo-go

Cesium terrain server 
**********************
If go is not installed

.. code-block::

    go get github.com/geo-data/cesium-terrain-server/cmd/cesium-terrain-server


NodeJS 
**********************
Check if NodeJS is installed :


.. code-block::

    node -v

If NodeJS is not installed

.. code-block::

    sudo apt install nodejs


ExpressJS 
**********************
Install ExpressJS as follow 


.. code-block::

    sudo apt install npm
    npm install express


Conda  
**********************
Check if conda is installed :

.. code-block::

    conda info

If conda is not installed,  `check this page <https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04>`_



Get Started
--------------------


Once the above elements are ready, the dependencies can be installed as follow :

.. code-block::

    sudo bash setup/install.sh


And conda environment :

.. code-block::

    conda env create -f setup/environment.yml
    conda activate topo-datagen

