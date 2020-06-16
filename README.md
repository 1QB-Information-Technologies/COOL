# Controlled Online Optimization Learning (COOL): Finding the ground state of spin Hamiltonians with reinforcement learning

![COOL](http://kylemills.ca/image/COOL_square_1k.jpg)

## Building
You must compile the Simulated Annealing backend (written in C++) before using the gym environment.

These instructions were made using a fresh Ubuntu 18.04 installation, with Anaconda 2020.02 (Python 3.7) installed. Anaconda can be obtained from [https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/).




### Linux (Debian) 
Install dependencies and build the Python interface to the Simulated Annealing backend:

``` bash
apt install swig g++ libtclap-dev libboost-dev 
make install
```



### Mac OS X 
This package should work on a Mac, but dependencies are more difficult to satisfy. You will probably need to edit `setup.py` to manually point to your boost include paths, tclap, etc. On Linux it just works (with apt).



### Examples
Example experiments relevant to the COOL manuscript are included in `experiments/`. The README examples for these experiments use the `COOL_HOME` environment variable. It's not necessary, but helpful to define this.  This is the top level directory of this repository.

```bash 
export COOL_HOME=/home/user/git/COOL/
```

If you wish to run the examples (reinforcement learning code), you will need to install the Python dependencies.

```bash
pip install stable-baselines
conda install tensorflow-gpu=1.13
```


## Research paper: 
[![arXiv link](http://kylemills.ca/image/COOL_card.png)](https://arxiv.org/abs/2003.00011)




## Note:
`environment.yml` is a list of the developer's Python environment used during development.  It contains more software than is required, but can be manually consulted for package version numbers, should the need arise.  Most users will not need this file.

<a href="https://zenodo.org/badge/latestdoi/250016240"><img src="https://zenodo.org/badge/250016240.svg" alt="DOI"></a>
