# Machine Leanrt Interatomic potentials

in this tutorial we will use three codes, [janus-core](https://github.com/stfc/janus-core), [lammps with mliap](https://www.lammps.org/) and lammps with [symmetryx](https://github.com/wcwitt/symmetrix).

The main aim is to deploy these code on an HPC system, Leonardo at cineca and run simple tasks with them.

Time permitting you can explore scalability of various systems.

While some of the codes can run multigpu, we concentrate here on single gpu only.


running python applications on supercomputers can be challenging we suggest to use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to setup a clean envinronment.

# installing janus core

janus core is an aggrecator of machine learnt interatomic potentials, that relies on Atomistic Simulation Environment calculaors.
It also bundles workflows for various modelling tasks, as geometry optimisation, molecular dynamics, equation of state.

to install

```bash

source ~/.bashrc
# safe versions for various things
py_ver=3.12
cueq_ver=0.5.1
torch_ver=2.7.1


# load any modules your environment may needs
module load nvhpc/24.3

# name of the environment
# check the path for your environments and use the name you want

m="janus"
rm -rf ~/micromamba/envs/$m
micromamba create -n $m
micromamba activate $m
micromamba install -y python==$py_ver -c conda-forge

python3 -m pip install uv
uv pip install 'janus-core[mace]@git+https://github.com/stfc/janus-core.git'

python3 -m pip uninstall mace-torch
python3 -m pip install -U torch==$torch_ver
python3 -m pip install git+https://github.com/ACEsuit/mace.git

# up to here you will have a cutting edge environment

# cuda speed up for mace
python3 -m pip install cuequivariance==$cueq_ver
python3 -m pip install cuequivariance-torch==$cueq_ver
python3 -m pip install cuequivariance-ops-torch-cu12==$cueq_ver

# this is needed for lammps at the moment
python3 -m pip install -U cupy-cuda12x


# ase latest version
python3 -m pip install git+https://gitlab.com/ase/ase.git

# faster dispersion if needed on cuda
python3 -m pip install -U git+https://github.com/CheukHinHoJerry/torch-dftd.git

```

you can save all in a bash script and run it.


test your installation.
a set of tutorials can be found on the [webpage](https://stfc.github.io/janus-core/) choose your favourite and test the
installation.
