In Leonardo $HOME

cd $HOME


curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

-----------------------------------------------------------------------------------

Welcome to Miniforge3 25.3.0-3

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>> 
Miniforge installer code uses BSD-3-Clause license as stated below.
......................

Do you accept the license terms? [yes|no]
>>> yes

Miniforge3 will now be installed into this location:
/leonardo/home/userinternal/mcelori1/miniforge3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/leonardo/home/userinternal/mcelori1/miniforge3] >>> 
PREFIX=/leonardo/home/userinternal/mcelori1/miniforge3
Unpacking payload ...

Installing base environment...

Transaction
......

Transaction starting
.......
Transaction finished

installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> no

You have chosen to not have conda modify your shell scripts at all.
To activate conda's base environment in your current shell session:

eval "$(/leonardo/home/userinternal/mcelori1/miniforge3/bin/conda shell.YOUR_SHELL_NAME hook)" 

To install conda's shell functions for easier access, first activate, then:

conda init

Thank you for installing Miniforge3!


-----------------------------------------------------------------------------------

Now, 

eval "$(/leonardo/home/userinternal/mcelori1/miniforge3/bin/conda shell.bash hook)"

and

cd /leonardo_work/cin_staff/mcelori1/CUDA_DASK/1_rapids 

Load the following modules

module purge
module load gcc/12.2.0
module load cuda/12.1

and set

conda config --set channel_priority flexible

We can now create the environment

conda create -p /leonardo_work/cin_staff/mcelori1/Magurele/rapids_venv -c conda-forge -c rapidsai -c nvidia --solver=libmamba
conda activate  /leonardo_work/cin_staff/mcelori1/Magurele/rapids_venv
conda install -c rapidsai -c conda-forge -c nvidia  rapids=25.06 dask-cuda ucx-py graphistry jupyterlab networkx nx-cugraph=25.06 dash xarray-spatial holoviews hvplot gcsfs graphviz cuda-cudart jupyterlab dask-jobqueue cuda-version=12.1
conda install cuda-cudart cuda-version=12


conda env create -p /leonardo_work/cin_staff/mcelori1/Magurele/rapids_venv --file=environment.yml
conda activate /leonardo_work/cin_staff/mcelori1/Magurele/rapids_venv
conda install cuda-cudart cuda-version=12
conda deactivate



