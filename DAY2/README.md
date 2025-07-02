## For running mpi4py 
In the repo there are some simple examples of python codes using mpi4py that you can look at. To run them, run the following commands.  
First, create a new virtual environment where to install mpi4py:  
```python -m venv .mpi4py```  
then activate the environment:  
```source .mpi4py/bin/activate```  
Then, install mpi4py based, for example, on openmpi by doing:  
```pip install openmpi mpi4py```  
For some of the snippets you will need numpy:  
```pip install numpy```  
Finally, run the scripts as:  
```mpiexec -n <ntasks> python filename.py```  

For a reference for the API see for example: [https://mpi4py.readthedocs.io/en/stable/index.html](https://mpi4py.readthedocs.io/en/stable/index.html)
