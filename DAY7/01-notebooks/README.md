# Jupyter notebook

Spawn a jupyter notebook on **Leonardo cluster** just executing the job file `get_notebook.sh`.

```bash
$  sbatch -A tra25_ictp_rom get_notebook.sh
```

And then follow the instruction on the file `connection_instruction.txt`:

```bash 
Run on you laptop: ssh -L 12345:10.8.0.201:12345 -N a08tra40@login.leonardo.cineca.it -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
Open on you browser: http://127.0.0.1:12345
Pls. do not closet the terminal just opened!
```
This open an SSH tunnel between your laptop and the compute node assigned to you by the scheduler via Login node.
## Content

### `00-pytorch.ipynb`

This notebook will show you the core concepts about pytorch: `DataSet`,`DataLoader`,`Optimizer` and `Gradients`.
You will train your first model with just 2 paramters. 

### `01-mnist-training.ipynb`

This notebook will show you the training framework that almost any model follow:

- Dataset setup
- Model setup
- Training loop
- Performance assessment
