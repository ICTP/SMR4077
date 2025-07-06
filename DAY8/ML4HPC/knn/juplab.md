On for example, 

mcelori1@login05

lsof -ti:8787 | xargs kill -9

Suppose that the outcome of 

squeue --me

is

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          16927999 boost_usr run-dask mcelori1  R       2:49      2 lrdn[2171,2189]


Then enter

On another shell on local PC

ssh -L 8787:localhost:8787 mcelori1@login05-ext.leonardo.cineca.it ssh -L 8787:localhost:8787 -N lrdn2171

Finally open a browser and

http://localhost:8787/status


