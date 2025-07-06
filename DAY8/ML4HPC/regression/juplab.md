On for example,
  
mcelori1@login05

lsof -ti:8787 | xargs kill -9

Suppose that the outcome of

[mcelori1@login05 regression]$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          17010991 boost_usr run-dask mcelori1  R       0:06      1 lrdn0932


Then enter

On another shell on local PC

ssh -L 8787:localhost:8787 mcelori1@login05-ext.leonardo.cineca.it ssh -L 8787:localhost:8787 -N lrdn0932

Finally open a browser and

http://localhost:8787/status

