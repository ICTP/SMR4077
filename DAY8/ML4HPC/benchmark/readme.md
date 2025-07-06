
Given:

[mcelori1@login05 ~]

and

/leonardo_work/cin_staff/mcelori1/CUDA_DASK/9_meluxina
File found
Starting 4 workers on lrdn0001
Starting 4 workers on lrdn0007
File found
8 workers are ready
DEBUGGG ucx://10.128.6.37:8786
Client done: 1749665073

and 

2025-06-11 19:45:46,879 - distributed.scheduler - INFO - -----------------------------------------------
2025-06-11 19:45:54,179 - distributed.scheduler - INFO - State start
2025-06-11 19:45:54,190 - distributed.scheduler - INFO - -----------------------------------------------
2025-06-11 19:45:56,345 - distributed.scheduler - INFO -   Scheduler at:    ucx://10.128.6.37:8786
2025-06-11 19:45:56,345 - distributed.scheduler - INFO -   dashboard at:  http://10.128.6.37:8787/status

To see the dashboard

(base) Mac:minutemen_fugazi marcoceloria$ ssh -L 8787:localhost:8787 mcelori1@login05-ext.leonardo.cineca.it ssh -L 8787:localhost:8787 -N lrdn0001


and open a browser and 

http://localhost:8787/status


