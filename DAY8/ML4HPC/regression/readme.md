# Linear Regression

In `regression.py`, we consider a linear regression on a synthetic dataset.

The dataset is of the form $(y_i,\,X_{i1},...,X_{im})_{i=1}^{n}$ of $n$ samples and $m$ features, generated from a linear model, that is a linear relationship between the dependent variable $y$ and the vector of regressors $X$.  

$$y_i=\sum_{f=1}^{m}X_{if} c_f + b $$

where $c_f$ is a vector of $m$ coefficients and $b$ is the bias.

Among the $m$ features, we suppose that only $l$ features are informative, that is only $c_1,...,c_l$ are different from zero.

Given the dataset $X, y$ generated from the true underlying coefficients $c_m$ and the true underlying bias $b$, the goal is to learn the coeffients $C_m$ and the bias $B$ from the data, by minimizing the mean squared error (MSE).

In `dask_cuda`, the $X$ is an array of the form 

X:
 dask.array<concatenate, shape=(320000000, 64), dtype=float32, chunksize=(40000000, 64), chunktype=cupy.ndarray>

while 

y:
 dask.array<getitem, shape=(320000000,), dtype=float32, chunksize=(40000000,), chunktype=cupy.ndarray>

for a dataset of $n=320000000$ samples, with $m=64$ features divided into 4 workers.
Notice that the `dask.array` elements are of  `dtype=float32` and each chunk is offloaded on GPU by means of `cupy.ndarray`.

Suppose that only 16 features are informative that is the true coefficients are

c_1  =   -2.981 

c_2  =   39.310
 
c_3  =  -80.567

c_4  = -138.952

c_5  = -144.547

c_6  = -105.075

c_7  = -107.430

c_8  =    9.976

c_9  =  -12.745

c_10 =  -81.459

c_11 =  204.275

c_12 =  -13.614

c_13 =  135.765

c_14 =  -11.804

c_15 =  -87.049

c_16 =  -20.891

The other true coefficients being zero.

Moreover, suppose that the true bias is one, that is b = 1.0
 
Then, we instantiate a cuml.dask.linear_model.LinearRegression `lr` and fit over the dataset:

`lr.fit(X, y)`

The resulting learned coefficients C are


C_1  =   -3.029

C_2  =   39.939

C_3  =  -81.855

C_4  = -141.177

C_5  = -146.860

C_6  = -106.758

C_7  = -109.151

C_8  =   10.136

C_9  =  -12.950

C_10 =  -82.763

C_11 =  207.544

C_12 =  -13.832

C_13 =  137.938

C_14 =  -11.993

C_15 =  -88.444

C_16 =  -21.226

The other C_i being of the order of 1e-4.

The learned bias B is

B = 0.9997920

Try to add noise to the dataset and see what happens.


