# PCA

In `pca.py`, we apply the Principal component analysis (PCA) on a synthetic dataset.

PCA is a linear dimensionality reduction technique with applications in exploratory data analysis, visualization and data preprocessing.

The data is linearly transformed onto a new coordinate system such that the directions (principal components) capturing the largest variation in the data can be easily identified. 

In other words, PCA is a statistical procedure that that reduces the number of dimensions in large datasets to principal components that retain most of the original information, thus summarizing the information content in large data tables.


Suppose a number of measurements are collected in a single experiment, and these measurements are typically arranged into a row vector.

The measurements may be features of an observable.
 
A number of experiments are conducted, and each measurement vector is arranged as a row in a large matrix $X$.

Thus, consider an $n\times m$ data matrix, $X$, where the $n$ rows represent $n$ samples, and each of the $m$ columns is a particular kind of feature.

Compute the average row  $\bar{X}$ (i.e., the mean of all rows), and subtract it from $X$. 

Subtracting $\bar{X}$ from X results in the mean-subtracted data B:

$$ B=X-\bar{X} $$

The covariance matrix of B is given by

$$ C=\frac{1}{n-1} B^* B $$

Note that the covariance is normalized by $n-1$ instead of $n$, even though there are nsample points. 

The covariance matrix $C$ is symmetric and positive semi-definite, having non-negative real eigenvalues. 

Each entry $C_{ij}$ quantifies the correlation of the i and j features across all experiments.

The principal components are the eigenvectors of $C$, and they define a change of coordinates in which the covariance matrix is diagonal:

$$ CV= VD $$

The columns of the eigenvector matrix $V$ are the principal components. 

The elements of the diagonal matrix $D$ are the variances of the data along these directions.

Consider for example an isotropic Gaussian blob X with n=1000000 samples (rows) and m=12 features (columns) where points are distributed around a single center (centers=1) with cluster standard deviation=0.01 divided into 32 parts 

We have

X:
 dask.array<concatenate, shape=(1000000, 12), dtype=float32, chunksize=(31250, 12), chunktype=cupy.ndarray>

since 1000000/32=31250 and notice the cupy.ndarray backend.

Let's scale with a diagonal matrix with non-zero components:

S = cp.diag(cp.array([4.0, 0.1, 0.1, 0.4, 0.5, 0.2, 0.3, 0.4, 2.0, 0.6, 0.2, 0.3]))

When transforming the cupy array into a dask array using da.from_array(cp.diag(sig), asarray=False), notice that the option asarray=True would then call np.asarray on chunks to convert them to numpy arrays. Since we do not want such conversion to numpy array, but relying on the cupy array, we set asarray=False so that chunks are passed through unchanged.

We then reduce the 12 features to just 2 features so that we can plot the dataset by instatiating a cuml.dask.decomposition.PCA model with 2 components, and call the fit and transform method.

The resulting dataset capturing the most informative features is 

XT\_persisted:
 dask.array<concatenate, shape=(1000000, 2), dtype=float64, chunksize=(31250, 2), chunktype=cupy.ndarray>

that can be plotted using holoviews. 
 



