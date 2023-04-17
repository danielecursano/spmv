## Sparse matrix-vector multiplication
This repo is a basic implementation of multiplication between a sparse matrix and a vector and its parallelization using CUDA C. In `csrmatrix_product.cu` there is an example which has been tested on google colab with a matrix of NxN dimension with N = 40000 and a block dimension of 1024 which gained a result in 0.000027s.
It is present also an implementation of the CSR Matrix struct in C to represent a sparse matrix, with methods to calculate the product with a vector that, with N=40000, cannot be compared with the Cuda results as the struct of the matrix in C represent a bottleneck for the cpu procedure.
## TODO
To increase the dimensions of the matrix use a dataset/text file and allocate the data directly in arrays.