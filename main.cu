#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "stdio.h"

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void productKernel(float *values, int *columns, int *rows, int *kernel, int *output, int *nnz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int section = (*nnz-1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section;
    for (size_t k=0; k<section; k++) {
        if (start+k < *nnz)
            atomicAdd(&(output[rows[start+k]]), values[start+k] * kernel[columns[start+k]]);
    }
}

int main(int argv, char **argc) {
    srand(time(NULL));
    if (argv!=3) {
        printf("Usage: ./exec BLOCKDIM DATAPATH\n");
        return 0;
    }
    FILE *data;
    data = fopen(argc[2], "r");
    int BLOCKDIM = atoi(argc[1]);
    int LEN, NNZ;
    fscanf(data, "%d %d %d", &LEN, &LEN, &NNZ);
    int i, *kernel = (int *)malloc(LEN*sizeof(int)), *output = (int *)calloc(LEN, sizeof(int)), *gpu_output = (int *)calloc(LEN, sizeof(int));
    float *values = (float*)malloc(NNZ*sizeof(float));
    int *col_index = (int *)malloc(NNZ*sizeof(int));
    int *row_index = (int *)malloc(NNZ*sizeof(int));
    double start, end;
    printf("STARTED SCANNING MATRIX...\n");
    for (i=0; i<NNZ; i++) {
        fscanf(data, "%d %d %f", &col_index[i], &row_index[i], &values[i]);
    }
    printf("VALUES SCANNED\n");
    dim3 blocksPerGrid((LEN + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
    dim3 threadsPerBlock(BLOCKDIM, 1, 1);
    float *d_values;
    int *dcol_index, *drow_index, *d_kernel, *d_output, *d_nnz;
    CHECK(cudaMalloc(&d_values, NNZ*sizeof(float)));
    CHECK(cudaMalloc(&dcol_index, NNZ*sizeof(int)));
    CHECK(cudaMalloc(&drow_index, NNZ*sizeof(int)));
    CHECK(cudaMalloc(&d_kernel, LEN*sizeof(int)));
    CHECK(cudaMalloc(&d_output, LEN*sizeof(int)));
    CHECK(cudaMalloc((void**)&d_nnz, sizeof(int)));

    CHECK(cudaMemcpy(d_values, values, NNZ*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dcol_index, col_index, NNZ*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drow_index, row_index, NNZ*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, kernel, LEN*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nnz, &NNZ, sizeof(int), cudaMemcpyHostToDevice));
    start = get_time();
    productKernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, dcol_index, drow_index, d_kernel, d_output, d_nnz);
    CHECK_KERNELCALL();
    end = get_time();
    CHECK(cudaMemcpy(gpu_output, d_output, LEN*sizeof(int), cudaMemcpyDeviceToHost));
    printf("EXEC TIME GPU: %lf s\n", end-start);

    CHECK(cudaFree(d_values));
    CHECK(cudaFree(dcol_index));
    CHECK(cudaFree(drow_index));
    CHECK(cudaFree(d_kernel));
    CHECK(cudaFree(d_output));
}
