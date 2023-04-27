#include "stdlib.h"
#include "time.h"
#include "sys/time.h"
#include "stdio.h"
#include "utils.h"

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

struct CSR_Matrix {
    int row, col, nnz;
    int *col_index, *row_index;
    float *values;
};

float *product(struct CSR_Matrix *matrix, int *vector) {
    int iter;
    float *output = (float *) calloc(matrix->row, sizeof(float ));
    for (iter=0; iter<matrix->nnz; iter++) {
        output[(*matrix).row_index[iter]] += (*matrix).values[iter]*vector[(*matrix).col_index[iter]];
    }
    return output;
}

__global__ void productKernel(float *values, int *columns, int *rows, int *kernel, float *output, int nnz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int section = (nnz-1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section;
    for (size_t k=0; k<section; k++) {
        if (start+k < nnz)
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
    int i, *kernel, *row_index, *col_index;
    float *values, *output, *gpu_output;
    double start, end;
    CHECK(cudaMallocManaged(&kernel, sizeof(int)*LEN));
    output = (float *)calloc(LEN, sizeof(float));
    CHECK(cudaMallocManaged(&gpu_output, sizeof(float)*LEN));
    CHECK(cudaMallocManaged(&row_index, sizeof(int)*NNZ));
    CHECK(cudaMallocManaged(&col_index, sizeof(int)*NNZ));
    CHECK(cudaMallocManaged(&values, sizeof(float)*NNZ));
    printf("STARTED SCANNING MATRIX...\n");
    for (i=0; i<NNZ; i++) {
        fscanf(data, "%d %d %f", &col_index[i], &row_index[i], &values[i]);
    }
    for (i=0; i<LEN; i++) {
        kernel[i] = rand()%20;
    }
    printf("VALUES SCANNED\n");

    //cpu process
    CSR_Matrix matrix = {LEN, LEN, NNZ, col_index, row_index, values};
    start = get_time();
    output = product(&matrix, kernel);
    end = get_time();
    printf("EXEC TIME CPU: %lf s\n", end-start);

    dim3 blocksPerGrid((LEN + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
    dim3 threadsPerBlock(BLOCKDIM, 1, 1);
    start = get_time();
    productKernel<<<blocksPerGrid, threadsPerBlock>>>(values, col_index, row_index, kernel, gpu_output, NNZ);
    CHECK_KERNELCALL();
    end = get_time();
    printf("EXEC TIME GPU: %lf s\n", end-start);

    CHECK(cudaFree(values));
    CHECK(cudaFree(col_index));
    CHECK(cudaFree(row_index));
    CHECK(cudaFree(kernel));
    CHECK(cudaFree(gpu_output));
    free(output);
}
