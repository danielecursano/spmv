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

struct CSR_Matrix {
    int row, col, nnz;
    int *values, *col_index, *row_index;
};

int isValid(int row, int col, struct CSR_Matrix *matrix) {
    int r;
    for (r=0; r<matrix->nnz; r++) {
        if ((*matrix).row_index[r]==row && (*matrix).col_index[r]==col) {
            return (*matrix).values[r];
        }
    }
    return 0;
}

void print_matrix(struct CSR_Matrix *matrix) {
    int c, r;
    for (r=0; r<matrix->row; r++) {
        for (c=0; c<matrix->col; c++)
            printf("%d ", isValid(r, c, matrix));
        printf("\n");
    }
}

int *product(struct CSR_Matrix *matrix, int *vector) {
    int iter, *output = (int *) calloc(matrix->row, sizeof(int));
    for (iter=0; iter<matrix->nnz; iter++) {
        output[(*matrix).row_index[iter]] += (*matrix).values[iter]*vector[(*matrix).col_index[iter]];
    }
    return output;
}

struct CSR_Matrix * fromMatrix(int **matrix, int r, int c) {
    int i, j, n_values = 0;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            if (matrix[i][j] != 0)
                n_values++;
        }
    }
    int *values = (int *) malloc(sizeof(int)*n_values);
    int *col_index = (int *) malloc(sizeof(int)*n_values);
    int *row_index = (int *) malloc(sizeof(int)*(n_values+1));
    int id=0;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            if (matrix[i][j] != 0) {
                values[id] = matrix[i][j];
                col_index[id] = j;
                row_index[id] = i;
                id++;
            }
        }
    }
    row_index[++id] = n_values;
    struct CSR_Matrix *m = (struct CSR_Matrix *) malloc(sizeof(struct CSR_Matrix));
    m->row = r;
    m->col = c;
    m->nnz = n_values;
    m->values = values;
    m->row_index = row_index;
    m->col_index = col_index;
    return m;
}

void freeMatrix(struct CSR_Matrix *matrix) {
    free(matrix->values);
    free(matrix->row_index);
    free(matrix->col_index);
}

__global__ void productKernel(int *values, int *columns, int *rows, int *kernel, int *output, int *nnz) {
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
        printf("Usage: ./exec MATRIXDIM BLOCKDIM\n");
        return 0;
    }
    int LEN = atoi(argc[1]), BLOCKDIM = atoi(argc[2]);
    int i, j, *kernel = (int *)calloc(LEN, sizeof(int)), *output = (int *)calloc(LEN, sizeof(int)), *gpu_output = (int *)calloc(LEN, sizeof(int));
    double start, end;

    int** matrix = (int**)malloc(LEN * sizeof(int*));
    for (i = 0; i < LEN; i++)
        matrix[i] = (int*)malloc(LEN * sizeof(int));

    for (i = 0; i < LEN; i++) {
        for (j = 0; j < LEN; j++) {
            if (rand() % 10 == 6) {
                matrix[i][j] = rand() % 11;
            } else {
                matrix[i][j] = 0;
            }
        }
        kernel[i] = rand() % 10;
    }
    struct CSR_Matrix *nex = fromMatrix(matrix, LEN, LEN);
    //print_matrix(nex);
    /*
    printf("KERNEL: ");
    for (i=0; i<LEN; i++) {
        printf("%d, ", kernel[i]);
    }
    printf("\n");
     */
    start = get_time();
    output = product(nex, kernel);
    end = get_time();
    printf("EXEC TIME CPU: %lf s\n", end-start);
    dim3 blocksPerGrid((LEN + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
    dim3 threadsPerBlock(BLOCKDIM, 1, 1);
    int *d_values, *dcol_index, *drow_index, *d_kernel, *d_output, *d_nnz;
    CHECK(cudaMalloc(&d_values, (*nex).nnz*sizeof(int)));
    CHECK(cudaMalloc(&dcol_index, (*nex).nnz*sizeof(int)));
    CHECK(cudaMalloc(&drow_index, ((*nex).nnz+1)*sizeof(int)));
    CHECK(cudaMalloc(&d_kernel, LEN*sizeof(int)));
    CHECK(cudaMalloc(&d_output, LEN*sizeof(int)));
    CHECK(cudaMalloc((void**)&d_nnz, sizeof(int)));

    CHECK(cudaMemcpy(d_values, nex->values, (*nex).nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dcol_index, nex->col_index, (*nex).nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drow_index, nex->row_index, (*nex).nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, kernel, LEN*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nnz, &(nex->nnz), sizeof(int), cudaMemcpyHostToDevice));
    start = get_time();
    productKernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, dcol_index, drow_index, d_kernel, d_output, d_nnz);
    CHECK_KERNELCALL();
    end = get_time();
    CHECK(cudaMemcpy(gpu_output, d_output, LEN*sizeof(int), cudaMemcpyDeviceToHost));
    printf("EXEC TIME GPU: %lf s\n", end-start);
    int flag=1;
    for (i=0; i<LEN; i++) {
        if(output[i]!=gpu_output[i]) {
            printf("VALUES NOT MATCHING\n");
            flag = 0;
            break;
        }
    }
    if (flag==1) {
        printf("SUCCESS!\n");
    }
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(dcol_index));
    CHECK(cudaFree(drow_index));
    CHECK(cudaFree(d_kernel));
    CHECK(cudaFree(d_output));
    freeMatrix(nex);
    free(nex);
    for (i=0; i<LEN; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(kernel);
    free(output);
    free(gpu_output);
}
