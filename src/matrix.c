#include "../include/matrix.h"
#include <stdio.h>
#include <stdlib.h>

float isValid(int row, int col, struct CSR_Matrix *matrix) {
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
            printf("%f ", isValid(r, c, matrix));
        printf("\n");
    }
}

float *product(struct CSR_Matrix *matrix, int *vector) {
    int iter;
    float *output = (float *) calloc(matrix->row, sizeof(float ));
    for (iter=0; iter<matrix->nnz; iter++) {
        output[(*matrix).row_index[iter]] += (*matrix).values[iter]*vector[(*matrix).col_index[iter]];
    }
    return output;
}

struct CSR_Matrix * fromMatrix(float **matrix, int r, int c) {
    int i, j, n_values = 0;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            if (matrix[i][j] != 0)
                n_values++;
        }
    }
    float *values = (float *) malloc(sizeof(float)*n_values);
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