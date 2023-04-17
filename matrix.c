#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

int isValid(int row, int col, struct CSR_Matrix *matrix) {
    for (int r=0; r<matrix->nnz; r++) {
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
    int *output = (int *) calloc(matrix->row, sizeof(int));
    int iter;
    for (iter=0; iter<matrix->nnz; iter++)
        output[(*matrix).row_index[iter]] += (*matrix).values[iter]*vector[(*matrix).col_index[iter]];
    return output;
}