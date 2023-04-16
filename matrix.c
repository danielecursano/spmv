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
    int r, iter_row;
    for (r=0; r<matrix->row; r++) {
        iter_row = 0;
        while (iter_row < matrix->nnz) {
            if (r==(*matrix).row_index[iter_row]) {
                output[r] += (*matrix).values[iter_row]*vector[(*matrix).col_index[iter_row]];
            }
            iter_row++;
        }
    }
    return output;
}