#ifndef SPMV_MATRIX_H
#define SPMV_MATRIX_H

struct CSR_Matrix {
    int row, col, nnz;
    int *values, *col_index, *row_index;
};

int isValid(int row, int col, struct CSR_Matrix *matrix);
void print_matrix(struct CSR_Matrix *matrix);
int *product(struct CSR_Matrix *matrix, int *vector);
struct CSR_Matrix * fromMatrix(int **matrix, int r, int c);
void freeMatrix(struct CSR_Matrix *matrix);

#endif //SPMV_MATRIX_H
