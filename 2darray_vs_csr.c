#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "utils.h"
#include "matrix.h"
//#define LEN 4

int * matrix_multiplication(int **matrix, int *kernel, int m_rows, int m_cols, int k_rows)
{
    // k_cols == 1
    // kernel is a column vector
    if (m_cols != k_rows) {
        printf("ERROR: %d should be equals to %d!", m_cols, k_rows);
        exit(0);
    }
    int *output = (int *)calloc(m_cols, sizeof(int));
    for (int i=0; i<m_rows; i++)
        for (int k=0; k<m_cols; k++) {
            output[i] += matrix[i][k] * kernel[k];
        }
    return output;
}

void print(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argv, char **argc) {
    srand(time(NULL));
    int LEN = atoi(argc[1]);
    int i, j, *kernel = (int *)calloc(LEN, sizeof(int));
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
    /*
    printf("KERNEL: ");
    for (i=0; i<LEN; i++)
        printf("%d ", kernel[i]);
    printf("\n");
    */
    //int matrix[4][4] = {{1, 0, 3, 0}, {0, 0, 0, 0}, {0, 2, 4, 0}, {7, 8, 0, 0}};
    //print(matrix, LEN, LEN);
    start = get_time();
    int *output = matrix_multiplication(matrix, kernel, LEN, LEN, LEN);
    end = get_time();
    printf("FIRST METHOD: %lf s\n", end-start);
    /*
    printf("OUTPUT: ");
    for (i=0; i<LEN; i++)
        printf("%d ", output[i]);
    printf("\n");
    */
    struct CSR_Matrix *new = fromMatrix(matrix, LEN, LEN);
    //print_matrix(new);
    start = get_time();
    int *output2 = product(new, kernel);
    end = get_time();
    printf("SECOND METHOD: %lf s\n", end-start);
    /*
    printf("OUTPUT 2: ");
    for (i=0; i<LEN; i++)
        printf("%d ", output2[i]);
    printf("\n");
    */

    for (i=0; i<LEN; i++) {
        if (output[i]!=output2[i]) {
            printf("OUTPUTS NOT MATCHING!\n");
            exit(0);
        }
    }

    for (i=0; i<LEN; i++)
        free(matrix[i]);
    free(matrix);
    free(output);
    free(kernel);
    free(output2);
}
