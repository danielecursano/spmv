#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"
#include "utils.h"
#include "sys/time.h"
#include "time.h"

void print(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argv, char **argc) {

    int values[]={1, 3, 2, 4, 7, 8}, col[]={0, 2, 1, 2, 0, 1}, row[]={0, 0, 2, 2, 3, 3, 6};
    double start, end;
    struct CSR_Matrix matrix={4, 4, 6, values, col, row};
    print_matrix(&matrix);
    int kernel[] = {1, 2, 3, 4};
    start = get_time();
    int *output = product(&matrix, kernel);
    end = get_time();
    for (int i=0; i<4; i++) {
        printf("%d\n", output[i]);
    }
    printf("EXEC TIME: %lf\n",  end-start);
    free(output);
    /*
     * TESTING FROMMATRIX FUNCTION
    int LEN = atoi(argc[1]), i, j;
    srand(time(NULL));
    int** matrix = (int**)malloc(LEN * sizeof(int*));
    for (i = 0; i < LEN; i++)
        matrix[i] = (int*)malloc(LEN * sizeof(int));

    for (i = 0; i < LEN; i++) {
        for (j = 0; j < LEN; j++)
            matrix[i][j] = rand() % 50;
    }
    print(matrix, LEN, LEN);
    struct CSR_Matrix *new = fromMatrix(matrix, LEN, LEN);
    print_matrix(new);
    */
}