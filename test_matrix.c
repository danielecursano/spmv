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
    srand(time(NULL));
    if (argv < 2) {
        printf("Usage: ./exec PATH\n");
        return 0;
    }

    int DIM, NNZ, i;
    double start, end;
    FILE *data;

    data = fopen(argc[1], "r");
    if (data==NULL) {
        printf("Error: file not open!\n");
        return 0;
    }
    fscanf(data, "%d %d %d", &DIM, &DIM, &NNZ);
    float *values = (float *)malloc(sizeof(float)*NNZ);
    int *col = (int *)malloc(sizeof(float)*NNZ);
    int *row = (int *)malloc(sizeof(float)*NNZ);
    int *kernel = (int *)malloc(sizeof(int)*DIM);
    printf("UPLOADING DATA... (LEN %d)\n", DIM);
    start = get_time();
    for (i=0; i<NNZ; i++) {
        fscanf(data, "%d %d %f", &col[i], &row[i], &values[i]);
    }
    for (i=0; i<DIM; i++)
        kernel[i] = rand() % 50;
    end = get_time();
    printf("DATA INITIALIZED in %lfs\n", end-start);
    struct CSR_Matrix matrix = {DIM, DIM, NNZ, col, row, values};
    start = get_time();
    float *output = product(&matrix, kernel);
    end = get_time();
    printf("EXEC TIME CPU: %lfs\n", end-start);
    for (i=0; i<DIM; i++) {
        printf("%f\n", output[i]);
    }
    /*
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
     */
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