#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"
#include "utils.h"

int main() {
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
}