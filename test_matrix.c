#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"

int main() {
    int values[]={1, 3, 2, 4, 7, 8}, col[]={0, 2, 1, 2, 0, 1}, row[]={0, 0, 2, 2, 3, 3, 6};
    struct CSR_Matrix matrix={4, 4, 6, values, col, row};
    print_matrix(&matrix);
    int kernel[] = {1, 2, 3, 4};
    int *output = product(&matrix, kernel);
    for (int i=0; i<4; i++) {
        printf("%d\n", output[i]);
    }
    free(output);
}