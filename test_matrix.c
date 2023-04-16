#include "matrix.h"

int main() {
    int values[]={5, 8, 3, 6}, col[]={0, 1, 2, 1}, row[]={0, 1, 2, 3, 4};
    struct CSR_Matrix matrix={4, 4, 4, values, col, row};
    print_matrix(&matrix);
}