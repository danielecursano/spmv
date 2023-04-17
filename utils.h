#ifndef SPMV_UTILS_H
#define SPMV_UTILS_H

#include <time.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#endif //SPMV_UTILS_H
