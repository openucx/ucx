/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_CUDA_COMMON_H
#define TEST_CUDA_COMMON_H

#include <cuda.h>
#include <mpi.h>
#include <stdio.h>


#define PRINT_ROOT(_fmt, ...) \
    do { \
        int _rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
        if (_rank == 0) { \
            fprintf(stdout, _fmt, ##__VA_ARGS__); \
        } \
    } while (0)


#define LIB_CHECK(_err_t, _func, _success_code, _lib_error_string) \
    do { \
        _err_t _err = (_func); \
        if (_err != _success_code) { \
            fprintf(stderr, "%s failed: %d (%s)\n", UCS_PP_MAKE_STRING(_func), \
                    _err, _lib_error_string(_err)); \
            exit(_err); \
        } \
    } while (0)


typedef struct init_params *init_params_h;


typedef struct {
    int         rank;
    CUdeviceptr send_ptr;
    CUdeviceptr recv_ptr;
    size_t      size;
    const void  *xfer_args;
} xfer_params_t;


typedef void (*xfer_t)(const xfer_params_t*);


int test_cuda_init(int argc, char **argv, init_params_h *init_params_p);


void test_cuda(const init_params_h init_params, const xfer_t xfer,
               const void *xfer_args);


void test_cuda_cleanup(const init_params_h init_params);

#endif
