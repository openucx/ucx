/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#ifndef UCX_PY_COMMON_H
#define UCX_PY_COMMON_H

struct data_buf {
    void            *buf;
};

#define DEBUG 0

#if DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__);
#else
#define DEBUG_PRINT(...) do{} while(0);
#endif

#define WARN_PRINT(...)  fprintf(stdout, __VA_ARGS__);
#define ERROR_PRINT(...) fprintf(stderr, __VA_ARGS__);

#define CHKERR_JUMP(_cond, _msg, _label)            \
do {                                                \
    if (_cond) {                                    \
        fprintf(stderr, "Failed to %s\n", _msg);    \
        goto _label;                                \
    }                                               \
} while (0)

#define CUDA_CHECK(stmt)                                                \
    do {                                                                \
        cudaError_t cuda_err = stmt;                                    \
        if(cudaSuccess != cuda_err) {                                   \
            fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(cuda_err)); \
        }                                                               \
    } while(0)

#endif
