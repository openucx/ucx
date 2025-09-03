/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_DEVICE_MEM_H_
#define CUDA_DEVICE_MEM_H_

#include <tools/perf/lib/libperf_int.h>

BEGIN_C_DECLS

#define CUDA_CALL(_handler, _ret, _func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            _handler("%s() failed: %d (%s)", UCS_PP_MAKE_STRING(_func), \
                     (int)_cerr, cudaGetErrorString(_cerr)); \
            return _ret; \
        } \
    } while (0)


typedef struct {
    void   *cpu_ptr;
    void   *gpu_ptr;
    size_t size;
} cuda_device_mem_t;

ucs_status_t cuda_device_mem_create(cuda_device_mem_t *mem, size_t size);
void cuda_device_mem_destroy(cuda_device_mem_t *mem);

END_C_DECLS

#endif /* CUDA_DEVICE_MEM_H_ */
