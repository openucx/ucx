/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef GDAKI_MEM_H_
#define GDAKI_MEM_H_

#include <ucs/sys/compiler_def.h>
#include <cuda_runtime.h>
#include <tools/perf/lib/libperf_int.h>


#define CUDA_CALL(_func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            ucs_error("%s() failed: %d (%s)", UCS_PP_MAKE_STRING(_func), \
                      _cerr, cudaGetErrorString(_cerr)); \
        } \
    } while (0)

class gdaki_mem {
public:
    gdaki_mem(size_t size) : m_size(size) {
        CUDA_CALL(cudaSetDeviceFlags, cudaDeviceMapHost |
                                      cudaDeviceScheduleBlockingSync);
        CUDA_CALL(cudaHostAlloc, &m_cpu_ptr, size, cudaHostAllocMapped);
        CUDA_CALL(cudaHostGetDevicePointer, &m_gpu_ptr, m_cpu_ptr, 0);
    }

    ~gdaki_mem() {
        CUDA_CALL(cudaFreeHost, m_cpu_ptr);
    }

    void *get_cpu_ptr() const { return m_cpu_ptr; }
    void *get_gpu_ptr() const { return m_gpu_ptr; }

private:
    void   *m_gpu_ptr;
    void   *m_cpu_ptr;
    size_t m_size;
};

#endif /* GDAKI_MEM_H_ */
