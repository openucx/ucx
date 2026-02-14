/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_UCT_TEST_KERNELS_H_
#define CUDA_UCT_TEST_KERNELS_H_

#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>

#if defined(__NVCC__) && !defined(__builtin_ia32_prefetch)
  #define __builtin_ia32_prefetch(P, RW, LOC, CACHE) \
    __builtin_prefetch((P), (RW), (LOC))
#endif

namespace ucx_cuda {

ucs_status_t launch_uct_put_single(uct_device_ep_h ep,
                                   uct_device_mem_element_t *mem_elem,
                                   const void *va, uint64_t rva, size_t length);

ucs_status_t launch_uct_atomic(uct_device_ep_h ep,
                               uct_device_mem_element_t *mem_elem, uint64_t rva,
                               uint64_t add);

template<size_t iovcnt>
ucs_status_t
launch_uct_put_multi(uct_device_ep_h ep, uct_device_mem_element_t *mem_list,
                     const void *va, uint64_t rva, uint64_t atomic_rva,
                     size_t length);

template<size_t iovcnt>
ucs_status_t
launch_uct_put_partial(uct_device_ep_h ep, uct_device_mem_element_t *mem_list,
                       const void *va, uint64_t rva, uint64_t atomic_rva,
                       size_t length);

}; // namespace ucx_cuda

#endif
