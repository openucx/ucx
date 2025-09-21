/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_CUDA_DEVICE_CUH
#define UCS_CUDA_DEVICE_CUH

#include <stdint.h>


/* Device function */
#define UCS_F_DEVICE __device__ __forceinline__ static


/* Device library function */
#define UCS_F_DEVICE_LIB __device__


/*
 * Read a 64-bit atomic value from a global memory address.
 */
UCS_F_DEVICE uint64_t ucs_device_atomic64_read(const uint64_t *ptr)
{
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}


/*
 * Write a 64-bit value to counter global memory address.
 */
UCS_F_DEVICE void ucs_device_atomic64_write(uint64_t *ptr, uint64_t value)
{
    asm volatile("st.release.sys.u64 [%0], %1;"
                 :
                 : "l"(ptr), "l"(value)
                 : "memory");
}


/*
 * Read the 64-bit GPU global nanosecond timer
 */
UCS_F_DEVICE uint64_t ucs_device_get_time_ns(void)
{
    uint64_t globaltimer;
    /* 64-bit GPU global nanosecond timer */
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

/*
 * Load a constant from global memory.
 */
template<typename T> UCS_F_DEVICE T ucs_device_load_const(const T *ptr)
{
    return __ldg(ptr);
}

template<> inline __device__ void *ucs_device_load_const(void *const *ptr)
{
    return (void*)__ldg((uint64_t*)ptr);
}

#endif /* UCS_CUDA_DEVICE_CUH */
