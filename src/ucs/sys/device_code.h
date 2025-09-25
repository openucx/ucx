/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_DEVICE_CODE_H
#define UCS_DEVICE_CODE_H

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

/*
 * Declare GPU specific functions
 */
#ifdef __NVCC__
#define UCS_F_DEVICE __device__ __forceinline__ static
#else
#define UCS_F_DEVICE static inline
#endif /* __NVCC__ */


/* Number of threads in a warp */
#define UCS_DEVICE_NUM_THREADS_IN_WARP 32


/**
 * @brief Cooperation level when calling device functions.
 */
typedef enum {
    UCS_DEVICE_LEVEL_THREAD = 0,
    UCS_DEVICE_LEVEL_WARP   = 1,
    UCS_DEVICE_LEVEL_BLOCK  = 2,
    UCS_DEVICE_LEVEL_GRID   = 3
} ucs_device_level_t;


static UCS_F_ALWAYS_INLINE const char*
ucs_device_level_name(ucs_device_level_t level)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        return "thread";
    case UCS_DEVICE_LEVEL_WARP:
        return "warp";
    case UCS_DEVICE_LEVEL_BLOCK:
        return "block";
    case UCS_DEVICE_LEVEL_GRID:
        return "grid";
    default:
        return "unknown";
    }
}


/*
 * Read a 64-bit atomic value from a global memory address.
 */
UCS_F_DEVICE uint64_t ucs_device_atomic64_read(const uint64_t *ptr)
{
    uint64_t ret;
#ifdef __NVCC__
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(ret)
                 : "l"(ptr));
#else
    ret = *ptr;
#endif
    return ret;
}


/*
 * Write a 64-bit value to counter global memory address.
 */
UCS_F_DEVICE void ucs_device_atomic64_write(uint64_t *ptr, uint64_t value)
{
#ifdef __NVCC__
    asm volatile("st.release.sys.u64 [%0], %1;"
             :
             : "l"(ptr), "l"(value)
             : "memory");
#else
    *ptr = value;
#endif
}


/* Helper macro to print a message from a device function including the
 * thread and block indices */
#define ucs_device_printf(_title, _fmt, ...) \
    printf("(%d:%d) %6s " _fmt "\n", threadIdx.x, blockIdx.x, _title, \
           ##__VA_ARGS__)

/* Print an error message from a device function */
#define ucs_device_error(_fmt, ...) \
    ucs_device_printf("ERROR", _fmt, ##__VA_ARGS__)


/* Print a debug message from a device function */
#define ucs_device_debug(_fmt, ...) \
    ucs_device_printf("DEBUG", _fmt, ##__VA_ARGS__)

#endif
