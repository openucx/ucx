/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_GENERIC_CPU_H_
#define UCS_GENERIC_CPU_H_

#include <sys/time.h>
#include <stdint.h>

/**
 * These hints can be used with the ucs_memcpy_relaxed() function
 * to optimize memory copy performance. Hints are derived based
 * on the total length of the buffer transfer or the temporal locality
 * of the data.
 *
 * If the total length of data transfer is more or closer
 * to the size of last level cache, it is safe to assume the destination
 * buffer is non-temporal.
 *
 * If the source buffer is no longer needed or not usable in the CPU that
 * performs the buffer transfer, it is safe to assume the source buffer
 * is non-temporal. The same is true for destination buffer as well.
 */
typedef enum {
    /* Both the source and destination buffers can be temporal */
    UCS_ARCH_MEMCPY_NT_NONE   = 0,
    /* Source buffer is of non-temporal nature */
    UCS_ARCH_MEMCPY_NT_SOURCE = UCS_BIT(0),
    /* Destination buffer is of non-temporal nature */
    UCS_ARCH_MEMCPY_NT_DEST   = UCS_BIT(1),
} ucs_arch_memcpy_hint_t;

static inline uint64_t ucs_arch_generic_read_hres_clock(void)
{
    struct timeval tv;

    if (gettimeofday(&tv, NULL) != 0) {
	return 0;
    }
    return ((((uint64_t)(tv.tv_sec)) * 1000000ULL) + ((uint64_t)(tv.tv_usec)));
}

static inline double ucs_arch_generic_get_clocks_per_sec()
{
    return 1.0E6;
}

static inline void ucs_arch_generic_wait_mem(void *address)
{
    /* NOP */
}

#endif
