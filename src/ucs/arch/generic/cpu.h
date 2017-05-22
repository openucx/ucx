/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_GENERIC_CPU_H_
#define UCS_GENERIC_CPU_H_

#include <sys/time.h>
#include <stdint.h>


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
