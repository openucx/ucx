/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_AARCH64_CPU_H_
#define UCS_AARCH64_CPU_H_

#include <ucs/debug/log.h>


#define UCS_SYS_CACHE_LINE_SIZE    64

/**
 * Assume the worst - weak memory ordering.
 */
#define ucs_memory_bus_fence()        ucs_fatal("unimplemented");
#define ucs_memory_bus_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_bus_load_fence()   ucs_memory_bus_fence()
#define ucs_memory_cpu_fence()        ucs_memory_bus_fence()
#define ucs_memory_cpu_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_cpu_load_fence()   ucs_memory_bus_fence()


static inline clock_t ucs_arch_read_hres_clock(void)
{
    struct tms accurate_clock;
    times(&accurate_clock);
    return accurate_clock.tms_utime + accurate_clock.tms_stime;
}

static inline double ucs_arch_get_clocks_per_sec()
{
    return CLOCKS_PER_SEC;
}

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_UNKNOWN;
}

#endif
