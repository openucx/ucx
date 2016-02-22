/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_AARCH64_CPU_H_
#define UCS_AARCH64_CPU_H_

#include <ucs/debug/log.h>
#include <time.h>
#include <sys/times.h>


#define UCS_SYS_CACHE_LINE_SIZE    64

/**
 * Assume the worst - weak memory ordering.
 */
#define ucs_memory_bus_fence()        asm volatile ("dsb sy" ::: "memory");
#define ucs_memory_bus_store_fence()  asm volatile ("dsb st" ::: "memory");
#define ucs_memory_bus_load_fence()   asm volatile ("dsb ld" ::: "memory");
#define ucs_memory_cpu_fence()        asm volatile ("dmb ish" ::: "memory");
#define ucs_memory_cpu_store_fence()  asm volatile ("dmb ishst" ::: "memory");
#define ucs_memory_cpu_load_fence()   asm volatile ("dmb ishld" ::: "memory");


static inline uint64_t ucs_arch_read_hres_clock(void)
{
    uint64_t ticks;
    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r" (ticks));
    return ticks;
}

static inline double ucs_arch_get_clocks_per_sec()
{
    uint32_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (freq));
    return (double) freq;
}

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_ARM_AARCH64;
}

static inline int ucs_arch_get_cpu_flag()
{
    return UCS_CPU_FLAG_UNKNOWN;
}

#endif
