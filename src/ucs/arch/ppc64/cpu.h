/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#ifndef UCS_PPC64_CPU_H_
#define UCS_PPC64_CPU_H_

#include <ucs/sys/compiler.h>
#ifdef HAVE_SYS_PLATFORM_PPC_H
#  include <sys/platform/ppc.h>
#endif
#include <stdint.h>


#define UCS_SYS_CACHE_LINE_SIZE    128

/* Assume the worst - weak memory ordering */
#define ucs_memory_bus_fence()        asm volatile ("sync"::: "memory")
#define ucs_memory_bus_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_bus_load_fence()   ucs_memory_bus_fence()
#define ucs_memory_cpu_fence()        ucs_memory_bus_fence()
#define ucs_memory_cpu_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_cpu_load_fence()   ucs_memory_bus_fence()


static inline uint64_t ucs_arch_read_hres_clock()
{
#ifndef HAVE_SYS_PLATFORM_PPC_H
    uint64_t tb;
    asm volatile ("mfspr %0, 268" : "=r" (tb));
    return tb;
#else
    return __ppc_get_timebase();
#endif
}

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_UNKNOWN;
}

double ucs_arch_get_clocks_per_sec();

#endif
