/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_PPC64_CPU_H_
#define UCS_PPC64_CPU_H_

#include <ucs/sys/compiler.h>
#ifdef HAVE_SYS_PLATFORM_PPC_H
#  include <sys/platform/ppc.h>
#endif
#include <ucs/sys/compiler_def.h>
#include <ucs/arch/generic/cpu.h>
#include <stdint.h>

BEGIN_C_DECLS

#define UCS_ARCH_CACHE_LINE_SIZE 128

/* Assume the worst - weak memory ordering */
#define ucs_memory_bus_fence()        asm volatile ("sync"::: "memory")
#define ucs_memory_bus_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_bus_load_fence()   ucs_memory_bus_fence()
#define ucs_memory_cpu_fence()        ucs_memory_bus_fence()
#define ucs_memory_cpu_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_cpu_load_fence()   asm volatile ("lwsync \n" \
                                                    "isync  \n" \
                                                    ::: "memory")


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

static inline int ucs_arch_get_cpu_flag()
{
    return UCS_CPU_FLAG_UNKNOWN;
}

double ucs_arch_get_clocks_per_sec();

#define ucs_arch_wait_mem ucs_arch_generic_wait_mem

#if !HAVE___CLEAR_CACHE
static inline void ucs_arch_clear_cache(void *start, void *end)
{
    ucs_memory_cpu_fence();
}
#endif

END_C_DECLS

#endif
