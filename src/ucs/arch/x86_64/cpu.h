/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASM_X86_64_H_
#define UCS_ASM_X86_64_H_

#include <ucs/sys/compiler.h>
#include <ucs/arch/generic/cpu.h>
#include <ucs/sys/compiler_def.h>
#include <stdint.h>

#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif
#ifdef __AVX__
#  include <immintrin.h>
#endif

BEGIN_C_DECLS

#define UCS_ARCH_CACHE_LINE_SIZE 64

/**
 * In x86_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 */
#define ucs_memory_bus_fence()        asm volatile ("mfence"::: "memory")
#define ucs_memory_bus_store_fence()  asm volatile ("sfence" ::: "memory")
#define ucs_memory_bus_load_fence()   asm volatile ("lfence" ::: "memory")
#define ucs_memory_cpu_fence()        ucs_compiler_fence()
#define ucs_memory_cpu_store_fence()  ucs_compiler_fence()
#define ucs_memory_cpu_load_fence()   ucs_compiler_fence()


static inline uint64_t ucs_arch_read_hres_clock()
{
    uint32_t low, high;
    asm volatile ("rdtsc" : "=a" (low), "=d" (high));
    return ((uint64_t)high << 32) | (uint64_t)low;
}

double ucs_arch_get_clocks_per_sec();

ucs_cpu_model_t ucs_arch_get_cpu_model() UCS_F_NOOPTIMIZE;
ucs_cpu_flag_t ucs_arch_get_cpu_flag() UCS_F_NOOPTIMIZE;

#define ucs_arch_wait_mem ucs_arch_generic_wait_mem

#if !HAVE___CLEAR_CACHE
static inline void ucs_arch_clear_cache(void *start, void *end)
{
    char *ptr;

    for (ptr = (char*)start; ptr < (char*)end; ptr++) {
        asm volatile("mfence; clflush %0; mfence" :: "m" (*ptr));
    }
}
#endif

END_C_DECLS

#endif

