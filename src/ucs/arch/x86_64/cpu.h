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
#include <ucs/config/types.h>
#include <ucs/config/global_opts.h>
#include <stdint.h>
#include <string.h>

#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif
#ifdef __AVX__
#  include <immintrin.h>
#endif

BEGIN_C_DECLS

/** @file cpu.h */

#define UCS_ARCH_CACHE_LINE_SIZE 64

/**
 * In x86_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 */
#define ucs_memory_bus_store_fence()  asm volatile ("sfence" ::: "memory")
#define ucs_memory_bus_load_fence()   asm volatile ("lfence" ::: "memory")
#define ucs_memory_bus_cacheline_wc_flush()
#define ucs_memory_cpu_fence()        ucs_compiler_fence()
#define ucs_memory_cpu_store_fence()  ucs_compiler_fence()
#define ucs_memory_cpu_load_fence()   ucs_compiler_fence()
#define ucs_memory_cpu_wc_fence()     asm volatile ("sfence" ::: "memory")

extern ucs_ternary_auto_value_t ucs_arch_x86_enable_rdtsc;

double ucs_arch_get_clocks_per_sec();
void ucs_x86_init_tsc_freq();

ucs_cpu_model_t ucs_arch_get_cpu_model() UCS_F_NOOPTIMIZE;
ucs_cpu_flag_t ucs_arch_get_cpu_flag() UCS_F_NOOPTIMIZE;
ucs_cpu_vendor_t ucs_arch_get_cpu_vendor();
void ucs_cpu_init();
ucs_status_t ucs_arch_get_cache_size(size_t *cache_sizes);
void ucs_x86_memcpy_sse_movntdqa(void *dst, const void *src, size_t len);

static UCS_F_ALWAYS_INLINE int ucs_arch_x86_rdtsc_enabled()
{
    if (ucs_unlikely(ucs_arch_x86_enable_rdtsc == UCS_TRY)) {
        ucs_x86_init_tsc_freq();
        ucs_assert(ucs_arch_x86_enable_rdtsc != UCS_TRY);
    }

    return ucs_arch_x86_enable_rdtsc;
}

static UCS_F_ALWAYS_INLINE uint64_t ucs_arch_x86_read_tsc()
{
    uint32_t low, high;

    asm volatile("rdtsc" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (uint64_t)low;
}

static UCS_F_ALWAYS_INLINE uint64_t ucs_arch_read_hres_clock()
{
    if (ucs_unlikely(ucs_arch_x86_rdtsc_enabled() == UCS_NO)) {
        return ucs_arch_generic_read_hres_clock();
    }

    return ucs_arch_x86_read_tsc();
}

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

static inline void *ucs_memcpy_relaxed(void *dst, const void *src, size_t len)
{
#if ENABLE_BUILTIN_MEMCPY
    if (ucs_unlikely((len > ucs_global_opts.arch.builtin_memcpy_min) &&
                     (len < ucs_global_opts.arch.builtin_memcpy_max))) {
        asm volatile ("rep movsb"
                      : "=D" (dst),
                      "=S" (src),
                      "=c" (len)
                      : "0" (dst),
                      "1" (src),
                      "2" (len)
                      : "memory");
        return dst;
    }
#endif
    return memcpy(dst, src, len);
}

static UCS_F_ALWAYS_INLINE void
ucs_memcpy_nontemporal(void *dst, const void *src, size_t len)
{
    ucs_x86_memcpy_sse_movntdqa(dst, src, len);
}

END_C_DECLS

#endif

