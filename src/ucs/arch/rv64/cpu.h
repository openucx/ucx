/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
* Copyright (C) Rivos Inc. 2023
* Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_RV64_CPU_H_
#define UCS_ARCH_RV64_CPU_H_

#include <ucs/arch/generic/cpu.h>
#include <ucs/config/global_opts.h>
#include <ucs/config/types.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/compiler_def.h>

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <unistd.h>

BEGIN_C_DECLS

/** @file cpu.h */

#define UCS_ARCH_CACHE_LINE_SIZE 64

/*
 * System call for flushing the instruction caches.
 *
 * Need to pass zero to lead to all HARTs (CPUs) to update their caches.
 */
#define ucs_rv64_icache_flush(_start, _end) \
    syscall(SYS_riscv_flush_icache, _start, _end, 0)

#define ucs_memory_bus_store_fence() asm volatile("fence ow, ow" ::: "memory")
#define ucs_memory_bus_load_fence()  asm volatile("fence ir, ir" ::: "memory")

/**
 * The RISC-V memory model is mostly weak. The fence instruction ensures that all
 * HARTs (CPUs) see any stores or loads before the fence before any stores or
 * loads after the fence.
 */

#define ucs_memory_cpu_fence()              asm volatile("fence rw, rw" ::: "memory")
#define ucs_memory_bus_cacheline_wc_flush() ucs_memory_cpu_fence()
#define ucs_memory_cpu_store_fence()        asm volatile("fence rw, w" ::: "memory")
#define ucs_memory_cpu_load_fence()         asm volatile("fence r, rw" ::: "memory")
#define ucs_memory_cpu_wc_fence()           ucs_memory_cpu_fence()

static inline double ucs_arch_get_clocks_per_sec()
{
    return ucs_arch_generic_get_clocks_per_sec();
}

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_RV64G;
}

static inline int ucs_arch_get_cpu_flag()
{
    return UCS_CPU_FLAG_UNKNOWN;
}

static inline void ucs_cpu_init()
{
}

ucs_cpu_vendor_t ucs_arch_get_cpu_vendor();

static inline ucs_status_t ucs_arch_get_cache_size(size_t *cache_sizes)
{
    return UCS_ERR_UNSUPPORTED;
}

static inline uint64_t ucs_arch_read_hres_clock()
{
    return ucs_arch_generic_read_hres_clock();
}

#define ucs_arch_wait_mem ucs_arch_generic_wait_mem

#if !HAVE___CLEAR_CACHE
static inline void ucs_arch_clear_cache(void *start, void *end)
{
    /*
    * The syscall will cause all other HARTs (CPUs) to invalidate their
    * instruction caches. This is the equivalent of the glibc __clear_cache()
    * implementation that ucs_clear_cache() will use if HAVE_CLEAR_CACHE is
    * defined.
    */
    ucs_rv64_icache_flush(start, end);
}
#endif

static inline void *ucs_memcpy_relaxed(void *dst, const void *src, size_t len,
                                       ucs_arch_memcpy_hint_t hint,
                                       size_t total_len)
{
    return memcpy(dst, src, len);
}

static UCS_F_ALWAYS_INLINE void
ucs_memcpy_nontemporal(void *dst, const void *src, size_t len)
{
    memcpy(dst, src, len);
}

END_C_DECLS

#endif
