/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RV64_CPU_H_
#define UCS_RV64_CPU_H_

#include <ucs/sys/compiler.h>
#include <ucs/arch/generic/cpu.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <ucs/config/global_opts.h>
#include <stdint.h>
#include <string.h>
#include <sys/cachectl.h>
#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <sys/mman.h>
#include <assert.h>

BEGIN_C_DECLS

/** @file cpu.h */

#define UCS_ARCH_CACHE_LINE_SIZE 64

/**
 * In rv64_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 *
 * The rv_64 specification from 2019 maps rv64 asm operations below to arm in a
 * reference table - to emphasize this mapping - the macros *dmb, *isb, *dsb
 * are defined with respect to arm. 
 *
 */
#define ucs_rv64_dmb()          asm volatile ("fence rw,rw" ::: "memory")
#define ucs_rv64_isb()          asm volatile ("fence.i; fence r,r" ::: "memory")
#define ucs_rv64_dsb()          __sync_synchronize()

#define ucs_acquire_barrier()   asm volatile ("fence r,rw" ::: "memory")
#define ucs_release_barrier()   asm volatile ("fence rw,w" ::: "memory")

/*
 * `__NR_riscv_flush_icache` maps to `sys_riscv_flush_icache`
 * 
 * kernel says flag is 1
 */
#define ucs_rv64_icache_flush(addr_beg, addr_end) syscall(SYS_riscv_flush_icache, addr_beg, addr_end, 1)
#define ucs_rv64_dcache_flush(addr_beg, addr_end) __builtin___clear_cache(addr_beg, addr_end)

#define ucs_memory_bus_store_fence()  ucs_rv64_dmb()
#define ucs_memory_bus_load_fence()   ucs_rv64_dmb()

#define ucs_memory_bus_fence()        asm volatile ("fence rw,rw" ::: "memory")

#define ucs_memory_bus_cacheline_wc_flush()     ucs_rv64_dmb()
#define ucs_memory_cpu_fence()                  ucs_rv64_dmb()
#define ucs_memory_cpu_store_fence()            ucs_rv64_dmb()
#define ucs_memory_cpu_load_fence()             ucs_rv64_dmb()
#define ucs_memory_cpu_wc_fence()               ucs_rv64_dmb()

static inline double ucs_arch_get_clocks_per_sec()
{
    uint64_t freq;
    asm volatile("rdcycle %0" : "=r" (freq));
    return (double) freq;
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
    uint64_t value;
    asm volatile ("rdtime %0" : "=r" (value) );
    return value;
}

#define ucs_arch_wait_mem ucs_arch_generic_wait_mem

static inline void ucs_arch_clear_cache(void * start, void * end) {
    ucs_acquire_barrier();
    ucs_rv64_dcache_flush(start, end);
    ucs_rv64_icache_flush(start, end);
    ucs_release_barrier();
    asm volatile("" ::: "memory");
    __builtin_prefetch(start, 0, 3);
}

static inline void *ucs_memcpy_relaxed(void *dst, const void *src, size_t len)
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
