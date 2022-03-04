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

BEGIN_C_DECLS

/** @file cpu.h */

#define UCS_ARCH_CACHE_LINE_SIZE 64

/**
 * In rv64_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 */
#define ucs_memory_bus_fence()        asm volatile ("fence rw,rw"::: "memory")
#define ucs_memory_bus_store_fence()  asm volatile ("fence w,w" ::: "memory")
#define ucs_memory_bus_load_fence()   asm volatile ("fence r,r" ::: "memory")
#define ucs_memory_bus_cacheline_wc_flush()
#define ucs_memory_cpu_fence()        ucs_compiler_fence()
#define ucs_memory_cpu_store_fence()  ucs_compiler_fence()
#define ucs_memory_cpu_load_fence()   ucs_compiler_fence()
#define ucs_memory_cpu_wc_fence()     asm volatile ("fence w,w" ::: "memory")

static inline double ucs_arch_get_clocks_per_sec()
{
    uint64_t freq;
    asm volatile("rdcycle %0" : "=r" (freq));
    return (double) freq;
}


//double ucs_rv64_init_tsc_freq();

typedef struct ucs_rv64_cpuid {
	 char isa[256];
	 char mmu[256];
	 char uarch[256];
} ucs_rv64_cpuid_t;

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
