/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RISCV64_CPU_H_
#define UCS_RISCV64_CPU_H_

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
 * In riscv64_64, there is strong ordering of each processor with respect to another
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

double ucs_arch_get_clocks_per_sec();
double ucs_riscv64_init_tsc_freq();

ucs_cpu_model_t ucs_arch_get_cpu_model() UCS_F_NOOPTIMIZE;
ucs_cpu_flag_t ucs_arch_get_cpu_flag() UCS_F_NOOPTIMIZE;
ucs_cpu_vendor_t ucs_arch_get_cpu_vendor();
void ucs_cpu_init();
ucs_status_t ucs_arch_get_cache_size(size_t *cache_sizes);

static inline uint64_t ucs_arch_read_hres_clock()
{
    uint64_t value;
    asm volatile ("rdtime" : "=r" (value));
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
