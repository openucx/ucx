/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_AARCH64_CPU_H_
#define UCS_AARCH64_CPU_H_

#include "config.h"
#include <time.h>
#include <sys/times.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/arch/generic/cpu.h>
#include <ucs/sys/math.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif


#define UCS_ARCH_CACHE_LINE_SIZE 64

BEGIN_C_DECLS

/**
 * Assume the worst - weak memory ordering.
 */
#define ucs_memory_bus_fence()        asm volatile ("dsb sy" ::: "memory");
#define ucs_memory_bus_store_fence()  asm volatile ("dsb st" ::: "memory");
#define ucs_memory_bus_load_fence()   asm volatile ("dsb ld" ::: "memory");

#define ucs_memory_cpu_fence()        asm volatile ("dmb ish" ::: "memory");
#define ucs_memory_cpu_store_fence()  asm volatile ("dmb ishst" ::: "memory");
#define ucs_memory_cpu_load_fence()   asm volatile ("dmb ishld" ::: "memory");


/*
 * ARM processor ID (ARM ISA - Main ID Register, EL1)
 */
typedef struct ucs_aarch64_cpuid {
    int       implementer;
    int       architecture;
    int       variant;
    int       part;
    int       revision;
} ucs_aarch64_cpuid_t;


/**
 * Get ARM CPU identifier and version
 */
void ucs_aarch64_cpuid(ucs_aarch64_cpuid_t *cpuid);


#if HAVE_HW_TIMER
static inline uint64_t ucs_arch_read_hres_clock(void)
{
    uint64_t ticks;
    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r" (ticks));
    return ticks;
}

static inline double ucs_arch_get_clocks_per_sec()
{
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (freq));
    return (double) freq;
}

#else

#define ucs_arch_read_hres_clock ucs_arch_generic_read_hres_clock
#define ucs_arch_get_clocks_per_sec ucs_arch_generic_get_clocks_per_sec

#endif

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_ARM_AARCH64;
}

static inline int ucs_arch_get_cpu_flag()
{
    return UCS_CPU_FLAG_UNKNOWN;
}

static inline void ucs_arch_wait_mem(void *address)
{
    unsigned long tmp;
    asm volatile ("ldxrb %w0, %1 \n"
                  "wfe           \n"
                  : "=&r"(tmp)
                  : "Q"(address));
}

#if !HAVE___CLEAR_CACHE
static inline void ucs_arch_clear_cache(void *start, void *end)
{
#if HAVE___AARCH64_SYNC_CACHE_RANGE
    /* do not allow global declaration of compiler intrinsic */
    void __aarch64_sync_cache_range(void* beg, void* end);

    __aarch64_sync_cache_range(start, end);
#else
    uintptr_t ptr;
    unsigned icache;
    unsigned dcache;
    unsigned ctr_el0;

    /* Get cache line size, using ctr_el0 register
     *
     * Bits    Name      Function
     * *****************************
     * [31]    -         Reserved, res1.
     * [30:28] -         Reserved, res0.
     * [27:24] CWG       Cache Write-Back granule. Log2 of the number of words of the
     *                   maximum size of memory that can be overwritten as a result of
     *                   the eviction of a cache entry that has had a memory location
     *                   in it modified:
     *                   0x4
     *                   Cache Write-Back granule size is 16 words.
     * [23:20] ERG       Exclusives Reservation Granule. Log2 of the number of words of
     *                   the maximum size of the reservation granule that has been
     *                   implemented for the Load-Exclusive and Store-Exclusive instructions:
     *                   0x4
     *                   Exclusive reservation granule size is 16 words.
     * [19:16] DminLine  Log2 of the number of words in the smallest cache line of all the
     *                   data and unified caches that the processor controls:
     *                   0x4
     *                   Smallest data cache line size is 16 words.
     * [15:14] L1lp      L1 Instruction cache policy. Indicates the indexing and tagging
     *                   policy for the L1 Instruction cache:
     *                   0b10
     *                   Virtually Indexed Physically Tagged (VIPT).
     * [13:4]  -         Reserved, res0.
     * [3:0]   IminLine  Log2 of the number of words in the smallest cache line of all
     *                   the instruction caches that the processor controls.
     *                   0x4
     *                   Smallest instruction cache line size is 16 words.
     */
    asm volatile ("mrs\t%0, ctr_el0":"=r" (ctr_el0));
    icache = sizeof(int) << (ctr_el0 & 0xf);
    dcache = sizeof(int) << ((ctr_el0 >> 16) & 0xf);

    for (ptr = ucs_align_down((uintptr_t)start, dcache); ptr < (uintptr_t)end; ptr += dcache) {
        asm volatile ("dc cvau, %0" :: "r" (ptr) : "memory");
    }
    asm volatile ("dsb ish" ::: "memory");

    for (ptr = ucs_align_down((uintptr_t)start, icache); ptr < (uintptr_t)end; ptr += icache) {
        asm volatile ("ic ivau, %0" :: "r" (ptr) : "memory");
    }
    asm volatile ("dsb ish; isb" ::: "memory");
#endif
}
#endif

END_C_DECLS

#endif
