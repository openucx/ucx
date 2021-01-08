/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2020.  ALL RIGHTS RESERVED.
* Copyright (C) Stony Brook University. 2016-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_AARCH64_CPU_H_
#define UCS_AARCH64_CPU_H_

#include "config.h"
#include <time.h>
#include <string.h>
#include <sys/times.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/arch/generic/cpu.h>
#include <ucs/sys/math.h>
#include <ucs/type/status.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif


#define UCS_ARCH_CACHE_LINE_SIZE 64

BEGIN_C_DECLS

/** @file cpu.h */

/**
 * Assume the worst - weak memory ordering.
 */

#define ucs_aarch64_dmb(_op)          asm volatile ("dmb " #_op ::: "memory")
#define ucs_aarch64_isb(_op)          asm volatile ("isb " #_op ::: "memory")
#define ucs_aarch64_dsb(_op)          asm volatile ("dsb " #_op ::: "memory")

/* The macro is used to serialize stores across Normal NC (or Device) and WB
 * memory, (see Arm Spec, B2.7.2).  Based on recent changes in Linux kernel:
 * https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=22ec71615d824f4f11d38d0e55a88d8956b7e45f
 *
 * The underlying barrier code was changed to use lighter weight DMB instead
 * of DSB. The barrier used for synchronization of access between write back
 * and device mapped memory (PCIe BAR).
 */
#define ucs_memory_bus_fence()        ucs_aarch64_dmb(oshsy)
#define ucs_memory_bus_store_fence()  ucs_aarch64_dmb(oshst)
#define ucs_memory_bus_load_fence()   ucs_aarch64_dmb(oshld)

/* The macro is used to flush all pending stores from write combining buffer.
 * Some uarch "auto" flush the stores once cache line is full (no need for additional barrier).
 */
#if defined(HAVE_AARCH64_THUNDERX2)
#define ucs_memory_bus_cacheline_wc_flush()
#else
/* The macro is used to flush stores to Normal NC or Device memory */
#define ucs_memory_bus_cacheline_wc_flush()     ucs_aarch64_dmb(oshst)
#endif

#define ucs_memory_cpu_fence()        ucs_aarch64_dmb(ish)
#define ucs_memory_cpu_store_fence()  ucs_aarch64_dmb(ishst)
#define ucs_memory_cpu_load_fence()   ucs_aarch64_dmb(ishld)

/* The macro is used to serialize stores to Normal NC or Device memory
 * (see Arm Spec, B2.7.2)
 */
#define ucs_memory_cpu_wc_fence()     ucs_aarch64_dmb(oshst)


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


#if defined(HAVE_AARCH64_THUNDERX2)
extern void *__memcpy_thunderx2(void *, const void *, size_t);
#endif


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

static inline ucs_cpu_vendor_t ucs_arch_get_cpu_vendor()
{
    ucs_aarch64_cpuid_t cpuid;
    ucs_aarch64_cpuid(&cpuid);

    if ((cpuid.implementer == 0x46) && (cpuid.architecture == 8)) {
        return UCS_CPU_VENDOR_FUJITSU_ARM;
    }

    return UCS_CPU_VENDOR_GENERIC_ARM;
}

static inline int ucs_arch_get_cpu_flag()
{
    return UCS_CPU_FLAG_UNKNOWN;
}

static inline void ucs_cpu_init()
{
}

static inline void ucs_arch_wait_mem(void *address)
{
    unsigned long tmp;
    asm volatile ("ldaxrb %w0, [%1] \n"
                  "wfe           \n"
                  : "=&r"(tmp)
                  : "r"(address)
                  : "memory");
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
    unsigned dic;
    unsigned idc;
    unsigned ctr_el0;

    /* Get cache line size, using ctr_el0 register
     *
     * Bits    Name      Function
     * *****************************
     * [31]    -         Reserved, RES1.
     * [30]    -         Reserved, RES0.
     * [29]    DIC       Instruction cache invalidation requirements for data to instruction
     *                   coherence.
     * [28]    IDC       Data cache clean requirements for instruction to data coherence.
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
    dic = (ctr_el0 >> 29) & 0x1;
    idc = (ctr_el0 >> 28) & 0x1;

    /* 
     * Check if Data cache clean to the Point of Unification is required for instruction to
     * data coherence
     */
    if (idc == 0) {
        for (ptr = ucs_align_down((uintptr_t)start, dcache); ptr < (uintptr_t)end; ptr += dcache) {
            asm volatile ("dc cvau, %0" :: "r" (ptr) : "memory");
        }
    }

    /*
     * Check if Instruction cache invalidation to the Point of Unification is required for
     * data to instruction coherence.
     */
    if (dic == 0) {
        ucs_aarch64_dsb(ish);
        for (ptr = ucs_align_down((uintptr_t)start, icache); ptr < (uintptr_t)end; ptr += icache) {
            asm volatile ("ic ivau, %0" :: "r" (ptr) : "memory");
        }
    }
    ucs_aarch64_dsb(ish);
    ucs_aarch64_isb();
#endif
}
#endif

#if defined(__ARM_FEATURE_SVE)
static inline void *memcpy_aarch64_sve(void *dest, const void *src, size_t len)
{
    uint8_t *dest_u8      = (uint8_t*) dest;
    const uint8_t *src_u8 = (uint8_t*) src;
    uint64_t i            = 0;
    svbool_t pg           = svwhilelt_b8_u64(i, (uint64_t)len);

    do {
        svst1_u8(pg, &dest_u8[i], svld1_u8(pg, &src_u8[i]));
        i += svcntb();
        pg = svwhilelt_b8_u64(i, (uint64_t)len);
    } while (svptest_first(svptrue_b8(), pg));

    return dest;
}
#endif

static inline void *ucs_memcpy_relaxed(void *dst, const void *src, size_t len)
{
#if defined(HAVE_AARCH64_THUNDERX2)
    return __memcpy_thunderx2(dst, src, len);
#elif defined(__ARM_FEATURE_SVE)
    return memcpy_aarch64_sve(dst, src, len);
#else
    return memcpy(dst, src, len);
#endif
}

static UCS_F_ALWAYS_INLINE void
ucs_memcpy_nontemporal(void *dst, const void *src, size_t len)
{
#if defined(HAVE_AARCH64_THUNDERX2)
    __memcpy_thunderx2(dst, src,len);
#elif defined(__ARM_FEATURE_SVE)
    memcpy_aarch64_sve(dst, src, len);
#else
    memcpy(dst, src, len);
#endif

}

static inline ucs_status_t ucs_arch_get_cache_size(size_t *cache_sizes)
{
    return UCS_ERR_UNSUPPORTED;
}

END_C_DECLS

#endif
