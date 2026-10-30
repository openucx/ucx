/**
* Copyright (C) Loongson Technology Corporation Limited, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_LOONGARCH64_CPU_H_
#define UCS_ARCH_LOONGARCH64_CPU_H_

#include <ucs/arch/generic/cpu.h>
#include <ucs/config/global_opts.h>
#include <ucs/config/types.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/compiler_def.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

BEGIN_C_DECLS

/** @file cpu.h */

#define UCS_ARCH_CACHE_LINE_SIZE 64

/*
 * LoongArch `dbar` (data barrier) hint encoding, as defined by the LoongArch
 * architecture manual and the Linux kernel (arch/loongarch/include/asm/barrier.h):
 *
 *   Bit4: ordering or completion (0: completion, 1: ordering)
 *   Bit3: barrier for previous read   (0: true, 1: false)
 *   Bit2: barrier for previous write  (0: true, 1: false)
 *   Bit1: barrier for succeeding read (0: true, 1: false)
 *   Bit0: barrier for succeeding write(0: true, 1: false)
 */
#define ucs_loongarch64_dbar(_hint) \
    asm volatile ("dbar %0" : : "I"(_hint) : "memory")

/* completion barriers (rmb/wmb), for I/O and device ordering */
#define ucs_memory_bus_store_fence() ucs_loongarch64_dbar(0x0a) /* wmb */
#define ucs_memory_bus_load_fence()  ucs_loongarch64_dbar(0x05) /* rmb */

/* ordering barriers (smp mb/rmb/wmb), for inter-CPU ordering */
#define ucs_memory_cpu_fence()              ucs_loongarch64_dbar(0x10) /* mb */
#define ucs_memory_cpu_store_fence()        ucs_loongarch64_dbar(0x1a) /* wmb */
#define ucs_memory_cpu_load_fence()         ucs_loongarch64_dbar(0x15) /* rmb */

/* completion barriers (iob), for device and write-combining memory */
#define ucs_memory_bus_cacheline_wc_flush() ucs_loongarch64_dbar(0)    /* iob */
#define ucs_memory_cpu_wc_fence()           ucs_loongarch64_dbar(0)    /* iob */

static inline uint32_t ucs_loongarch64_cpucfg(uint32_t index)
{
    uint32_t val;

    asm volatile ("cpucfg %0, %1" : "=r"(val) : "r"(index));

    return val;
}

static inline double ucs_arch_get_clocks_per_sec()
{
    uint32_t cc_cfg2 = ucs_loongarch64_cpucfg(0x2);
    uint32_t cc_freq = ucs_loongarch64_cpucfg(0x4);
    uint32_t cc_cfg5 = ucs_loongarch64_cpucfg(0x5);
    uint64_t cfm;
    uint64_t cfd;

    /* Fall back to the generic clock if the Stable Counter is not present
     * (CPUCFG.2 LLFTP) or the frequency fields are zero, matching the Linux
     * kernel's calc_const_freq(). */
    if (!(cc_cfg2 & UCS_BIT(14))) {
        return ucs_arch_generic_get_clocks_per_sec();
    }

    cfm = cc_cfg5 & 0xffff;
    cfd = (cc_cfg5 >> 16) & 0xffff;

    if ((cc_freq == 0) || (cfm == 0) || (cfd == 0)) {
        return ucs_arch_generic_get_clocks_per_sec();
    }

    /*
     * The Stable Counter (constant frequency timer) ticks at
     * CC_FREQ * CC_MUL / CC_DIV Hz, where CC_FREQ is the crystal oscillator
     * frequency in CPUCFG.4 and CC_MUL/CC_DIV are in CPUCFG.5. This matches
     * the Linux kernel's calc_const_freq() in arch/loongarch.
     */
    return (double)(cc_freq * cfm / cfd);
}

static inline ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    return UCS_CPU_MODEL_LOONGARCH64;
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
    uint64_t ticks;

    asm volatile ("rdtime.d %0, $zero" : "=r"(ticks));

    return ticks;
}

#define ucs_arch_wait_mem ucs_arch_generic_wait_mem

#if !HAVE___CLEAR_CACHE
static inline void ucs_arch_clear_cache(void *start, void *end)
{
    /*
     * LoongArch keeps the instruction and data caches coherent in hardware, so
     * a full data barrier followed by an instruction barrier is enough to make
     * self-modifying code visible to the instruction fetch unit.
     */
    asm volatile ("dbar 0\n\t"
                  "ibar 0\n\t" ::: "memory");
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

static UCS_F_ALWAYS_INLINE void ucs_cpu_relax()
{
    asm volatile ("" ::: "memory");
}

END_C_DECLS

#endif
