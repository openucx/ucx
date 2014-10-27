/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#ifndef UCS_ASM_PPC64_H_
#define UCS_ASM_PPC64_H_

#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#ifdef HAVE_SYS_PLATFORM_PPC_H
#include <sys/platform/ppc.h>
#else
/* Read the Time Base Register.   */
static inline uint64_t __ppc_get_timebase (void)
{
#ifdef __powerpc64__
    uint64_t __tb;
    /* "volatile" is necessary here, because the user expects this assembly
     *      isn't moved after an optimization.  */
    __asm__ volatile ("mfspr %0, 268" : "=r" (__tb));
    return __tb;
#else  /* not __powerpc64__ */
    uint32_t __tbu, __tbl, __tmp; \
        __asm__ volatile ("0:\n\t"
                "mftbu %0\n\t"
                "mftbl %1\n\t"
                "mftbu %2\n\t"
                "cmpw %0, %2\n\t"
                "bne- 0b"
                : "=r" (__tbu), "=r" (__tbl), "=r" (__tmp));
    return (((uint64_t) __tbu << 32) | __tbl);
#endif  /* not __powerpc64__ */
}

#endif


#define UCS_SYS_CACHE_LINE_SIZE    128

/**
 * Assume the worst - weak memory ordering.
 */
#define ucs_memory_bus_fence()        asm volatile ("sync"::: "memory")
#define ucs_memory_bus_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_bus_load_fence()   ucs_memory_bus_fence()
#define ucs_memory_cpu_fence()        ucs_memory_bus_fence()
#define ucs_memory_cpu_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_cpu_load_fence()   ucs_memory_bus_fence()


static inline uint64_t ucs_arch_read_hres_clock()
{
    return __ppc_get_timebase();
}

static inline double ucs_arch_get_clocks_per_sec()
{
#if HAVE_DECL___PPC_GET_TIMEBASE_FREQ
    return __ppc_get_timebase_freq();
#else
    return ucs_get_cpuinfo_clock_freq("timebase");
#endif
}

#define UCS_DEFINE_ATOMIC_ADD(wordsize, suffix) \
    static inline void ucs_atomic_add##wordsize(volatile uint##wordsize##_t *ptr, \
                                                uint##wordsize##_t value) { \
        __sync_add_and_fetch(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_FADD(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_fadd##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        return __sync_fetch_and_add(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_SWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_swap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        ucs_fatal("unimplemented"); \
        return 0; \
    }

#define UCS_DEFINE_ATOMIC_CSWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_cswap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                                uint##wordsize##_t compare, \
                                                                uint##wordsize##_t swap) { \
        return __sync_val_compare_and_swap(ptr, compare, swap); \
    }

static inline unsigned __ucs_ilog2_u32(uint32_t n)
{
    int bit;
    asm ("cntlzw %0,%1" : "=r" (bit) : "r" (n));
    return 31 - bit;
}

static inline unsigned __ucs_ilog2_u64(uint64_t n)
{
    int bit;
    asm ("cntlzd %0,%1" : "=r" (bit) : "r" (n));
    return 63 - bit;
}

static inline unsigned ucs_ffs64(uint64_t n)
{
    return __ucs_ilog2_u64(n & -n);
}

#endif
