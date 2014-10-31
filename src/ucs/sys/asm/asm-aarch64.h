/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#ifndef UCS_ASM_AARCH64_H_
#define UCS_ASM_AARCH64_H_

#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <sys/times.h>


#define UCS_SYS_CACHE_LINE_SIZE    64

/**
 * Assume the worst - weak memory ordering.
 */
#define ucs_memory_bus_fence()        ucs_fatal("unimplemented");
#define ucs_memory_bus_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_bus_load_fence()   ucs_memory_bus_fence()
#define ucs_memory_cpu_fence()        ucs_memory_bus_fence()
#define ucs_memory_cpu_store_fence()  ucs_memory_bus_fence()
#define ucs_memory_cpu_load_fence()   ucs_memory_bus_fence()


static inline clock_t ucs_arch_read_hres_clock(void)
{
    struct tms accurate_clock;
    times(&accurate_clock);
    return accurate_clock.tms_utime + accurate_clock.tms_stime;
}

static inline double ucs_arch_get_clocks_per_sec()
{
    return CLOCKS_PER_SEC;
}

#define UCS_DEFINE_ATOMIC_ADD(wordsize, suffix) \
    static inline void ucs_atomic_add##wordsize(volatile uint##wordsize##_t *ptr, \
                                                uint##wordsize##_t value) { \
        ucs_fatal("unimplemented"); \
    }

#define UCS_DEFINE_ATOMIC_FADD(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_fadd##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        ucs_fatal("unimplemented"); \
        return 0; \
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
        ucs_fatal("unimplemented"); \
        return 0; \
    }

static inline unsigned ucs_ffs64(uint64_t n)
{
    ucs_fatal("unimplemented");
    return 64;
}

static inline unsigned __ucs_ilog2_u32(uint32_t n)
{
    ucs_fatal("unimplemented");
    return 31;
}

static inline unsigned __ucs_ilog2_u64(uint64_t n)
{
    ucs_fatal("unimplemented");
    return 63;
}

#endif
