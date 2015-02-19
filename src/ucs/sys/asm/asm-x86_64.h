/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_ASM_X86_64_H_
#define UCS_ASM_X86_64_H_

#include <ucs/sys/compiler.h>
#include <stdint.h>

#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif
#ifdef __AVX__
#  include <immintrin.h>
#endif


#define UCS_SYS_CACHE_LINE_SIZE    64

/**
 * In x86_64, there is strong ordering of each processor with respect to another
 * processor, but weak ordering with respect to the bus.
 */
#define ucs_memory_bus_fence()        asm volatile ("mfence"::: "memory")
#define ucs_memory_bus_store_fence()  asm volatile ("sfence" ::: "memory")
#define ucs_memory_bus_load_fence()   asm volatile ("lfence" ::: "memory")
#define ucs_memory_cpu_fence()        ucs_compiler_fence()
#define ucs_memory_cpu_store_fence()  ucs_compiler_fence()
#define ucs_memory_cpu_load_fence()   ucs_compiler_fence()


static inline uint64_t ucs_arch_read_hres_clock()
{
    uint32_t low, high;
    asm volatile ("rdtsc" : "=a" (low), "=d" (high));
    return ((uint64_t)high << 32) | (uint64_t)low;
}

static inline double ucs_arch_get_clocks_per_sec()
{
    return ucs_get_cpuinfo_clock_freq("cpu MHz");
}

#define UCS_DEFINE_ATOMIC_ADD(wordsize, suffix) \
    static inline void ucs_atomic_add##wordsize(volatile uint##wordsize##_t *ptr, \
                                                uint##wordsize##_t value) { \
        asm volatile ( \
              "lock add" #suffix " %1, %0" \
              : "+m"(*ptr) \
              : "ir" (value)); \
    }

#define UCS_DEFINE_ATOMIC_FADD(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_fadd##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        asm volatile ( \
              "lock xadd" #suffix " %0, %1" \
              : "+r" (value), "+m" (*ptr) \
              : : "memory"); \
        return value; \
    }

#define UCS_DEFINE_ATOMIC_SWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_swap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        asm volatile ( \
              "lock xchg" #suffix " %0, %1" \
              : "+r" (value), "+m" (*ptr) \
              : : "memory", "cc"); \
        return value; \
    }

#define UCS_DEFINE_ATOMIC_CSWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_cswap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                                uint##wordsize##_t compare, \
                                                                uint##wordsize##_t swap) { \
        unsigned long prev; \
        asm volatile ( \
              "lock cmpxchg" # suffix " %1, %2" \
              : "=a" (prev) \
              : "r"(swap), "m"(*ptr), "0" (compare) \
              : "memory"); \
        return prev; \
    }

static inline unsigned ucs_ffs64(uint64_t n)
{
    uint64_t result;
    asm("bsfq %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

static inline unsigned __ucs_ilog2_u32(uint32_t n)
{
    uint32_t result;
    asm("bsrl %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

static inline unsigned __ucs_ilog2_u64(uint64_t n)
{
    uint64_t result;
    asm("bsrq %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

static inline void ucs_cpuid(unsigned level, unsigned *a, unsigned *b,
                             unsigned *c, unsigned *d)
{
  asm volatile ("cpuid\n\t"
                  : "=a" (*a), "=b" (*b), "=c" (*c), "=d" (*d)
                  : "0" (level));
}

#endif


