/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_ATOMIC_H_
#define UCS_X86_64_ATOMIC_H_


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

#endif
