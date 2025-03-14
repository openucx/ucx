/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_GENERIC_ATOMIC_H_
#define UCS_GENERIC_ATOMIC_H_


#define UCS_DEFINE_ATOMIC_ADD(wordsize, suffix) \
    static inline void ucs_atomic_add##wordsize(volatile uint##wordsize##_t *ptr, \
                                                uint##wordsize##_t value) { \
        __atomic_add_fetch(ptr, value, __ATOMIC_RELAXED); \
    }

#define UCS_DEFINE_ATOMIC_FADD(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_fadd##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        return __atomic_fetch_add(ptr, value, __ATOMIC_RELAXED); \
    }

#define UCS_DEFINE_ATOMIC_SWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_swap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        uint##wordsize##_t old; \
        do { \
           old = *ptr; \
        } while(!__atomic_compare_exchange_n(ptr, &old, value, /*weak=*/0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)); \
        return old; \
    }

#define UCS_DEFINE_ATOMIC_CSWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_cswap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                                uint##wordsize##_t compare, \
                                                                uint##wordsize##_t swap) { \
        uint##wordsize##_t expected = compare; \
        __atomic_compare_exchange_n(ptr, &expected, swap, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); \
        return expected; \
    }

#define UCS_DEFINE_ATOMIC_BOOL_CSWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_bool_cswap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                                     uint##wordsize##_t compare, \
                                                                     uint##wordsize##_t swap) { \
        return __atomic_compare_exchange_n(ptr, &compare, swap, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); \
    }

#endif
