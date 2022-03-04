/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RISCV64_ATOMIC_H_
#define UCS_RISCV64_ATOMIC_H_

#define UCS_DEFINE_ATOMIC_ADD(wordsize, suffix) \
    static inline void ucs_atomic_add##wordsize(volatile uint##wordsize##_t *ptr, \
                                                uint##wordsize##_t value) { \
        asm volatile ( \
              "amoadd.w.aqrl %1, %0, 0(s1)" \
              : "+m"(*ptr) \
              : "ir" (value)); \
    }

#define UCS_DEFINE_ATOMIC_FADD(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_fadd##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        asm volatile ( \
              "amoadd.w.aqrl %0, %1, 0(s1)" \
              : "+r" (value), "+m" (*ptr) \
              : : "memory"); \
        return value; \
    }

#define UCS_DEFINE_ATOMIC_SWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_swap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        asm volatile ( \
              "amoswap.w.aqrl %0, %1, 0(s1)" \
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
            "cas_riscv64_ucx__:                      \n\t\
               lr.w %1, (%0)                         \n\t\
               bne %1, %2, cas_fail_riscv64_ucx__    \n\t\
	       sc.w %1, %3, (%0)                     \n\t\
               bnez %1, cas_riscv64_ucx__            \n\t\
	       li %0, 0                              \n\t\
	    cas_fail_riscv64_ucx__:                  \n\t\
	       li %0, 1                              \n\t\
	       " \
              : "=r" (prev) \
              : "r"(swap), "m"(*ptr), "i"(&compare) \
              : "memory"); \
        return prev; \
    }

#define UCS_DEFINE_ATOMIC_BOOL_CSWAP(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_bool_cswap##wordsize(volatile uint##wordsize##_t *ptr, \
                                                                     uint##wordsize##_t compare, \
                                                                     uint##wordsize##_t swap) { \
        return ucs_atomic_cswap##wordsize(ptr, compare, swap) == compare; \
    }

#endif
