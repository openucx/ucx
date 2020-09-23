/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_ATOMIC_H
#define UCS_ARCH_ATOMIC_H

#include <stdint.h>

#if defined(__x86_64__)
#  include "x86_64/atomic.h"
#elif defined(__powerpc64__)
#  include "generic/atomic.h"
#elif defined(__aarch64__)
#  include "generic/atomic.h"
#else
#  error "Unsupported architecture"
#endif

#define UCS_DEFINE_ATOMIC_AND(_wordsize, _suffix) \
    static inline void ucs_atomic_and##_wordsize(volatile uint##_wordsize##_t *ptr, \
                                                 uint##_wordsize##_t value) { \
        __sync_and_and_fetch(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_FAND(_wordsize, _suffix) \
    static inline uint##_wordsize##_t ucs_atomic_fand##_wordsize(volatile uint##_wordsize##_t *ptr, \
                                                                 uint##_wordsize##_t value) { \
        return __sync_fetch_and_and(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_XOR(_wordsize, _suffix) \
    static inline void ucs_atomic_xor##_wordsize(volatile uint##_wordsize##_t *ptr, \
                                                 uint##_wordsize##_t value) { \
        __sync_xor_and_fetch(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_FXOR(_wordsize, _suffix) \
    static inline uint##_wordsize##_t ucs_atomic_fxor##_wordsize(volatile uint##_wordsize##_t *ptr, \
                                                                 uint##_wordsize##_t value) { \
        return __sync_fetch_and_xor(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_OR(_wordsize, _suffix) \
    static inline void ucs_atomic_or##_wordsize(volatile uint##_wordsize##_t *ptr, \
                                                uint##_wordsize##_t value) { \
        __sync_or_and_fetch(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_FOR(_wordsize, _suffix) \
    static inline uint##_wordsize##_t ucs_atomic_for##_wordsize(volatile uint##_wordsize##_t *ptr, \
                                                                uint##_wordsize##_t value) { \
        return __sync_fetch_and_or(ptr, value); \
    }

#define UCS_DEFINE_ATOMIC_SUB(wordsize, suffix) \
    static inline void ucs_atomic_sub##wordsize(volatile uint##wordsize##_t *ptr, \
                                                uint##wordsize##_t value) { \
        ucs_atomic_add##wordsize(ptr, (uint##wordsize##_t)-value); \
    }

#define UCS_DEFINE_ATOMIC_FSUB(wordsize, suffix) \
    static inline uint##wordsize##_t ucs_atomic_fsub##wordsize(volatile uint##wordsize##_t *ptr, \
                                                               uint##wordsize##_t value) { \
        return ucs_atomic_fadd##wordsize(ptr, (uint##wordsize##_t)-value); \
    }

/*
 * Define atomic functions
 */
UCS_DEFINE_ATOMIC_ADD(8,  b);
UCS_DEFINE_ATOMIC_ADD(16, w);
UCS_DEFINE_ATOMIC_ADD(32, l);
UCS_DEFINE_ATOMIC_ADD(64, q);

UCS_DEFINE_ATOMIC_FADD(8,  b);
UCS_DEFINE_ATOMIC_FADD(16, w);
UCS_DEFINE_ATOMIC_FADD(32, l);
UCS_DEFINE_ATOMIC_FADD(64, q);

UCS_DEFINE_ATOMIC_SUB(8,  b);
UCS_DEFINE_ATOMIC_SUB(16, w);
UCS_DEFINE_ATOMIC_SUB(32, l);
UCS_DEFINE_ATOMIC_SUB(64, q);

UCS_DEFINE_ATOMIC_FSUB(8,  b);
UCS_DEFINE_ATOMIC_FSUB(16, w);
UCS_DEFINE_ATOMIC_FSUB(32, l);
UCS_DEFINE_ATOMIC_FSUB(64, q);

UCS_DEFINE_ATOMIC_AND(8,  b);
UCS_DEFINE_ATOMIC_AND(16, w);
UCS_DEFINE_ATOMIC_AND(32, l);
UCS_DEFINE_ATOMIC_AND(64, q);

UCS_DEFINE_ATOMIC_FAND(8,  b);
UCS_DEFINE_ATOMIC_FAND(16, w);
UCS_DEFINE_ATOMIC_FAND(32, l);
UCS_DEFINE_ATOMIC_FAND(64, q);

UCS_DEFINE_ATOMIC_OR(8,  b);
UCS_DEFINE_ATOMIC_OR(16, w);
UCS_DEFINE_ATOMIC_OR(32, l);
UCS_DEFINE_ATOMIC_OR(64, q);

UCS_DEFINE_ATOMIC_FOR(8,  b);
UCS_DEFINE_ATOMIC_FOR(16, w);
UCS_DEFINE_ATOMIC_FOR(32, l);
UCS_DEFINE_ATOMIC_FOR(64, q);

UCS_DEFINE_ATOMIC_XOR(8,  b);
UCS_DEFINE_ATOMIC_XOR(16, w);
UCS_DEFINE_ATOMIC_XOR(32, l);
UCS_DEFINE_ATOMIC_XOR(64, q);

UCS_DEFINE_ATOMIC_FXOR(8,  b);
UCS_DEFINE_ATOMIC_FXOR(16, w);
UCS_DEFINE_ATOMIC_FXOR(32, l);
UCS_DEFINE_ATOMIC_FXOR(64, q);

UCS_DEFINE_ATOMIC_SWAP(8,  b);
UCS_DEFINE_ATOMIC_SWAP(16, w);
UCS_DEFINE_ATOMIC_SWAP(32, l);
UCS_DEFINE_ATOMIC_SWAP(64, q);

UCS_DEFINE_ATOMIC_CSWAP(8,  b);
UCS_DEFINE_ATOMIC_CSWAP(16, w);
UCS_DEFINE_ATOMIC_CSWAP(32, l);
UCS_DEFINE_ATOMIC_CSWAP(64, q);

UCS_DEFINE_ATOMIC_BOOL_CSWAP(8,  b);
UCS_DEFINE_ATOMIC_BOOL_CSWAP(16, w);
UCS_DEFINE_ATOMIC_BOOL_CSWAP(32, l);
UCS_DEFINE_ATOMIC_BOOL_CSWAP(64, q);

#endif
