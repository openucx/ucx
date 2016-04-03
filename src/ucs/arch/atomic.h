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

UCS_DEFINE_ATOMIC_SWAP(8,  b);
UCS_DEFINE_ATOMIC_SWAP(16, w);
UCS_DEFINE_ATOMIC_SWAP(32, l);
UCS_DEFINE_ATOMIC_SWAP(64, q);

UCS_DEFINE_ATOMIC_CSWAP(8,  b);
UCS_DEFINE_ATOMIC_CSWAP(16, w);
UCS_DEFINE_ATOMIC_CSWAP(32, l);
UCS_DEFINE_ATOMIC_CSWAP(64, q);

#endif
