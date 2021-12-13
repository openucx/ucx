/*
* Copyright (c) 2021 Nvidia Corporation. All Rights Reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_LOCK_H
#define UCS_ARCH_LOCK_H

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/**
 * Spinlock static initializer
 */
#define UCS_SPINLOCK_INITIALIZER {UCS_SPINLOCK_FREE}


/**
 * Spinlock states
 */
enum {
    UCS_SPINLOCK_FREE = 0,
    UCS_SPINLOCK_BUSY = 1
};

/**
 * Simple spinlock.
 */
typedef struct ucs_spinlock {
    volatile unsigned int lock;
} ucs_spinlock_t;


#if defined(__x86_64__)
#  include "x86_64/lock.h"
#elif defined(__powerpc64__)
#  include "ppc64/lock.h"
#elif defined(__aarch64__)
#  include "aarch64/lock.h"
#else
#  error "Unsupported architecture"
#endif


static UCS_F_ALWAYS_INLINE void ucs_spinlock_init(ucs_spinlock_t *lock)
{
    lock->lock = UCS_SPINLOCK_FREE;
}


static UCS_F_ALWAYS_INLINE void ucs_spinlock_destroy(ucs_spinlock_t *lock)
{
}

END_C_DECLS

#endif /* UCS_ARCH_LOCK_H */
