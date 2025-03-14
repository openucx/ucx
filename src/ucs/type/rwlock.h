/*
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RWLOCK_H
#define UCS_RWLOCK_H

#include <ucs/arch/atomic.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/compiler_def.h>
#include <errno.h>

/**
 * The ucs_rw_spinlock_t type.
 *
 * Readers increment the counter by UCS_RWLOCK_READ (4)
 * Writers set the UCS_RWLOCK_WRITE bit when lock is held
 * and set the UCS_RWLOCK_WAIT bit while waiting.
 *
 * 31                 2 1 0
 * +-------------------+-+-+
 * |  readers          | | |
 * +-------------------+-+-+
 *                      ^ ^
 *                      | |
 * WRITE: lock held ----/ |
 * WAIT: writer pending --/
 */

#define UCS_RWLOCK_WAIT  UCS_BIT(0) /* Writer is waiting */
#define UCS_RWLOCK_WRITE UCS_BIT(1) /* Writer has the lock */
#define UCS_RWLOCK_MASK  (UCS_RWLOCK_WAIT | UCS_RWLOCK_WRITE)
#define UCS_RWLOCK_READ  UCS_BIT(2) /* Reader increment */

#define UCS_RWLOCK_STATIC_INITIALIZER {0}


/**
 * Reader-writer spin lock.
 */
typedef struct {
    int state;
} ucs_rw_spinlock_t;


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_read_lock(ucs_rw_spinlock_t *lock)
{
    int x;

    for (;;) {
        while (ucs_atomic_get(&lock->state, 0) & UCS_RWLOCK_MASK) {
            ucs_cpu_relax();
        }

        x = ucs_atomic_fadd(&lock->state, UCS_RWLOCK_READ,
                            UCS_ATOMIC_FENCE_LOCK);
        if (!(x & UCS_RWLOCK_MASK)) {
            return;
        }

        ucs_atomic_sub(&lock->state, UCS_RWLOCK_READ, 0);
    }
}


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_read_unlock(ucs_rw_spinlock_t *lock)
{
    ucs_assert(lock->state >= UCS_RWLOCK_READ);
    ucs_atomic_sub(&lock->state, UCS_RWLOCK_READ, UCS_ATOMIC_FENCE_UNLOCK);
}


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_write_lock(ucs_rw_spinlock_t *lock)
{
    int x;

    for (;;) {
        x = ucs_atomic_get(&lock->state, 0);
        if ((x < UCS_RWLOCK_WRITE) &&
            ucs_atomic_cswap(&lock->state, x, UCS_RWLOCK_WRITE,
                             UCS_ATOMIC_FENCE_LOCK)) {
            return;
        }

        if (!(x & UCS_RWLOCK_WAIT)) {
            ucs_atomic_or(&lock->state, UCS_RWLOCK_WAIT, 0);
        }

        while (ucs_atomic_get(&lock->state, 0) > UCS_RWLOCK_WAIT) {
            ucs_cpu_relax();
        }
    }
}


static UCS_F_ALWAYS_INLINE int
ucs_rw_spinlock_write_trylock(ucs_rw_spinlock_t *lock)
{
    int x;

    x = ucs_atomic_get(&lock->state, 0);
    if ((x < UCS_RWLOCK_WRITE) &&
        ucs_atomic_cswap(&lock->state, x, x + UCS_RWLOCK_WRITE,
                         UCS_ATOMIC_FENCE_LOCK | UCS_ATOMIC_WEAK)) {
        return 1;
    }

    return 0;
}


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_write_unlock(ucs_rw_spinlock_t *lock)
{
    ucs_assert(lock->state >= UCS_RWLOCK_WRITE);
    ucs_atomic_sub(&lock->state, UCS_RWLOCK_WRITE, UCS_ATOMIC_FENCE_UNLOCK);
}


static UCS_F_ALWAYS_INLINE void ucs_rw_spinlock_init(ucs_rw_spinlock_t *lock)
{
    lock->state = 0;
}


static UCS_F_ALWAYS_INLINE void ucs_rw_spinlock_cleanup(ucs_rw_spinlock_t *lock)
{
    ucs_assert(lock->state == 0);
}

#endif
