/*
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RWLOCK_H
#define UCS_RWLOCK_H

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
 * UCS_RWLOCK_WAIT bit is meant for all subsequent reader
 * to let any writer go first to avoid write starvation.
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
    uint32_t state;
} ucs_rw_spinlock_t;


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_read_lock(ucs_rw_spinlock_t *lock)
{
    uint32_t x;

    for (;;) {
        while (__atomic_load_n(&lock->state, __ATOMIC_RELAXED) &
               UCS_RWLOCK_MASK) {
            ucs_cpu_relax();
        }

        x = __atomic_fetch_add(&lock->state, UCS_RWLOCK_READ, __ATOMIC_ACQUIRE);
        if (!(x & UCS_RWLOCK_MASK)) {
            return;
        }

        __atomic_fetch_sub(&lock->state, UCS_RWLOCK_READ, __ATOMIC_RELAXED);
    }
}


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_read_unlock(ucs_rw_spinlock_t *lock)
{
    ucs_assertv(lock->state >= UCS_RWLOCK_READ, "lock underrun state:%u",
                lock->state);
    __atomic_fetch_sub(&lock->state, UCS_RWLOCK_READ, __ATOMIC_RELEASE);
}


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_write_lock(ucs_rw_spinlock_t *lock)
{
    uint32_t x;

    for (;;) {
        x = __atomic_load_n(&lock->state, __ATOMIC_RELAXED);
        if ((x < UCS_RWLOCK_WRITE) &&
            (__atomic_compare_exchange_n(&lock->state, &x, UCS_RWLOCK_WRITE, 0,
                                         __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))) {
            return;
        }

        if (!(x & UCS_RWLOCK_WAIT)) {
            __atomic_fetch_or(&lock->state, UCS_RWLOCK_WAIT, __ATOMIC_RELAXED);
        }

        while (__atomic_load_n(&lock->state, __ATOMIC_RELAXED) > UCS_RWLOCK_WAIT) {
            ucs_cpu_relax();
        }
    }
}


static UCS_F_ALWAYS_INLINE int
ucs_rw_spinlock_write_trylock(ucs_rw_spinlock_t *lock)
{
    uint32_t x;

    x = __atomic_load_n(&lock->state, __ATOMIC_RELAXED);
    if ((x < UCS_RWLOCK_WRITE) &&
        (__atomic_compare_exchange_n(&lock->state, &x, x + UCS_RWLOCK_WRITE, 1,
                                     __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))) {
        return 1;
    }

    return 0;
}


static UCS_F_ALWAYS_INLINE void
ucs_rw_spinlock_write_unlock(ucs_rw_spinlock_t *lock)
{
    ucs_assertv(lock->state >= UCS_RWLOCK_WRITE, "lock underrun state:%u",
                lock->state);
    __atomic_fetch_sub(&lock->state, UCS_RWLOCK_WRITE, __ATOMIC_RELEASE);
}


static UCS_F_ALWAYS_INLINE void ucs_rw_spinlock_init(ucs_rw_spinlock_t *lock)
{
    lock->state = 0;
}


static UCS_F_ALWAYS_INLINE void ucs_rw_spinlock_cleanup(ucs_rw_spinlock_t *lock)
{
    ucs_assertv(lock->state == 0, "lock no released state:%u", lock->state);
}

#endif
