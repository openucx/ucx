/*
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_RWLOCK_H
#define UCS_RWLOCK_H

#include <sched.h>
#include <errno.h>

/**
 * The ucs_rwlock_t type.
 *
 * Readers increment the counter by UCS_RWLOCK_READ (4)
 * Writers set the UCS_RWLOCK_WRITE bit when lock is held
 *     and set the UCS_RWLOCK_WAIT bit while waiting.
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

#define UCS_RWLOCK_WAIT  0x1 /* Writer is waiting */
#define UCS_RWLOCK_WRITE 0x2 /* Writer has the lock */
#define UCS_RWLOCK_MASK  (UCS_RWLOCK_WAIT | UCS_RWLOCK_WRITE)
                             /* Writer is waiting or has lock */
#define UCS_RWLOCK_READ  0x4 /* Reader increment */


/**
 * Read-write lock.
 */
typedef struct {
    volatile int l;
} ucs_rwlock_t;


static inline void ucs_rwlock_read_lock(ucs_rwlock_t *lock) {
    int x;

    while (1) {
        while (lock->l & UCS_RWLOCK_MASK) {
            sched_yield();
        }

        x = __sync_fetch_and_add(&lock->l, UCS_RWLOCK_READ);
        if (!(x & UCS_RWLOCK_MASK)) {
            return;
        }

        __sync_fetch_and_sub(&lock->l, UCS_RWLOCK_READ);
    }
}


static inline void ucs_rwlock_read_unlock(ucs_rwlock_t *lock) {
    __sync_fetch_and_sub(&lock->l, UCS_RWLOCK_READ);
}


static inline void ucs_rwlock_write_lock(ucs_rwlock_t *lock) {
    int x;

    while (1) {
        x = lock->l;
        if ((x < UCS_RWLOCK_WRITE) &&
            (__sync_val_compare_and_swap(&lock->l, x, UCS_RWLOCK_WRITE) == x)) {
            return;
        }

        __sync_fetch_and_or(&lock->l, UCS_RWLOCK_WAIT);
        while (lock->l > UCS_RWLOCK_WAIT) {
            sched_yield();
        }
    }
}


static inline int ucs_rwlock_write_trylock(ucs_rwlock_t *lock) {
    int x;

    x = lock->l;
    if ((x < UCS_RWLOCK_WRITE) &&
        (__sync_val_compare_and_swap(&lock->l, x, UCS_RWLOCK_WRITE) == x)) {
        return 0;
    }

    return -EBUSY;
}


static inline void ucs_rwlock_write_unlock(ucs_rwlock_t *lock) {
    __sync_fetch_and_sub(&lock->l, UCS_RWLOCK_WRITE);
}


static inline void ucs_rwlock_init(ucs_rwlock_t *lock) {
    lock->l = 0;
}

#endif
