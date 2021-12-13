/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_SPINLOCK_H
#define UCS_SPINLOCK_H

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/async/async_fwd.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/lock.h>
#include <pthread.h>
#include <errno.h>

BEGIN_C_DECLS

/** @file spinlock.h */


/**
 * Reentrant spinlock.
 */
typedef struct ucs_recursive_spinlock {
    ucs_spinlock_t super;
    int            count;
    pthread_t      owner;
} ucs_recursive_spinlock_t;


void ucs_recursive_spinlock_destroy(ucs_recursive_spinlock_t *lock);

/**
 * Recursive implementation section
 */
static UCS_F_ALWAYS_INLINE void
ucs_recursive_spinlock_init(ucs_recursive_spinlock_t* lock)
{
    lock->count = 0;
    lock->owner = UCS_ASYNC_PTHREAD_ID_NULL;

    ucs_spinlock_init(&lock->super);
}

static UCS_F_ALWAYS_INLINE int
ucs_recursive_spin_is_owner(const ucs_recursive_spinlock_t *lock,
                            pthread_t self)
{
    return lock->owner == self;
}

static UCS_F_ALWAYS_INLINE void
ucs_recursive_spin_lock(ucs_recursive_spinlock_t *lock)
{
    pthread_t self = pthread_self();

    if (ucs_recursive_spin_is_owner(lock, self)) {
        ++lock->count;
        return;
    }

    ucs_spin_lock(&lock->super);
    lock->owner = self;
    ++lock->count;
}

static UCS_F_ALWAYS_INLINE int
ucs_recursive_spin_trylock(ucs_recursive_spinlock_t *lock)
{
    pthread_t self = pthread_self();

    if (ucs_recursive_spin_is_owner(lock, self)) {
        ++lock->count;
        return 1;
    }

    if (ucs_spin_try_lock(&lock->super) == 0) {
        return 0;
    }

    lock->owner = self;
    ++lock->count;
    return 1;
}

static UCS_F_ALWAYS_INLINE void
ucs_recursive_spin_unlock(ucs_recursive_spinlock_t *lock)
{
    --lock->count;
    if (lock->count == 0) {
        lock->owner = UCS_ASYNC_PTHREAD_ID_NULL;
        ucs_spin_unlock(&lock->super);
    }
}

int ucs_spinlock_is_held(ucs_spinlock_t *lock);

int ucs_recursive_spinlock_is_held(const ucs_recursive_spinlock_t *lock);

END_C_DECLS

#endif
