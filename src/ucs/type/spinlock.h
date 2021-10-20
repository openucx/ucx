/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_SPINLOCK_H
#define UCS_SPINLOCK_H

#include <ucs/type/status.h>
#include <ucs/async/async_fwd.h>
#include <pthread.h>
#include <errno.h>

BEGIN_C_DECLS

/** @file spinlock.h */


/* Spinlock creation modifiers */
enum {
    UCS_SPINLOCK_FLAG_SHARED = UCS_BIT(0) /**< Make spinlock sharable in memory */
};

/**
 * Simple spinlock.
 */
typedef struct ucs_spinlock {
    pthread_spinlock_t lock;
} ucs_spinlock_t;

/**
 * Reentrant spinlock.
 */
typedef struct ucs_recursive_spinlock {
    ucs_spinlock_t super;
    int            count;
    pthread_t      owner;
} ucs_recursive_spinlock_t;


static ucs_status_t ucs_spinlock_init(ucs_spinlock_t *lock, int flags)
{
    int ret, lock_flags;

    if (flags & UCS_SPINLOCK_FLAG_SHARED) {
        lock_flags = PTHREAD_PROCESS_SHARED;
    } else {
        lock_flags = PTHREAD_PROCESS_PRIVATE;
    }

    ret = pthread_spin_init(&lock->lock, lock_flags);
    if (ret != 0) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static inline ucs_status_t
ucs_recursive_spinlock_init(ucs_recursive_spinlock_t* lock, int flags)
{
    lock->count = 0;
    lock->owner = UCS_ASYNC_PTHREAD_ID_NULL;

    return ucs_spinlock_init(&lock->super, flags);
}

void ucs_spinlock_destroy(ucs_spinlock_t *lock);

void ucs_recursive_spinlock_destroy(ucs_recursive_spinlock_t *lock);

static inline int
ucs_recursive_spin_is_owner(const ucs_recursive_spinlock_t *lock,
                            pthread_t self)
{
    return lock->owner == self;
}

static inline void ucs_spin_lock(ucs_spinlock_t *lock)
{
    pthread_spin_lock(&lock->lock);
}

static inline void ucs_recursive_spin_lock(ucs_recursive_spinlock_t *lock)
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

static inline int ucs_spin_try_lock(ucs_spinlock_t *lock)
{
    if (pthread_spin_trylock(&lock->lock) != 0) {
        return 0;
    }

    return 1;
}

static inline int ucs_recursive_spin_trylock(ucs_recursive_spinlock_t *lock)
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

static inline void ucs_spin_unlock(ucs_spinlock_t *lock)
{
    pthread_spin_unlock(&lock->lock);
}

static inline void ucs_recursive_spin_unlock(ucs_recursive_spinlock_t *lock)
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
