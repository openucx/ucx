/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_SPINLOCK_H
#define UCS_SPINLOCK_H

#include <config.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/log.h>

/**
 * Reentrant spinlock.
 */
typedef struct ucs_spinlock {
    pthread_spinlock_t lock;
    int                count;
    pthread_t          owner;
#if ENABLE_DEBUG_DATA
    const char *       file;
    int                line;
#endif
} ucs_spinlock_t;


static inline ucs_status_t ucs_spinlock_init(ucs_spinlock_t *lock)
{
    int ret;

    ret = pthread_spin_init(&lock->lock, 0);
    if (ret != 0) {
        ucs_error("pthread_spin_init() returned %d: %m", ret);
        return UCS_ERR_IO_ERROR;
    }

    lock->count = 0;
    lock->owner = 0xfffffffful;
#if ENABLE_DEBUG_DATA
    lock->file = "";
    lock->line = 0;
#endif
    return UCS_OK;
}

static inline int ucs_spin_is_owner(ucs_spinlock_t *lock, pthread_t self)
{
    return lock->owner == self;
}

static inline void __ucs_spin_lock(ucs_spinlock_t *lock, const char *file, int line)
{
    pthread_t self = pthread_self();

    if (ucs_spin_is_owner(lock, self)) {
        ++lock->count;
        return;
    }

    pthread_spin_lock(&lock->lock);
    ucs_assertv(lock->count == 0, "count=%d owner=0x%lx", lock->count, lock->owner);
    lock->owner = self;
    ++lock->count;
#if ENABLE_DEBUG_DATA
    lock->file = file;
    lock->line = line;
#endif
}
#define ucs_spin_lock(_lock)  __ucs_spin_lock(_lock, __FILE__, __LINE__)

static inline int __ucs_spin_trylock(ucs_spinlock_t *lock, const char *file, int line)
{
    pthread_t self = pthread_self();

    if (ucs_spin_is_owner(lock, self)) {
        ++lock->count;
        return 1;
    }

    if (pthread_spin_trylock(&lock->lock) != 0) {
        return 0;
    }

    lock->owner = self;
    ++lock->count;
#if ENABLE_DEBUG_DATA
    lock->file = file;
    lock->line = line;
#endif
    return 1;
}
#define ucs_spin_trylock(_lock) __ucs_spin_trylock(_lock, __FILE__, __LINE__)

static inline void ucs_spin_unlock(ucs_spinlock_t *lock)
{
    ucs_assert(ucs_spin_is_owner(lock, pthread_self()));

    --lock->count;
    if (lock->count == 0) {
        lock->owner = 0xfffffffful;
        pthread_spin_unlock(&lock->lock);
    }
}

#endif
