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
#include <pthread.h>
#include <errno.h>

BEGIN_C_DECLS

/** @file spinlock.h */


/* Spinlock static initializer */
#define UCS_SPINLOCK_INITIALIZER {UCS_SPINLOCK_FREE}


/* Spinlock states */
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

/**
 * Reentrant spinlock.
 */
typedef struct ucs_recursive_spinlock {
    ucs_spinlock_t super;
    int            count;
    pthread_t      owner;
} ucs_recursive_spinlock_t;


void ucs_recursive_spinlock_destroy(ucs_recursive_spinlock_t *lock);

/* spinlock implementation section */
static ucs_status_t ucs_spinlock_init(ucs_spinlock_t *lock)
{
    lock->lock = UCS_SPINLOCK_FREE;
    return UCS_OK;
}

static inline void ucs_spinlock_destroy(ucs_spinlock_t *lock) {
}

static UCS_F_ALWAYS_INLINE int ucs_spin_try_lock(ucs_spinlock_t *lock)
{
    return ucs_atomic_cswap32(&lock->lock, UCS_SPINLOCK_FREE,
                              UCS_SPINLOCK_BUSY) == UCS_SPINLOCK_FREE;
}

static UCS_F_ALWAYS_INLINE void ucs_spin_lock(ucs_spinlock_t *lock)
{
    while (!ucs_spin_try_lock(lock)) {
        while (lock->lock != UCS_SPINLOCK_FREE) {
            /* spin */
#ifdef __x86_64__
            asm volatile("pause" ::: "memory");
#elif defined __aarch64__
            asm volatile ("dmb ishld" ::: "memory");
#endif
        }
    }
}

static UCS_F_ALWAYS_INLINE void ucs_spin_unlock(ucs_spinlock_t *lock)
{
    lock->lock = UCS_SPINLOCK_FREE;
}

/* recirsive implementation section */
static inline ucs_status_t
ucs_recursive_spinlock_init(ucs_recursive_spinlock_t* lock)
{
    lock->count = 0;
    lock->owner = UCS_ASYNC_PTHREAD_ID_NULL;

    return ucs_spinlock_init(&lock->super);
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
