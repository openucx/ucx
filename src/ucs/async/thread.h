/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_THREAD_H
#define UCS_ASYNC_THREAD_H

#include <ucs/type/spinlock.h>
#include <ucs/sys/checker.h>
#include <ucs/debug/assert.h>


typedef struct ucs_async_thread_mutex {
    pthread_mutex_t lock;
#if UCS_ENABLE_ASSERT
    pthread_t       owner;
    unsigned        count;
#endif
} ucs_async_thread_mutex_t;


typedef struct ucs_async_thread_context {
    union {
        ucs_recursive_spinlock_t spinlock;
        ucs_async_thread_mutex_t mutex;
    };
} ucs_async_thread_context_t;


static UCS_F_ALWAYS_INLINE int
ucs_recursive_mutex_is_blocked(const ucs_async_thread_mutex_t *mutex)
{
#if UCS_ENABLE_ASSERT
    return mutex->owner == pthread_self();
#else
    ucs_fatal("must not be called without assertion");
#endif
}

static UCS_F_ALWAYS_INLINE void
ucs_recursive_mutex_block(ucs_async_thread_mutex_t *mutex)
{
    (void)pthread_mutex_lock(&mutex->lock);

#if UCS_ENABLE_ASSERT
    if (mutex->count++ == 0) {
        mutex->owner = pthread_self();
    }
#endif
}

static UCS_F_ALWAYS_INLINE void
ucs_recursive_mutex_unblock(ucs_async_thread_mutex_t *mutex)
{
    ucs_assert(ucs_recursive_mutex_is_blocked(mutex));

#if UCS_ENABLE_ASSERT
    if (--mutex->count == 0) {
        mutex->owner = UCS_ASYNC_PTHREAD_ID_NULL;
    }
#endif

    (void)pthread_mutex_unlock(&mutex->lock);
}

#endif
