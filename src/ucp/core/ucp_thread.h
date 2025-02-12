/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_THREAD_H_
#define UCP_THREAD_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/async/async.h>
#include <ucs/async/thread.h>
#include <ucs/type/spinlock.h>


/*
 * Multi-thread mode
 */
typedef enum ucp_mt_type {
    UCP_MT_TYPE_NONE = 0,
    UCP_MT_TYPE_SPINLOCK,
    UCP_MT_TYPE_MUTEX,
    UCP_MT_TYPE_WORKER_ASYNC
} ucp_mt_type_t;


/**
 * Multi-thread lock
 */
typedef struct ucp_mt_lock {
    ucp_mt_type_t                 mt_type;
    union {
        /* Lock for multithreading support. Either spinlock or mutex is used at
           at one time. Spinlock is the default option. */
        ucs_recursive_spinlock_t  mt_spinlock;
        pthread_mutex_t           mt_mutex;
        /* Lock for MULTI_THREAD_WORKER case, when mt-single context is used by
         * a single mt-shared worker. In this case the worker progress flow is
         * already protected by worker mutex, and we don't need to lock inside
         * that flow. This is to protect certain API calls that can be triggered
         * from the user thread without holding a worker mutex.
         * Essentially this mutex is a pointer to a worker mutex */
        ucs_async_context_t       *mt_worker_async;
    } lock;
} ucp_mt_lock_t;


#define UCP_THREAD_IS_REQUIRED(_lock_ptr) \
    ((_lock_ptr)->mt_type)
#define UCP_THREAD_LOCK_INIT(_lock_ptr) \
    do { \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_SPINLOCK) { \
            ucs_recursive_spinlock_init(&((_lock_ptr)->lock.mt_spinlock), 0); \
        } else if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) { \
            pthread_mutex_init(&((_lock_ptr)->lock.mt_mutex), NULL); \
        } \
    } while (0)
#define UCP_THREAD_LOCK_FINALIZE(_lock_ptr) \
    do { \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_SPINLOCK) { \
            ucs_recursive_spinlock_destroy(&((_lock_ptr)->lock.mt_spinlock)); \
        } else if ((_lock_ptr)->mt_type == UCP_MT_TYPE_MUTEX) { \
            pthread_mutex_destroy(&((_lock_ptr)->lock.mt_mutex)); \
        } \
    } while (0)

static UCS_F_ALWAYS_INLINE void ucp_mt_lock_lock(ucp_mt_lock_t *lock)
{
    if (lock->mt_type == UCP_MT_TYPE_SPINLOCK) {
        ucs_recursive_spin_lock(&lock->lock.mt_spinlock);
    } else if (lock->mt_type == UCP_MT_TYPE_MUTEX) {
        pthread_mutex_lock(&lock->lock.mt_mutex);
    }
}

static UCS_F_ALWAYS_INLINE void ucp_mt_lock_unlock(ucp_mt_lock_t *lock)
{
    if (lock->mt_type == UCP_MT_TYPE_SPINLOCK) {
        ucs_recursive_spin_unlock(&lock->lock.mt_spinlock);
    } else if (lock->mt_type == UCP_MT_TYPE_MUTEX) {
        pthread_mutex_unlock(&lock->lock.mt_mutex);
    }
}

#define UCP_THREAD_CS_ENTER(_lock_ptr) ucp_mt_lock_lock(_lock_ptr)
#define UCP_THREAD_CS_EXIT(_lock_ptr)  ucp_mt_lock_unlock(_lock_ptr)

#define UCP_THREAD_CS_ASYNC_ENTER(_lock_ptr) \
    do { \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_WORKER_ASYNC) { \
            UCS_ASYNC_BLOCK((_lock_ptr)->lock.mt_worker_async); \
        } else { \
            ucp_mt_lock_lock(_lock_ptr); \
        } \
    } while(0)

#define UCP_THREAD_CS_ASYNC_EXIT(_lock_ptr) \
    do { \
        if ((_lock_ptr)->mt_type == UCP_MT_TYPE_WORKER_ASYNC) { \
            UCS_ASYNC_UNBLOCK((_lock_ptr)->lock.mt_worker_async); \
        } else { \
            ucp_mt_lock_unlock(_lock_ptr); \
        } \
    } while(0)

#endif
