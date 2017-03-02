/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_THREAD_H
#define UCS_ASYNC_THREAD_H

#include <ucs/type/spinlock.h>
#include <ucs/sys/checker.h>


typedef struct ucs_async_thread_context {
    union {
#ifndef NVALGRIND
        pthread_mutex_t mutex;
#endif
        ucs_spinlock_t  spinlock;
    };
} ucs_async_thread_context_t;


#if defined(NVALGRIND) || defined(__COVERITY__)

#define UCS_ASYNC_THREAD_BLOCK(_async) \
    ucs_spin_lock(&(_async)->thread.spinlock)

#define UCS_ASYNC_THREAD_UNBLOCK(_async) \
    ucs_spin_unlock(&(_async)->thread.spinlock)

#else

#define UCS_ASYNC_THREAD_BLOCK(_async) \
    { \
        (RUNNING_ON_VALGRIND) ? \
            (void)pthread_mutex_lock(&(_async)->thread.mutex) : \
            ucs_spin_lock(&(_async)->thread.spinlock); \
    }

#define UCS_ASYNC_THREAD_UNBLOCK(_async) \
    { \
        (RUNNING_ON_VALGRIND) ? \
            (void)pthread_mutex_unlock(&(_async)->thread.mutex) : \
            ucs_spin_unlock(&(_async)->thread.spinlock); \
    }

#endif /* NVALGRIND */

#endif
