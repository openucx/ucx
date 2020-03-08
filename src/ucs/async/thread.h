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
        ucs_recursive_spinlock_t spinlock;
        pthread_mutex_t          mutex;
    };
} ucs_async_thread_context_t;

#endif
