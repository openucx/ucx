/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_SIGNAL_H
#define UCS_ASYNC_SIGNAL_H


#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <ucs/sys/sys.h> /* for ucs_get_tid() */
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>


typedef struct ucs_async_signal_context {
    pid_t               tid;         /* Thread ID to receive the signal */
    int                 block_count; /* How many times this context is blocked */
    pthread_t           pthread;     /* Thread ID for pthreads */
    timer_t             timer;
} ucs_async_signal_context_t;


#define UCS_ASYNC_SIGNAL_BLOCK(_async) \
    { \
        ucs_assert((_async)->signal.tid == ucs_get_tid()); \
        ++(_async)->signal.block_count; \
        ucs_memory_cpu_fence(); \
    }

#define UCS_ASYNC_SIGNAL_UNBLOCK(_async) \
    { \
        ucs_memory_cpu_fence(); \
        --(_async)->signal.block_count; \
    }

#ifndef SIGEV_THREAD_ID
#define SIGEV_THREAD_ID 4
#endif

struct ksigevent {
    union sigval sigev_value;
    int sigev_signo;
    int sigev_notify;
    union {
       int _pad[64-sizeof(int) * 2 + sizeof(union sigval)];
       int _tid;
    } _sigev_un;
};

#endif
