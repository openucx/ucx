/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_ASYNC_SIGNAL_H
#define UCS_ASYNC_SIGNAL_H


#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <pthread.h>


typedef struct ucs_async_signal_context {
    pid_t               tid;         /* Thread ID to receive the signal */
    int                 block_count; /* How many times this context is blocked */
    timer_t             timer;
} ucs_async_signal_context_t;


#define UCS_ASYNC_SIGNAL_BLOCK(_async) \
    { \
        ucs_assert((_async)->signal.tid == ucs_get_tid()); \
        ++(_async)->signal.block_count; \
        ucs_compiler_fence(); \
    }

#define UCS_ASYNC_SIGNAL_UNBLOCK(_async) \
    { \
        ucs_memory_cpu_fence(); \
        --(_async)->signal.block_count; \
    }

#endif
