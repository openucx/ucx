/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ASYNC_INT_H
#define UCS_ASYNC_INT_H

#include "async.h"

#include <ucs/datastruct/queue.h>
#include <ucs/time/timerq.h>


/* Async event handler */
typedef struct ucs_async_handler ucs_async_handler_t;
struct ucs_async_handler {
    int                        id;      /* Event/Timer ID */
    ucs_async_mode_t           mode;    /* Event delivery mode */
    ucs_event_set_types_t      events;  /* Bitmap of events */
    pthread_t                  caller;  /* Thread which invokes the callback */
    ucs_async_event_cb_t       cb;      /* Callback function */
    void                       *arg;    /* Callback argument */
    ucs_async_context_t        *async;  /* Async context for the handler. Can be NULL */
    volatile uint32_t          missed;  /* Protect against adding to miss queue multiple times */
    volatile uint32_t          refcount;
};


/**
 * Dispatch event coming from async context.
 *
 * @param handler_ids   Array of handler IDs to dispatch.
 * @param count         Number of events
 * @param events        Events to pass to the handler
 */
ucs_status_t ucs_async_dispatch_handlers(int *handler_ids, size_t count,
                                         ucs_event_set_types_t events);


/**
 * Dispatch timers from a timer queue.
 *
 * @param timerq        Timer queue whose timers to dispatch.
 * @param current_time  Current time for checking timer expiration.
 */
ucs_status_t ucs_async_dispatch_timerq(ucs_timer_queue_t *timerq,
                                       ucs_time_t current_time);


typedef void (*ucs_async_init_t)();

typedef void (*ucs_async_cleanup_t)();

typedef int (*ucs_async_is_from_async_t)();

typedef void (*ucs_async_block_t)();

typedef void (*ucs_async_unblock_t)();

typedef ucs_status_t (*ucs_async_context_init_t)(ucs_async_context_t *async);

typedef void (*ucs_async_context_cleanup_t)(ucs_async_context_t *async);

typedef int (*ucs_async_context_try_block_t)(ucs_async_context_t *async);

typedef void (*ucs_async_context_unblock_t)(ucs_async_context_t *async);

typedef ucs_status_t (*ucs_async_add_event_fd_t)(ucs_async_context_t *async,
                                                 int event_fd,
                                                 ucs_event_set_types_t events);

typedef ucs_status_t (*ucs_async_remove_event_fd_t)(ucs_async_context_t *async,
                                                    int event_fd);

typedef ucs_status_t (*ucs_async_modify_event_fd_t)(ucs_async_context_t *async,
                                                    int event_fd,
                                                    ucs_event_set_types_t events);

typedef ucs_status_t (*ucs_async_add_timer_t)(ucs_async_context_t *async,
                                              int timer_id,
                                              ucs_time_t interval);

typedef ucs_status_t (*ucs_async_remove_timer_t)(ucs_async_context_t *async,
                                                 int timer_id);


/**
 * Operation for specific async event delivery method.
 */
typedef struct ucs_async_ops {
    ucs_async_init_t              init;
    ucs_async_cleanup_t           cleanup;
    ucs_async_is_from_async_t     is_from_async;

    ucs_async_block_t             block;
    ucs_async_unblock_t           unblock;

    ucs_async_context_init_t      context_init;
    ucs_async_context_cleanup_t   context_cleanup;
    ucs_async_context_try_block_t context_try_block;
    ucs_async_context_unblock_t   context_unblock;

    ucs_async_add_event_fd_t      add_event_fd;
    ucs_async_remove_event_fd_t   remove_event_fd;
    ucs_async_modify_event_fd_t   modify_event_fd;

    ucs_async_add_timer_t         add_timer;
    ucs_async_remove_timer_t      remove_timer;
} ucs_async_ops_t;


extern ucs_async_ops_t ucs_async_thread_spinlock_ops;
extern ucs_async_ops_t ucs_async_thread_mutex_ops;
extern ucs_async_ops_t ucs_async_signal_ops;

#endif
