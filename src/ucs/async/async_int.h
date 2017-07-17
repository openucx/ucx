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
    int                        events;  /* Bitmap of events */
    ucs_async_event_cb_t       cb;      /* Callback function */
    void                       *arg;    /* Callback argument */
    ucs_async_context_t        *async;  /* Async context for the handler. Can be NULL */
    volatile uint32_t          missed;  /* Protect against adding to miss queue multiple times */
    volatile uint32_t          refcount;
};


/**
 * Dispatch event coming from async context.
 *
 * @param id         Array of event IDs to dispatch.
 * @param count      Number of events
 */
ucs_status_t ucs_async_dispatch_handlers(int *events, size_t count);


/**
 * Dispatch timers from a timer queue.
 *
 * @param timerq        Timer queue whose timers to dispatch.
 * @param current_time  Current time for checking timer expiration.
 */
ucs_status_t ucs_async_dispatch_timerq(ucs_timer_queue_t *timerq,
                                       ucs_time_t current_time);


/**
 * Operation for specific async event delivery method.
 */
typedef struct ucs_async_ops {
    void         (*init)();
    void         (*cleanup)();

    void         (*block)();
    void         (*unblock)();

    ucs_status_t (*context_init)(ucs_async_context_t *async);
    int          (*context_try_block)(ucs_async_context_t *async);
    void         (*context_unblock)(ucs_async_context_t *async);

    ucs_status_t (*add_event_fd)(ucs_async_context_t *async, int event_fd,
                                 int events);
    ucs_status_t (*remove_event_fd)(ucs_async_context_t *async, int event_fd);
    ucs_status_t (*modify_event_fd)(ucs_async_context_t *async, int event_fd,
                                    int events);

    ucs_status_t (*add_timer)(ucs_async_context_t *async, int timer_id,
                              ucs_time_t interval);
    ucs_status_t (*remove_timer)(ucs_async_context_t *async, int timer_id);
} ucs_async_ops_t;


extern ucs_async_ops_t ucs_async_thread_ops;
extern ucs_async_ops_t ucs_async_signal_ops;

#endif
