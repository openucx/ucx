/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "uct_worker.h"

#include <ucs/type/class.h>
#include <ucs/datastruct/callbackq.inl>


static UCS_CLASS_INIT_FUNC(uct_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    self->async       = async;
    self->thread_mode = thread_mode;
    ucs_callbackq_init(&self->progress_q, 64, async);
    ucs_list_head_init(&self->tl_data);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_worker_t)
{
    ucs_callbackq_cleanup(&self->progress_q);
}

UCS_CLASS_DEFINE(uct_worker_t, void);
UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_worker_create, uct_worker_t, uct_worker_t,
                                ucs_async_context_t*, ucs_thread_mode_t)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(uct_worker_destroy, uct_worker_t, uct_worker_t)

void uct_worker_progress(uct_worker_h worker)
{
    ucs_callbackq_dispatch(&worker->progress_q);
}

void uct_worker_progress_register(uct_worker_h worker,
                                  ucs_callback_t func, void *arg)
{
    ucs_callbackq_add(&worker->progress_q, func, arg);
}

void uct_worker_progress_unregister(uct_worker_h worker,
                                    ucs_callback_t func, void *arg)
{
    ucs_callbackq_remove(&worker->progress_q, func, arg);
}

void uct_worker_slowpath_progress_register(uct_worker_h worker,
                                           ucs_callbackq_slow_elem_t *elem)
{
    ucs_callbackq_add_slow_path(&worker->progress_q, elem);
}

void uct_worker_slowpath_progress_unregister(uct_worker_h worker,
                                             ucs_callbackq_slow_elem_t *elem)
{
    ucs_callbackq_remove_slow_path(&worker->progress_q, elem);
}

