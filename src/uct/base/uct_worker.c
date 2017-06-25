/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "uct_worker.h"

#include <ucs/type/class.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/callbackq.inl>


typedef struct uct_worker_slowpath_elem {
    ucs_callbackq_slow_elem_t super;
    ucs_callback_t            cb;
    void                      *arg;
} uct_worker_slowpath_elem_t;


static UCS_CLASS_INIT_FUNC(uct_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    self->async       = async;
    self->thread_mode = thread_mode;
    ucs_callbackq_init(&self->progress_q, 64, NULL);
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

void uct_worker_progress_init(uct_worker_progress_t *prog)
{
    prog->cb       = NULL;
    prog->arg      = NULL;
    prog->refcount = 0;
}

void uct_worker_progress_register(uct_worker_h worker, ucs_callback_t cb,
                                  void *arg, uct_worker_progress_t *prog)
{
    UCS_ASYNC_BLOCK(worker->async);
    if (prog->refcount++ == 0) {
        prog->cb  = cb;
        prog->arg = arg;
        ucs_callbackq_add(&worker->progress_q, cb, arg);
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_worker_progress_unregister(uct_worker_h worker,
                                    uct_worker_progress_t *prog)
{
    UCS_ASYNC_BLOCK(worker->async);
    ucs_assert(prog->refcount > 0);
    if (--prog->refcount == 0) {
        ucs_callbackq_remove(&worker->progress_q, prog->cb, prog->arg);
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

static void uct_worker_slowpath_proxy(ucs_callbackq_slow_elem_t *self)
{
    uct_worker_slowpath_elem_t *elem = ucs_derived_of(self, uct_worker_slowpath_elem_t);
    elem->cb(elem->arg);
}

void uct_worker_progress_register_safe(uct_worker_h worker, ucs_callback_t func,
                                       void *arg, unsigned flags,
                                       uct_worker_cb_id_t *id_p)
{
    uct_worker_slowpath_elem_t *elem;

    if (*id_p == NULL) {
        UCS_ASYNC_BLOCK(worker->async);

        elem = ucs_malloc(sizeof(*elem), "uct_worker_slowpath_elem");
        ucs_assert_always(elem != NULL);

        elem->super.cb = uct_worker_slowpath_proxy;
        elem->cb       = func;
        elem->arg      = arg;
        ucs_callbackq_add_slow_path(&worker->progress_q, &elem->super);
        *id_p          = elem;

        UCS_ASYNC_UNBLOCK(worker->async);
    }
}

void uct_worker_progress_unregister_safe(uct_worker_h worker,
                                         uct_worker_cb_id_t *id_p)
{
    uct_worker_slowpath_elem_t *elem;

    if (*id_p != NULL) {
        UCS_ASYNC_BLOCK(worker->async);
        elem = *id_p;
        ucs_callbackq_remove_slow_path(&worker->progress_q, &elem->super);
        ucs_free(elem);
        UCS_ASYNC_UNBLOCK(worker->async);
        *id_p = NULL;
    }
}
