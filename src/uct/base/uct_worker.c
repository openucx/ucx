/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "uct_worker.h"

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/async/async.h>


static UCS_CLASS_INIT_FUNC(uct_worker_t)
{
    ucs_callbackq_init(&self->progress_q);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_worker_t)
{
    ucs_callbackq_cleanup(&self->progress_q);
}

UCS_CLASS_DEFINE(uct_worker_t, void);

static UCS_CLASS_INIT_FUNC(uct_priv_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_worker_t);

    self->async       = async;
    self->thread_mode = thread_mode;
    ucs_list_head_init(&self->tl_data);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_priv_worker_t)
{
}

UCS_CLASS_DEFINE(uct_priv_worker_t, uct_worker_t);

UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_worker_create, uct_priv_worker_t, uct_worker_t,
                                ucs_async_context_t*, ucs_thread_mode_t)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(uct_worker_destroy, uct_priv_worker_t, uct_worker_t)

void uct_worker_progress_init(uct_worker_progress_t *prog)
{
    prog->id       = UCS_CALLBACKQ_ID_NULL;
    prog->refcount = 0;
}

void uct_worker_progress_add_safe(uct_priv_worker_t *worker, ucs_callback_t cb,
                                  void *arg, uct_worker_progress_t *prog)
{
    UCS_ASYNC_BLOCK(worker->async);
    if (ucs_atomic_fadd32(&prog->refcount, 1) == 0) {
        prog->id = ucs_callbackq_add_safe(&worker->super.progress_q, cb, arg,
                                          UCS_CALLBACKQ_FLAG_FAST);
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_worker_progress_remove(uct_priv_worker_t *worker, uct_worker_progress_t *prog)
{
    UCS_ASYNC_BLOCK(worker->async);
    ucs_assert(prog->refcount > 0);
    if (ucs_atomic_fadd32(&prog->refcount, -1) == 1) {
        ucs_callbackq_remove(&worker->super.progress_q, prog->id);
        prog->id = UCS_CALLBACKQ_ID_NULL;
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_worker_progress_remove_all(uct_priv_worker_t *worker,
                                    uct_worker_progress_t *prog)
{
    uint32_t ref;

    UCS_ASYNC_BLOCK(worker->async);
    ref = prog->refcount;
    while (ref > 0) {
        if (ucs_atomic_cswap32(&prog->refcount, ref, 0) == ref) {
            ucs_callbackq_remove(&worker->super.progress_q, prog->id);
            prog->id = UCS_CALLBACKQ_ID_NULL;
        }
        ref = prog->refcount;
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_worker_progress_register_safe(uct_worker_h tl_worker, ucs_callback_t func,
                                       void *arg, unsigned flags,
                                       uct_worker_cb_id_t *id_p)
{
    uct_priv_worker_t *worker = ucs_derived_of(tl_worker, uct_priv_worker_t);

    if (*id_p == UCS_CALLBACKQ_ID_NULL) {
        UCS_ASYNC_BLOCK(worker->async);
        *id_p = ucs_callbackq_add_safe(&worker->super.progress_q, func, arg, flags);
        UCS_ASYNC_UNBLOCK(worker->async);
        ucs_assert(*id_p != UCS_CALLBACKQ_ID_NULL);
    }
}

void uct_worker_progress_unregister_safe(uct_worker_h tl_worker,
                                         uct_worker_cb_id_t *id_p)
{
    uct_priv_worker_t *worker = ucs_derived_of(tl_worker, uct_priv_worker_t);

    if (*id_p != UCS_CALLBACKQ_ID_NULL) {
        UCS_ASYNC_BLOCK(worker->async);
        ucs_callbackq_remove_safe(&worker->super.progress_q, *id_p);
        UCS_ASYNC_UNBLOCK(worker->async);
        *id_p = UCS_CALLBACKQ_ID_NULL;
    }
}
