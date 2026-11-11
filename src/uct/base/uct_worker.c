/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_worker.h"

#include <ucs/arch/atomic.h>
#include <ucs/type/class.h>
#include <ucs/async/async.h>
#include <ucs/vfs/base/vfs_obj.h>


/* Oneshot callbacks ID range, since their context needs to be released */
#define UCT_WORKER_ONESHOT_ID_START (INT_MAX / 2)

/*
 * Context for callbacks registered with uct_worker_progress_register_safe and
 * UCS_CALLBACKQ_FLAG_ONESHOT flag.
 */
typedef struct {
    ucs_callback_t  func; /* Original user callback */
    void            *arg; /* Original user argument */
    ucs_callbackq_t *cbq; /* Callback queue for removing after dispatch */
    int             id;   /* Callback id for removing after dispatch */
} uct_worker_oneshot_cb_ctx_t;

static UCS_CLASS_INIT_FUNC(uct_worker_t)
{
    ucs_callbackq_init(&self->progress_q);
    ucs_vfs_obj_add_dir(NULL, self, "uct/worker/%p", self);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_worker_t)
{
    ucs_vfs_obj_remove(self);
    ucs_callbackq_cleanup(&self->progress_q);
}

UCS_CLASS_DEFINE(uct_worker_t, void);

static UCS_CLASS_INIT_FUNC(uct_priv_worker_t, ucs_async_context_t *async,
                           ucs_thread_mode_t thread_mode)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_worker_t);

    if (async == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

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
        prog->id = ucs_callbackq_add_safe(&worker->super.progress_q, cb, arg);
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_worker_progress_remove(uct_priv_worker_t *worker, uct_worker_progress_t *prog)
{
    UCS_ASYNC_BLOCK(worker->async);
    ucs_assert(prog->refcount > 0);
    if (ucs_atomic_fsub32(&prog->refcount, 1) == 1) {
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
            break; /* coverity thinks that `UCS_CALLBACKQ_ID_NULL`
                    * can be passed to `ucs_callbackq_remove()`
                    * make coverity happy - return from the loop */
        }
        ref = prog->refcount;
    }
    UCS_ASYNC_UNBLOCK(worker->async);
}

static unsigned uct_worker_oneshot_callback_proxy(void *arg)
{
    uct_worker_oneshot_cb_ctx_t *ctx = arg;
    unsigned count;

    count = ctx->func(ctx->arg);
    ucs_callbackq_remove_safe(ctx->cbq, ctx->id);
    ucs_free(ctx);

    return count;
}

void uct_worker_progress_register_safe(uct_worker_h tl_worker,
                                       ucs_callback_t func, void *arg,
                                       unsigned flags, uct_worker_cb_id_t *id_p)
{
    uct_priv_worker_t *worker = ucs_derived_of(tl_worker, uct_priv_worker_t);
    uct_worker_oneshot_cb_ctx_t *ctx;
    uct_worker_cb_id_t id;

    if (*id_p != UCS_CALLBACKQ_ID_NULL) {
        return;
    }

    UCS_ASYNC_BLOCK(worker->async);
    if (flags & UCS_CALLBACKQ_FLAG_ONESHOT) {
        /*
         * Implement UCS_CALLBACKQ_FLAG_ONESHOT flag, for backward compatibility
         * after removing flags parameter from ucs_callbackq. Since we have to
         * return a callback id, we cannot use ucs_callbackq_add_oneshot() API,
         * and we emulate oneshot behavior by making the callback remove itself.
         */
        ctx = ucs_malloc(sizeof(*ctx), "uct_worker_oneshot_cb_ctx");
        if (ctx == NULL) {
            ucs_error("failed to allocate oneshot callback element");
            goto out;
        }

        ctx->func = func;
        ctx->arg  = arg;
        ctx->cbq  = &worker->super.progress_q;
        ctx->id   = ucs_callbackq_add_safe(&worker->super.progress_q,
                                           uct_worker_oneshot_callback_proxy,
                                           ctx);

        id = ctx->id + UCT_WORKER_ONESHOT_ID_START;
        ucs_assertv(id >= UCT_WORKER_ONESHOT_ID_START, "id=%d", id);
    } else {
        /* Normal callback */
        id = ucs_callbackq_add_safe(&worker->super.progress_q, func, arg);
        ucs_assertv(id < UCT_WORKER_ONESHOT_ID_START, "id=%d", id);
    }

    ucs_assert(id != UCS_CALLBACKQ_ID_NULL);
    *id_p = id;

out:
    UCS_ASYNC_UNBLOCK(worker->async);
}

void uct_worker_progress_unregister_safe(uct_worker_h tl_worker,
                                         uct_worker_cb_id_t *id_p)
{
    uct_priv_worker_t *worker = ucs_derived_of(tl_worker, uct_priv_worker_t);
    uct_worker_oneshot_cb_ctx_t *ctx;

    if (*id_p == UCS_CALLBACKQ_ID_NULL) {
        return;
    }

    UCS_ASYNC_BLOCK(worker->async);
    if (*id_p < UCT_WORKER_ONESHOT_ID_START) {
        /* Normal callback */
        ucs_callbackq_remove_safe(&worker->super.progress_q, *id_p);
    } else {
        /* Oneshot with proxy callback */
        ctx = ucs_callbackq_remove_safe(&worker->super.progress_q,
                                        *id_p - UCT_WORKER_ONESHOT_ID_START);
        ucs_free(ctx);
    }
    UCS_ASYNC_UNBLOCK(worker->async);

    *id_p = UCS_CALLBACKQ_ID_NULL;
}
