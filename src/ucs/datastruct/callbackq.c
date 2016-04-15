/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/type/spinlock.h>
#include <ucs/arch/atomic.h>
#include <ucs/async/async.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug.h>
#include <ucs/sys/sys.h>

#include "callbackq.h"


typedef struct ucs_callbackq_priv {
    ucs_spinlock_t                   lock;     /**< Protects adding / removing */
    ucs_async_context_t              *async;   /**< Async context */
} ucs_callbackq_priv_t;


static inline ucs_callbackq_priv_t* ucs_callbackq_priv(ucs_callbackq_t *cbq)
{
    UCS_STATIC_ASSERT(sizeof(cbq->priv) == sizeof(ucs_callbackq_priv_t));
    return (void*)cbq->priv;
}

static void ucs_callbackq_service_enable(ucs_callbackq_t *cbq)
{
    cbq->start = cbq->ptr;
}

static void ucs_callbackq_service_disable(ucs_callbackq_t *cbq)
{
    cbq->start = cbq->ptr + 1;
}

static void ucs_callbackq_enter(ucs_callbackq_t *cbq)
{
    if (ucs_callbackq_priv(cbq)->async != NULL) {
        UCS_ASYNC_BLOCK(ucs_callbackq_priv(cbq)->async);
    }
    ucs_spin_lock(&ucs_callbackq_priv(cbq)->lock);
}

static void ucs_callbackq_leave(ucs_callbackq_t *cbq)
{
    ucs_spin_unlock(&ucs_callbackq_priv(cbq)->lock);
    if (ucs_callbackq_priv(cbq)->async != NULL) {
        UCS_ASYNC_UNBLOCK(ucs_callbackq_priv(cbq)->async);
    }
}

/* called with lock held */
static ucs_callbackq_elem_t* ucs_callbackq_find(ucs_callbackq_t *cbq,
                                                ucs_callback_t cb, void *arg)
{
    ucs_callbackq_elem_t *elem;
    ucs_callbackq_for_each(elem, cbq) {
        if ((elem->cb == cb) && (elem->arg == arg)) {
            return elem;
        }
    }
    return NULL;
}

/* called with lock held */
static void ucs_callbackq_remove_elem(ucs_callbackq_t *cbq, ucs_callbackq_elem_t *elem)
{
    char func_name[200];

    ucs_trace("cbq %p: remove %p %s(arg=%p) [start:%p end:%p]", cbq, elem,
              ucs_debug_get_symbol_name(elem->cb, func_name, sizeof(func_name)),
              elem->arg, cbq->start, cbq->end);

    ucs_assert(cbq->start < cbq->end);

    if (elem != cbq->end - 1) {
        *elem = *(cbq->end - 1);
    }
    --cbq->end;
}

/*
 * Service callback is a special callback in the callback queue, which is always
 * in the first entry in the array. It is responsible for adding / removing items
 * to the callback queue on behalf of other threads, since it is guaranteed to
 * run from the "main" thread.
 */
static void ucs_callbackq_service_cb(void *arg)
{
    ucs_callbackq_t *cbq = arg;
    ucs_callbackq_elem_t *elem;

    ucs_callbackq_enter(cbq);
    elem = cbq->ptr + 1;
    while (elem < cbq->end) {
        if (elem->refcount == 0) {
            ucs_callbackq_remove_elem(cbq, elem);
        } else {
            ++elem;
        }
    }
    ucs_callbackq_service_disable(cbq);
    ucs_callbackq_leave(cbq);
}

ucs_status_t ucs_callbackq_init(ucs_callbackq_t *cbq, size_t size,
                                ucs_async_context_t *async)
{
    /* reserve a slot for the special service callback */
    ++size;

    cbq->ptr  = ucs_malloc(size * sizeof(*cbq->ptr), "callback queue");
    if (cbq->ptr == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    cbq->ptr->cb       = ucs_callbackq_service_cb;
    cbq->ptr->arg      = cbq;
    cbq->ptr->refcount = 1;
    cbq->size          = size;
    cbq->start         = cbq->ptr + 1;
    cbq->end           = cbq->start;
    ucs_callbackq_priv(cbq)->async = async;
    ucs_spinlock_init(&ucs_callbackq_priv(cbq)->lock);
    return UCS_OK;
}

void ucs_callbackq_cleanup(ucs_callbackq_t *cbq)
{
    ucs_callbackq_elem_t *elem;
    char func_name[200];

    if (cbq->start != cbq->end) {
        ucs_warn("%zd callbacks still remain in callback queue",
                 cbq->end - cbq->start);
        ucs_callbackq_for_each(elem, cbq) {
            ucs_warn("cbq %p: remain %p %s(arg=%p)", cbq, elem,
                     ucs_debug_get_symbol_name(elem->cb, func_name,
                                               sizeof(func_name)),
                     elem->arg);
        }
    }
    ucs_free(cbq->ptr);
}

void ucs_callbackq_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg)
{
    ucs_callbackq_elem_t *elem;
    char func_name[200];

    ucs_callbackq_enter(cbq);

    elem = ucs_callbackq_find(cbq, cb, arg);
    if (elem != NULL) {
        ucs_atomic_add32(&elem->refcount, 1);
        ucs_callbackq_leave(cbq);
        return;
    }

    if (cbq->end >= cbq->ptr + cbq->size) {
        /* TODO support expanding the callback queue */
        ucs_fatal("callback queue %p is full, cannot add %s()", cbq,
                  ucs_debug_get_symbol_name(cb, func_name, sizeof(func_name)));
    }

    elem = cbq->end;

    ucs_trace("cbq %p: adding %p %s(arg=%p) [start:%p end:%p]", cbq, elem,
              ucs_debug_get_symbol_name(cb, func_name, sizeof(func_name)),
              arg, cbq->start, cbq->end);

    elem->cb       = cb;
    elem->arg      = arg;
    elem->refcount = 1;

    /* Make sure a thread dispatching the callbacks would see 'end' only after
     * the new element is set.
     */
    ucs_memory_cpu_store_fence();

    ++cbq->end;
    ucs_callbackq_leave(cbq);
}

ucs_status_t ucs_callbackq_remove(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                  void *arg)
{
    ucs_callbackq_elem_t *elem;

    ucs_callbackq_enter(cbq);

    elem = ucs_callbackq_find(cbq, cb, arg);
    if (elem == NULL) {
        ucs_debug("callback not found in progress chain");
        ucs_callbackq_leave(cbq);
        return UCS_ERR_NO_ELEM;
    }

    if (ucs_atomic_fadd32(&elem->refcount, -1) == 1) {
        ucs_callbackq_remove_elem(cbq, elem);
    }

    ucs_callbackq_leave(cbq);
    return UCS_OK;
}

void ucs_callbackq_remove_all(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg)
{
    ucs_callbackq_elem_t *elem;

    ucs_callbackq_enter(cbq);
    elem = ucs_callbackq_find(cbq, cb, arg);
    if (elem != NULL) {
        ucs_callbackq_remove_elem(cbq, elem);
    }
    ucs_callbackq_leave(cbq);
}

ucs_status_t ucs_callbackq_add_safe(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                    void *arg)
{
    ucs_callbackq_add(cbq, cb, arg);
    return UCS_OK;
}

ucs_status_t ucs_callbackq_remove_safe(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                       void *arg)
{
    ucs_callbackq_elem_t *elem;

    ucs_callbackq_enter(cbq);

    elem = ucs_callbackq_find(cbq, cb, arg);
    if (elem == NULL) {
        ucs_debug("callback not found in progress chain");
        ucs_callbackq_leave(cbq);
        return UCS_ERR_NO_ELEM;
    }

    if (ucs_atomic_fadd32(&elem->refcount, -1) == 1) {
        ucs_callbackq_service_enable(cbq);
    }

    ucs_callbackq_leave(cbq);
    return UCS_OK;
}
