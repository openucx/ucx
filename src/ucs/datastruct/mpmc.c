/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mpmc.h"

#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack_int.h>


#define UCS_MPMC_INVALID_VALUE -1

ucs_status_t ucs_mpmc_queue_init(ucs_mpmc_queue_t *mpmc)
{
    ucs_queue_head_init(&mpmc->queue);
    return ucs_spinlock_init(&mpmc->lock, 0);
}

void ucs_mpmc_queue_cleanup(ucs_mpmc_queue_t *mpmc)
{
    ucs_mpmc_elem_t *elem;

    while (!ucs_queue_is_empty(&mpmc->queue)) {
        elem = ucs_queue_pull_elem_non_empty(&mpmc->queue,
                                             ucs_mpmc_elem_t, super);
        ucs_free(elem);
    }
}

ucs_status_t ucs_mpmc_queue_push(ucs_mpmc_queue_t *mpmc, uint64_t value)
{
    ucs_mpmc_elem_t *elem;

    elem = ucs_malloc(sizeof(ucs_mpmc_elem_t), "mpmc elem");
    if (elem == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    elem->value = value;

    ucs_spin_lock(&mpmc->lock);
    ucs_queue_push(&mpmc->queue, &elem->super);
    ucs_spin_unlock(&mpmc->lock);

    return UCS_OK;
}

ucs_status_t ucs_mpmc_queue_pull(ucs_mpmc_queue_t *mpmc, uint64_t *value_p)
{
    ucs_status_t status = UCS_ERR_NO_PROGRESS;
    ucs_mpmc_elem_t *elem;

    if (ucs_queue_is_empty(&mpmc->queue)) {
        return status;
    }

    ucs_spin_lock(&mpmc->lock);
    while (!ucs_queue_is_empty(&mpmc->queue)) {
        elem = ucs_queue_pull_elem_non_empty(&mpmc->queue, ucs_mpmc_elem_t,
                                             super);
        if (elem->value != UCS_MPMC_INVALID_VALUE) {
            *value_p = elem->value;
            status   = UCS_OK;
            ucs_free(elem);
            break;
        }

        ucs_free(elem);
    }
    ucs_spin_unlock(&mpmc->lock);

    return status;
}

void ucs_mpmc_queue_remove_if(ucs_mpmc_queue_t *mpmc,
                              ucs_mpmc_queue_predicate_t predicate, void *arg)
{
    ucs_mpmc_elem_t *elem;
    ucs_queue_iter_t iter;

    ucs_spin_lock(&mpmc->lock);
    ucs_queue_for_each_safe(elem, iter, &mpmc->queue, super) {
        if (predicate(elem->value, arg)) {
            elem->value = UCS_MPMC_INVALID_VALUE;
        }
    }
    ucs_spin_unlock(&mpmc->lock);
}
