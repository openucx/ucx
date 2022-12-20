/*
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_signal.h"

static ucs_mpool_t rocm_base_signal_pool;
static ucs_queue_head_t rocm_base_signal_queue;
static int rocm_base_signal_is_init=0;

ucs_status_t
uct_rocm_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                         uct_completion_t *comp)
{
    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_queue_is_empty(&rocm_base_signal_queue)) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

unsigned uct_rocm_base_iface_progress(uct_iface_h tl_iface)
{
    static const unsigned max_signals = 16;
    unsigned count = 0;
    uct_rocm_base_signal_desc_t *rocm_signal;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(rocm_signal, iter, &rocm_base_signal_queue, queue) {
        if (hsa_signal_load_scacquire(rocm_signal->signal) != 0) {
            continue;
        }

        ucs_queue_del_iter(&rocm_base_signal_queue, iter);
        if (rocm_signal->comp != NULL) {
            uct_invoke_completion(rocm_signal->comp, UCS_OK);
        }

        ucs_trace_poll("ROCM_BASE Signal Done :%p", rocm_signal);
        ucs_mpool_put(rocm_signal);
        count++;

        if (count >= max_signals) {
            break;
        }
    }

    return count;
}

static void uct_rocm_base_signal_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_rocm_base_signal_desc_t *base = (uct_rocm_base_signal_desc_t *)obj;
    hsa_status_t status;

    memset(base, 0, sizeof(*base));
    status = hsa_signal_create(1, 0, NULL, &base->signal);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_fatal("fail to create signal");
    }
}

static void uct_rocm_base_signal_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_rocm_base_signal_desc_t *base = (uct_rocm_base_signal_desc_t *)obj;
    hsa_status_t status;

    status = hsa_signal_destroy(base->signal);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_fatal("fail to destroy signal");
    }
}

static ucs_mpool_ops_t uct_rocm_base_signal_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_rocm_base_signal_desc_init,
    .obj_cleanup   = uct_rocm_base_signal_desc_cleanup,
    .obj_str       = NULL
};

ucs_status_t uct_rocm_base_signal_init(void)
{
    ucs_mpool_params_t mp_params;
    ucs_status_t status;

    if (!rocm_base_signal_is_init) {
        ucs_mpool_params_reset(&mp_params);
        mp_params.elem_size       = sizeof(uct_rocm_base_signal_desc_t);
        mp_params.elems_per_chunk = 128;
        mp_params.max_elems       = 1024;
        mp_params.ops             = &uct_rocm_base_signal_desc_mpool_ops;
        mp_params.name            = "ROCM signal objects";

        status = ucs_mpool_init(&mp_params, &rocm_base_signal_pool);
        if (status != UCS_OK) {
          ucs_error("rocm/base signal mpool creation failed");
          return status;
        }

        ucs_queue_head_init(&rocm_base_signal_queue);
        rocm_base_signal_is_init = 1;
    }
    return UCS_OK;
}

void uct_rocm_base_signal_finalize()
{
    if (rocm_base_signal_is_init) {
        ucs_mpool_cleanup(&rocm_base_signal_pool, 1);
        rocm_base_signal_is_init = 0;
    }
}

uct_rocm_base_signal_desc_t* uct_rocm_base_get_signal()
{
    uct_rocm_base_signal_desc_t *rocm_signal;

    rocm_signal = ucs_mpool_get(&rocm_base_signal_pool);
    return rocm_signal;
}

void uct_rocm_base_signal_push(uct_rocm_base_signal_desc_t* rocm_signal)
{
    ucs_queue_push(&rocm_base_signal_queue, &rocm_signal->queue);
}
