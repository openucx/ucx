/*
 * Copyright (C) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/rocm/base/rocm_base.h>
#include <uct/rocm/base/rocm_signal.h>

static void uct_rocm_base_signal_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_rocm_base_signal_desc_t *base = (uct_rocm_base_signal_desc_t*)obj;
    hsa_status_t status;

    memset(base, 0, sizeof(*base));
    status = hsa_signal_create(1, 0, NULL, &base->signal);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_fatal("fail to create signal");
    }
}

static void uct_rocm_base_signal_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_rocm_base_signal_desc_t *base = (uct_rocm_base_signal_desc_t*)obj;
    hsa_status_t status;

    status = hsa_signal_destroy(base->signal);
    if (status != HSA_STATUS_SUCCESS) {
        ucs_warn("fail to destroy signal");
    }
}

ucs_mpool_ops_t uct_rocm_base_signal_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_rocm_base_signal_desc_init,
    .obj_cleanup   = uct_rocm_base_signal_desc_cleanup,
    .obj_str       = NULL
};

unsigned uct_rocm_base_progress(ucs_queue_head_t *signal_queue)
{
    static const unsigned max_signals = 16;
    unsigned count                    = 0;
    uct_rocm_base_signal_desc_t *rocm_signal;

    ucs_queue_for_each_extract(rocm_signal, signal_queue, queue,
			       (hsa_signal_load_scacquire(rocm_signal->signal) == 0) &&
			       (count < max_signals)) {
        if (rocm_signal->comp != NULL) {
            uct_invoke_completion(rocm_signal->comp, UCS_OK);
        }

        ucs_trace_poll("rocm signal done :%p", rocm_signal);
        ucs_mpool_put(rocm_signal);
        count++;
    }

    return count;
}

