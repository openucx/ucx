/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>

#if _OPENMP
#include <omp.h>
#endif


int main(int argc, char **argv)
{
    int count = 0;

#pragma omp parallel
    {
        ucs_status_t ctx_status, worker_status;
        ucp_context_h context;
        ucp_worker_h worker;
        ucp_params_t params;
        ucp_worker_params_t wparams;

        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features   = UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
        ctx_status        = ucp_init(&params, NULL, &context);
        if (ctx_status == UCS_OK) {
            wparams.field_mask = 0;
            worker_status      = ucp_worker_create(context, &wparams, &worker);
            if (worker_status == UCS_OK) {
                __sync_add_and_fetch(&count, 1);
            }
        }

#pragma omp barrier

        if (ctx_status == UCS_OK) {
            if (worker_status == UCS_OK) {
                ucp_worker_destroy(worker);
            }
            ucp_cleanup(context);
        }
    }

#pragma omp barrier

    printf("finished %d threads\n", count);
    return 0;
}
