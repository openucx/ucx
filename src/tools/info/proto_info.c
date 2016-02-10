/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucx_info.h"

#include <ucp/api/ucp.h>
#include <string.h>


void print_proto_info(ucs_config_print_flags_t print_flags)
{
    ucp_config_t *config;
    ucs_status_t status;
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_params_t params;

    status = ucp_config_read(NULL, NULL, &config);
    if (status != UCS_OK) {
        return;
    }

    memset(&params, 0, sizeof(params));
    params.features = UCP_FEATURE_TAG |
                      UCP_FEATURE_RMA |
                      UCP_FEATURE_AMO32 |
                      UCP_FEATURE_AMO64;
    status = ucp_init(&params, config, &context);
    if (status != UCS_OK) {
        printf("<Failed to create UCP context>\n");
        goto out_release_config;
    }

    status = ucp_worker_create(context, UCS_THREAD_MODE_MULTI, &worker);
    if (status != UCS_OK) {
        printf("<Failed to create UCP worker>\n");
        goto out_cleanup_context;
    }

    ucp_worker_proto_print(worker, stdout, "UCP protocols", print_flags);

    ucp_worker_destroy(worker);

out_cleanup_context:
    ucp_cleanup(context);
out_release_config:
    ucp_config_release(config);
}
