/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/api/ucp.h>
#include <dlfcn.h>
#include <stdio.h>


int main(int argc, char **argv)
{
    ucp_params_t params;
    ucs_status_t status;
    ucp_context_h context;

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_TAG;

    status = ucp_init(&params, NULL, &context);
    if (status != UCS_OK) {
        return -1;
    }

    dlopen("libselinux.so", RTLD_LAZY);

    ucp_cleanup(context);

    printf("SUCCESS\n");
    return 0;
}

