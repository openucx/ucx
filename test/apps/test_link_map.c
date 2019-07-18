/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

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

    /* This could segfault if libucm_cuda.so is marked as linker nodelete but
     * could not be loaded due to libcuda dependency, because of a corrupted
     * link_map in the program.
     */
    dlopen("libgcc_s.so.1", RTLD_LAZY);

    ucp_cleanup(context);

    printf("SUCCESS\n");
    return 0;
}

