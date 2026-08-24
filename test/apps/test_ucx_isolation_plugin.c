/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <dlfcn.h>
#include <stdio.h>


const char *test_ucx_isolation_plugin_ucp_path(void)
{
    const char *(*version_fn)(void) = ucp_get_version_string;
    Dl_info info;

    if (!dladdr((void*)version_fn, &info) || (info.dli_fname == NULL)) {
        return NULL;
    }

    return info.dli_fname;
}

int test_ucx_isolation_plugin_init(void)
{
    ucp_context_h context;
    ucp_params_t params;
    ucs_status_t status;

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_TAG;

    status = ucp_init_version(UCP_API_MAJOR, UCP_API_MINOR, &params, NULL,
                              &context);
    if (status != UCS_OK) {
        fprintf(stderr, "ucp_init_version() failed: %s\n",
                ucs_status_string(status));
        return -1;
    }

    ucp_cleanup(context);
    return 0;
}
