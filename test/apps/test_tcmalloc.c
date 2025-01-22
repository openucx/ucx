/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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

