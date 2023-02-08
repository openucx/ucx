/**
 * Copyright (C) 2023 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <cuda.h>
#include <stdio.h>


#define CUDA_CALL(_func) \
    ({ \
        CUresult _cu_result = (_func); \
        const char *_error_string; \
        if (_cu_result != CUDA_SUCCESS) { \
            cuGetErrorString(_cu_result, &_error_string); \
            printf("%s failed: %s\n", #_func, _error_string); \
        } \
        _cu_result; \
    })


int main(int argc, char **argv)
{
    int ret = -EXIT_FAILURE;
    ucp_params_t ucp_params;
    ucp_context_h ucp_context;
    ucs_status_t status;
    CUcontext cu_context;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG;

    status = ucp_init(&ucp_params, NULL, &ucp_context);
    if (status != UCS_OK) {
        printf("ucp_init failed: %s\n", ucs_status_string(status));
        goto out;
    }

    if (CUDA_CALL(cuInit(0)) != CUDA_SUCCESS) {
        goto cleanup;
    }

    if (CUDA_CALL(cuCtxGetCurrent(&cu_context)) != CUDA_SUCCESS) {
        goto cleanup;
    }

    if (cu_context != NULL) {
        printf("failed: CUDA context is not NULL\n");
        goto cleanup;
    }

    printf("SUCCESS\n");
    ret = EXIT_SUCCESS;

cleanup:
    ucp_cleanup(ucp_context);
out:
    return ret;
}
