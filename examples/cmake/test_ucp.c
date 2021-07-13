/**
* Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucp/api/ucp.h>
#include <assert.h>
#include <string.h>

int main(int argc, char **argv)
{
    ucp_params_t ucp_params = {};
    ucp_context_h ucp_context;
    ucs_status_t status;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_AM;
    status = ucp_init(&ucp_params, NULL, &ucp_context);
    assert(status == UCS_OK);

    ucp_cleanup(ucp_context);
    return 0;
}
