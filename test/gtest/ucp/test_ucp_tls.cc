/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <ucp/core/ucp_context.h>

class test_ucp_tl : public test_ucp_context {
};

UCS_TEST_P(test_ucp_tl, check_ucp_tl, "SELF_NUM_DEVICES?=50")
{
    ucs::handle<ucp_config_t *> config;
    ucp_params_t                params;

    UCS_TEST_CREATE_HANDLE(ucp_config_t *, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    VALGRIND_MAKE_MEM_UNDEFINED(&params, sizeof(params));
    params.features   = get_variant_ctx_params().features;
    params.field_mask = UCP_PARAM_FIELD_FEATURES;

    ucs::handle<ucp_context_h> ucph;
    UCS_TEST_CREATE_HANDLE(ucp_context_h, ucph, ucp_cleanup, ucp_init, &params,
                           config.get());

    EXPECT_GE(ucph.get()->num_tls, 50);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tl, self, "self");
