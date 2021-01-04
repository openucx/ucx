/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
extern "C" {
#include <ucs/sys/sys.h>
}

UCS_TEST_P(test_ucp_context, minimal_field_mask) {
    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    ucs::handle<ucp_context_h> ucph;
    ucs::handle<ucp_worker_h> worker;

    {
        /* Features ONLY */
        ucp_params_t params;
        VALGRIND_MAKE_MEM_UNDEFINED(&params, sizeof(params));
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features   = get_variant_ctx_params().features;

        UCS_TEST_CREATE_HANDLE(ucp_context_h, ucph, ucp_cleanup,
                               ucp_init, &params, config.get());
    }

    {
        /* Empty set */
        ucp_worker_params_t params;
        VALGRIND_MAKE_MEM_UNDEFINED(&params, sizeof(params));
        params.field_mask = 0;

        UCS_TEST_CREATE_HANDLE(ucp_worker_h, worker, ucp_worker_destroy,
                               ucp_worker_create, ucph.get(), &params);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_context, all, "all")

class test_ucp_aliases : public test_ucp_context {
};

UCS_TEST_P(test_ucp_aliases, aliases) {
    create_entity();
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, rcv, "rc_v")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, rcx, "rc_x")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, ud, "ud")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, ud_mlx5, "ud_mlx5")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, ugni, "ugni")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, shm, "shm")


class test_ucp_version : public test_ucp_context {
};

UCS_TEST_P(test_ucp_version, wrong_api_version) {

    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    ucp_context_h ucph;
    ucs_status_t status;
    size_t warn_count;
    {
        scoped_log_handler slh(hide_warns_logger);
        warn_count = m_warnings.size();
        status = ucp_init_version(99, 99, &get_variant_ctx_params(),
                                  config.get(), &ucph);
    }
    if (status != UCS_OK) {
        ADD_FAILURE() << "Failed to create UCP with wrong version";
    } else {
        if (m_warnings.size() == warn_count) {
            ADD_FAILURE() << "Missing wrong version warning";
        }
        ucp_cleanup(ucph);
    }
}

UCS_TEST_P(test_ucp_version, version_string) {

    unsigned major_version, minor_version, release_number;

    ucp_get_version(&major_version, &minor_version, &release_number);

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%d.%d.%d", major_version, minor_version,
             release_number);

    EXPECT_EQ(std::string(buffer), std::string(ucp_get_version_string()));
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_version, all, "all")
