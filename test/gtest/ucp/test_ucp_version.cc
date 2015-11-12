/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"


class test_ucp_version : public ucp_test {
protected:
    static ucs_log_func_rc_t
    log_handler(const char *file, unsigned line, const char *function,
                ucs_log_level_t level, const char *prefix, const char *message,
                va_list ap)
    {
        return UCS_LOG_FUNC_RC_STOP;
    }
};


UCS_TEST_F(test_ucp_version, wrong_api_version) {

    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    ucp_params_t params;
    params.features        = UCP_FEATURE_TAG;
    params.request_size    = 0;
    params.request_init    = NULL;
    params.request_cleanup = NULL;

    ucs_log_push_handler(log_handler);

    ucp_context_h ucph;
    ucs_status_t status;
    status = ucp_init_version(99, 99, &params, config.get(), &ucph);
    if (status == UCS_OK) {
        ucp_cleanup(ucph);
        ADD_FAILURE() << "Created UCP with wrong version";
    }

    ucs_log_pop_handler();
}

UCS_TEST_F(test_ucp_version, version_string) {

    unsigned major_version, minor_version, release_number;

    ucp_get_version(&major_version, &minor_version, &release_number);

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "%d.%d.%d", major_version, minor_version,
             release_number);

    EXPECT_EQ(std::string(buffer), std::string(ucp_get_version_string()));
}
