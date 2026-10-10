/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <ucp/core/ucp_context.h>

extern "C" {
#include <ucp/wireup/address.h>
}

class test_ucp_tl : public test_ucp_context {
protected:
    virtual void init()
    {
        test_base::init(); /* Skip entities creation, tests create their own */
    }

    void check_resources(unsigned num_devices)
    {
        ucp_unpacked_address_t unpacked_address;
        ucs_status_t status;
        size_t size;
        void *buffer;

        create_entity();

        /* One resource per self device */
        ucp_worker_h worker = sender().worker();
        EXPECT_EQ(num_devices, worker->context->num_tls);

        /* Pack the address directly: with UCP_FEATURE_WAKEUP, an endpoint
         * needs send completion events, which self does not have */
        ucp_object_version_t addr_version =
                worker->context->config.ext.worker_addr_version;
        ASSERT_UCS_OK(ucp_address_pack(worker, NULL, &ucp_tl_bitmap_max,
                                       UCP_ADDRESS_PACK_FLAGS_ALL, addr_version,
                                       NULL, UINT_MAX, &size, &buffer));

        status = ucp_address_unpack(worker, buffer, UCP_ADDRESS_PACK_FLAGS_ALL,
                                    &unpacked_address);
        ucs_free(buffer);
        ASSERT_UCS_OK(status);

        /* One address entry per resource */
        EXPECT_EQ(worker->context->num_tls, unpacked_address.address_count);
        ucs_free(unpacked_address.address_list);
    }
};

UCS_TEST_P(test_ucp_tl, check_ucp_tl, "SELF_NUM_DEVICES?=50")
{
    check_resources(50);
}

UCS_TEST_P(test_ucp_tl, check_ucp_tl_max)
{
    modify_config("SELF_NUM_DEVICES", ucs::to_string(UCP_MAX_RESOURCES),
                  SETENV_IF_NOT_EXIST);
    check_resources(UCP_MAX_RESOURCES);
}

UCS_TEST_P(test_ucp_tl, check_ucp_tl_exceeds_limit)
{
    ucp_context_h ucph;
    ucs_status_t status;

    modify_config("SELF_NUM_DEVICES", ucs::to_string(UCP_MAX_RESOURCES + 1),
                  SETENV_IF_NOT_EXIST);

    {
        const scoped_log_handler slh(hide_errors_logger);
        status = ucp_init(&get_variant_ctx_params(), m_ucp_config, &ucph);
    }

    EXPECT_EQ(UCS_ERR_EXCEEDS_LIMIT, status);
    if (status == UCS_OK) {
        ucp_cleanup(ucph);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_tl, self, "self");
