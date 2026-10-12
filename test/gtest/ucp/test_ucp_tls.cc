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

#include <numeric>
#include <vector>

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
        const ucp_object_version_t addr_version =
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

class test_ucp_tl_bitmap : public ucs::test {
protected:
    virtual void init()
    {
        UCS_STATIC_BITMAP_RESET_ALL(&m_tl_bitmap);
    }

    /* Indices visited by UCP_TL_BITMAP_FOR_EACH_BIT, which must end the
     * iteration at UCP_MAX_RESOURCES */
    std::vector<ucp_rsc_index_t> visited_bits() const
    {
        std::vector<ucp_rsc_index_t> bits;
        ucp_rsc_index_t rsc_index;

        UCP_TL_BITMAP_FOR_EACH_BIT(rsc_index, &m_tl_bitmap) {
            bits.push_back(rsc_index);
        }

        EXPECT_EQ(UCP_MAX_RESOURCES, rsc_index);
        return bits;
    }

    ucp_tl_bitmap_t m_tl_bitmap;
};

UCS_TEST_F(test_ucp_tl_bitmap, min_max_values) {
    /* No bits are set in the min bitmap */
    EXPECT_TRUE(ucs_bitmap_bits_is_equal(
            UCS_STATIC_BITMAP_BITS_CARGS(&m_tl_bitmap),
            UCS_STATIC_BITMAP_BITS_CARGS(&ucp_tl_bitmap_min)));

    /* Exactly the valid resource indices are set in the max bitmap */
    for (unsigned rsc_index = 0; rsc_index < UCP_MAX_RESOURCES; ++rsc_index) {
        UCS_STATIC_BITMAP_SET(&m_tl_bitmap, rsc_index);
    }

    EXPECT_TRUE(ucs_bitmap_bits_is_equal(
            UCS_STATIC_BITMAP_BITS_CARGS(&m_tl_bitmap),
            UCS_STATIC_BITMAP_BITS_CARGS(&ucp_tl_bitmap_max)));
}

UCS_TEST_F(test_ucp_tl_bitmap, empty) {
    EXPECT_TRUE(visited_bits().empty());
}

UCS_TEST_F(test_ucp_tl_bitmap, set_bits) {
    /* Set bits at word boundaries and the last valid index */
    const std::vector<ucp_rsc_index_t> expected_visited_bits =
            {0, 63, 64, 128, UCP_MAX_RESOURCES - 1};

    for (auto rsc_index : expected_visited_bits) {
        UCS_STATIC_BITMAP_SET(&m_tl_bitmap, rsc_index);
    }

    EXPECT_EQ(expected_visited_bits, visited_bits());
}

UCS_TEST_F(test_ucp_tl_bitmap, max) {
    std::vector<ucp_rsc_index_t> expected_visited_bits(UCP_MAX_RESOURCES);

    /* Fill the expected_visited_bits vector with the valid resource indices
     * [0, 1, 2, ..., (UCP_MAX_RESOURCES - 1)] */
    std::iota(expected_visited_bits.begin(), expected_visited_bits.end(), 0);

    m_tl_bitmap = ucp_tl_bitmap_max;
    EXPECT_EQ(expected_visited_bits, visited_bits());
}

/* Bit UCP_MAX_RESOURCES exists only because the bitmap is rounded up to words */
UCS_TEST_F(test_ucp_tl_bitmap, rounding_bit) {
    UCS_STATIC_BITMAP_SET(&m_tl_bitmap, UCP_MAX_RESOURCES);
    EXPECT_TRUE(visited_bits().empty());
}
