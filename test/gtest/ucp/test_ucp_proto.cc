/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_rkey.h>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_select.h>
#include <ucp/proto/proto_select.inl>
#include <ucp/core/ucp_worker.inl>
}

class test_ucp_proto : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_RMA);
    }

protected:
    virtual void init() {
        modify_config("PROTO_ENABLE", "y");
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
    }

    ucp_worker_h worker() {
        return sender().worker();
    }
};

UCS_TEST_P(test_ucp_proto, dump_protocols) {
    ucp_proto_select_param_t select_param;
    ucs_string_buffer_t strb;

    select_param.op_id      = UCP_OP_ID_TAG_SEND;
    select_param.op_flags   = 0;
    select_param.dt_class   = UCP_DATATYPE_CONTIG;
    select_param.mem_type   = UCS_MEMORY_TYPE_HOST;
    select_param.sys_dev    = UCS_SYS_DEVICE_ID_UNKNOWN;
    select_param.sg_count   = 1;
    select_param.padding[0] = 0;
    select_param.padding[1] = 0;

    ucs_string_buffer_init(&strb);
    ucp_proto_select_param_str(&select_param, &strb);
    UCS_TEST_MESSAGE << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);

    ucp_worker_h worker                   = sender().worker();
    ucp_worker_cfg_index_t ep_cfg_index   = sender().ep()->cfg_index;
    ucp_worker_cfg_index_t rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;

    ucp_proto_select_lookup(worker, &worker->ep_config[ep_cfg_index].proto_select,
                            ep_cfg_index, rkey_cfg_index, &select_param, 0);
    ucp_ep_print_info(sender().ep(), stdout);
}

UCS_TEST_P(test_ucp_proto, rkey_config) {
    ucp_rkey_config_key_t rkey_config_key;

    rkey_config_key.ep_cfg_index = 0;
    rkey_config_key.md_map       = 0;
    rkey_config_key.mem_type     = UCS_MEMORY_TYPE_HOST;
    rkey_config_key.sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;

    ucs_status_t status;

    /* similar configurations should return same index */
    ucp_worker_cfg_index_t cfg_index1;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index1);
    ASSERT_UCS_OK(status);

    ucp_worker_cfg_index_t cfg_index2;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index2);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(static_cast<int>(cfg_index1), static_cast<int>(cfg_index2));

    rkey_config_key.ep_cfg_index = 0;
    rkey_config_key.md_map       = 1;
    rkey_config_key.mem_type     = UCS_MEMORY_TYPE_HOST;
    rkey_config_key.sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;

    /* different configuration should return different index */
    ucp_worker_cfg_index_t cfg_index3;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index3);
    ASSERT_UCS_OK(status);

    EXPECT_NE(static_cast<int>(cfg_index1), static_cast<int>(cfg_index3));
}

UCS_TEST_P(test_ucp_proto, worker_print_info_rkey)
{
    ucp_rkey_config_key_t rkey_config_key;

    rkey_config_key.ep_cfg_index = 0;
    rkey_config_key.md_map       = 0;
    rkey_config_key.mem_type     = UCS_MEMORY_TYPE_HOST;
    rkey_config_key.sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;

    /* similar configurations should return same index */
    ucp_worker_cfg_index_t cfg_index;
    ucs_status_t status = ucp_worker_rkey_config_get(worker(), &rkey_config_key,
                                                     NULL, &cfg_index);
    ASSERT_UCS_OK(status);

    ucp_worker_print_info(worker(), stdout);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_proto)
