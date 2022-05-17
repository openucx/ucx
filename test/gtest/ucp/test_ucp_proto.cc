/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

#include <common/test.h>
#include <common/mem_buffer.h>

extern "C" {
#include <ucp/core/ucp_rkey.h>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_select.inl>
#include <ucp/core/ucp_worker.inl>
}

class test_ucp_proto : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_RMA);
    }

protected:
    void do_mem_reg(ucp_datatype_iter_t *dt_iter, ucp_md_map_t md_map);

    ucp_md_map_t get_md_map(ucs_memory_type_t mem_type);

    void test_dt_iter_mem_reg(ucs_memory_type_t mem_type, size_t size,
                              ucp_md_map_t md_map);

    virtual void init() {
        modify_config("PROTO_ENABLE", "y");
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
    }

    ucp_context_h context() {
        return sender().ucph();
    }

    ucp_worker_h worker() {
        return sender().worker();
    }
};

ucp_md_map_t test_ucp_proto::get_md_map(ucs_memory_type_t mem_type)
{
    ucp_md_map_t md_map = 0;

    for (ucp_md_index_t md_index = 0; md_index < context()->num_mds;
         ++md_index) {
        const uct_md_attr_t *md_attr = &context()->tl_mds[md_index].attr;
        if ((md_attr->cap.flags & UCT_MD_FLAG_REG) &&
            (md_attr->cap.reg_mem_types & UCS_BIT(mem_type)) &&
            (ucs_popcount(md_map) < UCP_MAX_OP_MDS)) {
            md_map |= UCS_BIT(md_index);
        }
    }
    return md_map;
}

void test_ucp_proto::do_mem_reg(ucp_datatype_iter_t *dt_iter,
                                ucp_md_map_t md_map)
{
    ucp_datatype_iter_mem_reg(context(), dt_iter, md_map, UCT_MD_MEM_ACCESS_ALL,
                              UCP_DT_MASK_ALL);
    ucp_datatype_iter_mem_dereg(context(), dt_iter, UCP_DT_MASK_ALL);
}

void test_ucp_proto::test_dt_iter_mem_reg(ucs_memory_type_t mem_type,
                                          size_t size, ucp_md_map_t md_map)
{
    const double test_time_sec = 1.0;
    mem_buffer buffer(size, mem_type);

    ucp_datatype_iter_t dt_iter;
    uint8_t sg_count;
    ucp_datatype_iter_init(context(), buffer.ptr(), size, UCP_DATATYPE_CONTIG,
                           size, 1, &dt_iter, &sg_count);

    ucs_time_t start_time = ucs_get_time();
    ucs_time_t deadline   = start_time + ucs_time_from_sec(test_time_sec);
    ucs_time_t end_time   = start_time;
    unsigned count        = 0;
    do {
        do_mem_reg(&dt_iter, md_map);
        ++count;
        if ((count % 8) == 0) {
            end_time = ucs_get_time();
        }
    } while (end_time < deadline);

    char memunits_str[32];
    UCS_TEST_MESSAGE << ucs_memory_type_names[mem_type] << " "
                     << ucs_memunits_to_str(size, memunits_str,
                                            sizeof(memunits_str))
                     << " md_map 0x" << std::hex << md_map << std::dec
                     << " registration time: "
                     << (ucs_time_to_nsec(end_time - start_time) / count)
                     << " nsec";
}

UCS_TEST_P(test_ucp_proto, dump_protocols) {
    ucp_proto_select_param_t select_param;
    ucs_string_buffer_t strb;

    select_param.op_id      = UCP_OP_ID_TAG_SEND;
    select_param.op_flags   = 0;
    select_param.dt_class   = UCP_DATATYPE_CONTIG;
    select_param.mem_type   = UCS_MEMORY_TYPE_HOST;
    select_param.sys_dev    = UCS_SYS_DEVICE_ID_UNKNOWN;
    select_param.sg_count   = 1;
    select_param.padding    = 0;

    ucs_string_buffer_init(&strb);
    ucp_proto_select_param_str(&select_param, ucp_operation_names, &strb);
    UCS_TEST_MESSAGE << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);

    ucp_worker_h worker                   = sender().worker();
    ucp_worker_cfg_index_t ep_cfg_index   = sender().ep()->cfg_index;
    ucp_worker_cfg_index_t rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;

    auto select_elem = ucp_proto_select_lookup(
            worker, &worker->ep_config[ep_cfg_index].proto_select, ep_cfg_index,
            rkey_cfg_index, &select_param, 0);
    EXPECT_NE(nullptr, select_elem);

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

UCS_TEST_P(test_ucp_proto, dt_iter_mem_reg)
{
    static const size_t buffer_size = 8192;

    for (auto mem_type : mem_buffer::supported_mem_types()) {
        ucp_md_map_t md_map = get_md_map(mem_type);
        if (md_map == 0) {
            UCS_TEST_MESSAGE << "No memory domains can register "
                             << ucs_memory_type_names[mem_type] << " memory";
            continue;
        }

        test_dt_iter_mem_reg(mem_type, buffer_size, md_map);
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_proto)
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_proto, shm_ipc,
                                        "shm,cuda_ipc,rocm_ipc")
