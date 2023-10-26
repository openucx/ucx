/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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
#include <ucs/datastruct/linear_func.h>
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

    static ucp_rkey_config_key_t create_rkey_config_key(ucp_md_map_t md_map);
};

ucp_md_map_t test_ucp_proto::get_md_map(ucs_memory_type_t mem_type)
{
    return context()->reg_md_map[mem_type] &
    /* ucp_datatype_iter_mem_reg() always goes directly to registration cache */
           context()->cache_md_map[mem_type];
}

void test_ucp_proto::do_mem_reg(ucp_datatype_iter_t *dt_iter,
                                ucp_md_map_t md_map)
{
    ucp_datatype_iter_mem_reg(context(), dt_iter, md_map, UCT_MD_MEM_ACCESS_ALL,
                              UCP_DT_MASK_ALL);
    ucp_datatype_iter_mem_dereg(dt_iter, UCP_DT_MASK_ALL);
}

void test_ucp_proto::test_dt_iter_mem_reg(ucs_memory_type_t mem_type,
                                          size_t size, ucp_md_map_t md_map)
{
    const double test_time_sec = 1.0;
    mem_buffer buffer(size, mem_type);

    ucp_datatype_iter_t dt_iter;
    uint8_t sg_count;
    /* Pass empty param argument to disable memh initialization */
    ucp_request_param_t param;
    param.op_attr_mask = 0;

    ucp_datatype_iter_init(context(), buffer.ptr(), size, UCP_DATATYPE_CONTIG,
                           size, 1, &dt_iter, &sg_count, &param);

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

ucp_rkey_config_key_t
test_ucp_proto::create_rkey_config_key(ucp_md_map_t md_map)
{
    ucp_rkey_config_key_t rkey_config_key;

    rkey_config_key.ep_cfg_index       = 0;
    rkey_config_key.md_map             = md_map;
    rkey_config_key.mem_type           = UCS_MEMORY_TYPE_HOST;
    rkey_config_key.sys_dev            = UCS_SYS_DEVICE_ID_UNKNOWN;
    rkey_config_key.unreachable_md_map = 0;

    return rkey_config_key;
}

UCS_TEST_P(test_ucp_proto, dump_protocols) {
    ucp_proto_select_param_t select_param;
    ucs_string_buffer_t strb;

    select_param.op_id_flags   = UCP_OP_ID_TAG_SEND;
    select_param.op_attr       = 0;
    select_param.dt_class      = UCP_DATATYPE_CONTIG;
    select_param.mem_type      = UCS_MEMORY_TYPE_HOST;
    select_param.sys_dev       = UCS_SYS_DEVICE_ID_UNKNOWN;
    select_param.sg_count      = 1;
    select_param.op.padding[0] = 0;
    select_param.op.padding[1] = 0;

    ucs_string_buffer_init(&strb);
    ucp_proto_select_param_str(&select_param, ucp_operation_names, &strb);
    UCS_TEST_MESSAGE << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);

    ucp_worker_h worker                   = sender().worker();
    ucp_worker_cfg_index_t ep_cfg_index   = sender().ep()->cfg_index;
    ucp_worker_cfg_index_t rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;

    auto proto_select = &ucs_array_elem(&worker->ep_config,
                                        ep_cfg_index).proto_select;
    auto select_elem  = ucp_proto_select_lookup(worker, proto_select,
                                                ep_cfg_index, rkey_cfg_index,
                                                &select_param, 0);
    EXPECT_NE(nullptr, select_elem);

    ucp_ep_print_info(sender().ep(), stdout);
}

UCS_TEST_P(test_ucp_proto, rkey_config) {
    ucp_rkey_config_key_t rkey_config_key = create_rkey_config_key(0);
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

    rkey_config_key = create_rkey_config_key(1);

    /* different configuration should return different index */
    ucp_worker_cfg_index_t cfg_index3;
    status = ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                        &cfg_index3);
    ASSERT_UCS_OK(status);

    EXPECT_NE(static_cast<int>(cfg_index1), static_cast<int>(cfg_index3));
}

UCS_TEST_P(test_ucp_proto, worker_print_info_rkey)
{
    ucp_rkey_config_key_t rkey_config_key = create_rkey_config_key(0);

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

class test_perf_node : public test_ucp_proto {
};

UCS_TEST_P(test_perf_node, basic)
{
    static const std::string nullstr = "(null)";

    EXPECT_EQ(nullstr, ucp_proto_perf_node_name(NULL));
    EXPECT_EQ(nullstr, ucp_proto_perf_node_desc(NULL));

    ucp_proto_perf_node_t *n1 = ucp_proto_perf_node_new_compose("n1", "node%d",
                                                                1);
    ASSERT_NE(nullptr, n1);
    EXPECT_EQ(std::string("n1"), ucp_proto_perf_node_name(n1));
    EXPECT_EQ(std::string("node1"), ucp_proto_perf_node_desc(n1));

    ucp_proto_perf_node_t *n2 = ucp_proto_perf_node_new_select("n2", 0,
                                                               "node%d", 2);
    ASSERT_NE(nullptr, n2);
    EXPECT_EQ(std::string("n2"), ucp_proto_perf_node_name(n2));
    EXPECT_EQ(std::string("node2"), ucp_proto_perf_node_desc(n2));

    ucp_proto_perf_node_t *n3 = ucp_proto_perf_node_new_data("n3", "node%d", 3);
    ASSERT_NE(nullptr, n3);
    EXPECT_EQ(std::string("n3"), ucp_proto_perf_node_name(n3));
    EXPECT_EQ(std::string("node3"), ucp_proto_perf_node_desc(n3));

    ucp_proto_perf_node_add_data(n3, "zero", UCS_LINEAR_FUNC_ZERO);
    ucp_proto_perf_node_add_data(n3, "one", ucs_linear_func_make(0, 1));
    ucp_proto_perf_node_add_scalar(n3, "lat", 1e-6);
    ucp_proto_perf_node_add_bandwidth(n3, "bw", UCS_MBYTE);

    /* NULL child is ignored */
    ucp_proto_perf_node_add_child(n1, NULL);
    ucp_proto_perf_node_add_child(n2, NULL);
    ucp_proto_perf_node_add_child(n3, NULL);

    ucp_proto_perf_node_t *tmp = NULL;
    ucp_proto_perf_node_own_child(n3, &tmp);

    /* NULL parent is ignored */
    ucp_proto_perf_node_add_child(NULL, n1);
    ucp_proto_perf_node_add_child(NULL, n2);
    ucp_proto_perf_node_add_child(NULL, n3);
    ucp_proto_perf_node_add_child(NULL, NULL);

    /* NULL owner should remove extra ref */
    ucp_proto_perf_node_t *n2_ref = n2;
    ucp_proto_perf_node_ref(n2_ref);
    ucp_proto_perf_node_own_child(NULL, &n2_ref);
    EXPECT_EQ(nullptr, n2_ref);

    /* NULL node is ignored */
    ucp_proto_perf_node_add_data(NULL, "ignored", UCS_LINEAR_FUNC_ZERO);
    ucp_proto_perf_node_add_scalar(NULL, "ignored", 1.0);

    /* n1 -> n2 -> n3 */
    ucp_proto_perf_node_own_child(n2, &n3); /* Dropped extra ref to n3 */
    ucp_proto_perf_node_add_child(n1, n2);  /* Have 2 references to n2 */

    ucp_proto_perf_node_deref(&n2); /* n3 should still be alive */
    EXPECT_EQ(nullptr, n2);
    EXPECT_EQ(std::string("node3"), ucp_proto_perf_node_desc(n3));

    ucp_proto_perf_node_deref(&n1); /* Release n1,n2,n3 */
    EXPECT_EQ(nullptr, n1);
}

UCS_TEST_P(test_perf_node, replace_node)
{
    ucp_proto_perf_node_t *parent1 = ucp_proto_perf_node_new_data("parent1", "");
    ucp_proto_perf_node_t *child1 = ucp_proto_perf_node_new_data("child1", "");
    ucp_proto_perf_node_t *child2 = ucp_proto_perf_node_new_data("child2", "");

    ucp_proto_perf_node_add_child(parent1, child1);
    ucp_proto_perf_node_add_child(parent1, child2);
    EXPECT_EQ(child1, ucp_proto_perf_node_get_child(parent1, 0));
    EXPECT_EQ(child2, ucp_proto_perf_node_get_child(parent1, 1));

    ucp_proto_perf_node_t *parent2 = ucp_proto_perf_node_new_data("parent2", "");
    ucp_proto_perf_node_t *parent2_ptr = parent2;
    ASSERT_NE(parent2_ptr, parent1);

    ucp_proto_perf_node_replace(&parent1, &parent2);

    /* parent2 variable should be set to NULL */
    EXPECT_EQ(nullptr, parent2);

    /* parent1 variable should be set to parent2 */
    EXPECT_EQ(parent2_ptr, parent1);

    /* Children should be reassigned */
    EXPECT_EQ(child1, ucp_proto_perf_node_get_child(parent2_ptr, 0));
    EXPECT_EQ(child2, ucp_proto_perf_node_get_child(parent2_ptr, 1));

    ucp_proto_perf_node_deref(&parent1);
    ucp_proto_perf_node_deref(&child1);
    ucp_proto_perf_node_deref(&child2);
}


UCS_TEST_P(test_perf_node, replace_null)
{
    /* Replacing NULL with NULL is a no-op */
    {
        ucp_proto_perf_node_t *n1 = nullptr;
        ucp_proto_perf_node_t *n2 = nullptr;
        ucp_proto_perf_node_replace(&n1, &n2);
    }

    /* Replacing a node by NULL should release the node (without leaks) */
    {
        ucp_proto_perf_node_t *parent1 = ucp_proto_perf_node_new_data("parent1", "");
        ucp_proto_perf_node_t *child1 = ucp_proto_perf_node_new_data("child1", "");
        ucp_proto_perf_node_t *child2 = ucp_proto_perf_node_new_data("child2", "");
        ucp_proto_perf_node_own_child(parent1, &child1);
        ucp_proto_perf_node_own_child(parent1, &child2);

        ucp_proto_perf_node_t *null_node = nullptr;
        ucp_proto_perf_node_replace(&parent1, &null_node);
        EXPECT_EQ(nullptr, parent1);
    }

    /* Replacing a NULL should swap variables */
    {
        ucp_proto_perf_node_t *node = ucp_proto_perf_node_new_data("node", "");
        ucp_proto_perf_node_t *null_node = nullptr;
        ucp_proto_perf_node_replace(&null_node, &node);
        EXPECT_EQ(nullptr, node);
        EXPECT_NE(nullptr, null_node);
        ucp_proto_perf_node_deref(&null_node);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_perf_node, all, "all")
