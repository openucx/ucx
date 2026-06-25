/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

#include <common/test.h>
#include <common/mem_buffer.h>
#include <cstring>
#include <unordered_map>
#include <memory>

extern "C" {
#include <ucp/core/ucp_rkey.h>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_perf.h>
#include <ucp/proto/proto_init.h>
#include <ucp/rndv/proto_rndv.h>
#include <ucs/datastruct/linear_func.h>
#include <ucp/proto/proto_select.inl>
#include <ucp/core/ucp_worker.inl>
#include <uct/api/v2/uct_v2.h>
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
    rkey_config_key.flags              = 0;

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
    select_param.op.mem_flags  = UCS_MEM_FLAG_REGISTRABLE;
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

UCS_TEST_P(test_ucp_proto, buffer_copy_host_memory_class)
{
    ucp_proto_common_init_params_t params = {};

    params.reg_mem_info.type = UCS_MEMORY_TYPE_HOST;
    EXPECT_EQ(UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
              ucp_proto_init_buffer_copy_host_memory_class(
                      &params, UCS_MEMORY_TYPE_HOST));

    params.flags = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY;
    EXPECT_EQ(UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
              ucp_proto_init_buffer_copy_host_memory_class(
                      &params, UCS_MEMORY_TYPE_HOST));

    params.flags = UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR;
    EXPECT_EQ(UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
              ucp_proto_init_buffer_copy_host_memory_class(
                      &params, UCS_MEMORY_TYPE_HOST));

    params.flags             = 0;
    params.reg_mem_info.type = UCS_MEMORY_TYPE_UNKNOWN;
    EXPECT_EQ(UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
              ucp_proto_init_buffer_copy_host_memory_class(
                      &params, UCS_MEMORY_TYPE_HOST));

    params.reg_mem_info.type = UCS_MEMORY_TYPE_HOST;
    EXPECT_EQ(UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
              ucp_proto_init_buffer_copy_host_memory_class(
                      &params, UCS_MEMORY_TYPE_CUDA));

    params.reg_mem_info.type = UCS_MEMORY_TYPE_CUDA;
    EXPECT_EQ(UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
              ucp_proto_init_buffer_copy_host_memory_class(
                      &params, UCS_MEMORY_TYPE_HOST));
}

UCS_TEST_P(test_ucp_proto, buffer_copy_flags_attached_host_staging)
{
    const unsigned attached_flag =
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING;
    const unsigned skip_send_pre_flag =
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_SKIP_SEND_PRE_OVERHEAD;
    const unsigned attached_skip_send_pre_flags =
            attached_flag | skip_send_pre_flag;

    EXPECT_EQ(attached_flag, ucp_proto_init_buffer_copy_flags(
            UCS_MEMORY_TYPE_HOST,
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
            UCS_MEMORY_TYPE_CUDA, UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
            attached_flag));
    EXPECT_EQ(attached_skip_send_pre_flags, ucp_proto_init_buffer_copy_flags(
            UCS_MEMORY_TYPE_HOST,
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
            UCS_MEMORY_TYPE_CUDA, UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
            attached_skip_send_pre_flags));
    EXPECT_EQ(attached_skip_send_pre_flags, ucp_proto_init_buffer_copy_flags(
            UCS_MEMORY_TYPE_CUDA, UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
            UCS_MEMORY_TYPE_HOST,
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
            attached_skip_send_pre_flags));
    EXPECT_EQ(UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE,
              ucp_proto_init_buffer_copy_flags(
            UCS_MEMORY_TYPE_HOST,
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
            UCS_MEMORY_TYPE_CUDA_MANAGED,
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
            attached_skip_send_pre_flags));
    EXPECT_EQ(UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE,
              ucp_proto_init_buffer_copy_flags(
            UCS_MEMORY_TYPE_HOST,
            UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
            UCS_MEMORY_TYPE_ROCM, UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
            attached_skip_send_pre_flags));
    EXPECT_EQ(UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE,
              ucp_proto_init_buffer_copy_flags(
                      UCS_MEMORY_TYPE_HOST,
                      UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
                      UCS_MEMORY_TYPE_CUDA,
                      UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
                      skip_send_pre_flag));
    EXPECT_EQ(UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE,
              ucp_proto_init_buffer_copy_flags(
                      UCS_MEMORY_TYPE_HOST,
                      UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
                      UCS_MEMORY_TYPE_CUDA,
                      UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
                      attached_skip_send_pre_flags));
    EXPECT_EQ(UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE,
              ucp_proto_init_buffer_copy_flags(
                      UCS_MEMORY_TYPE_HOST,
                      UCT_PERF_ATTR_HOST_MEMORY_CLASS_REGISTERED_LOCKED,
                      UCS_MEMORY_TYPE_HOST,
                      UCT_PERF_ATTR_HOST_MEMORY_CLASS_UNKNOWN,
                      attached_skip_send_pre_flags));
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

UCS_TEST_P(test_ucp_proto, memtype_copy_shared_divisor)
{
    const ucs_sys_device_t sys_dev0 = 0;
    const ucs_sys_device_t sys_dev1 = 1;
    const ucs_sys_device_t sys_dev2 = 2;
    uct_perf_attr_t perf_attr       = {};
    int orig_ppn                    = context()->config.est_num_ppn;

    auto set_scope = [&](uct_perf_attr_bandwidth_shared_scope_t scope,
                         ucs_sys_device_t sys_dev) {
        perf_attr.bandwidth_shared_scope      = scope;
        perf_attr.bandwidth_shared_sys_device = sys_dev;
    };

    context()->config.est_num_ppn = 2;
    set_scope(UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_SYS_DEVICE, sys_dev1);
    EXPECT_EQ(1u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));

    context()->config.est_num_ppn = 4;
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));

    context()->config.est_num_ppn = 2;
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_CUDA, UCS_SYS_DEVICE_ID_UNKNOWN,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_HOST,
            UCS_SYS_DEVICE_ID_UNKNOWN, UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));
    EXPECT_EQ(1u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_HOST,
            UCS_SYS_DEVICE_ID_UNKNOWN, UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_HOST,
            UCS_SYS_DEVICE_ID_UNKNOWN, UCS_MEMORY_TYPE_CUDA,
            UCS_SYS_DEVICE_ID_UNKNOWN,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING));
    EXPECT_EQ(1u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCS_MEMORY_TYPE_HOST, UCS_SYS_DEVICE_ID_UNKNOWN,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_HOST, UCS_SYS_DEVICE_ID_UNKNOWN,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_HOST,
            UCS_SYS_DEVICE_ID_UNKNOWN, UCS_MEMORY_TYPE_CUDA_MANAGED, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_HOST,
            UCS_SYS_DEVICE_ID_UNKNOWN, UCS_MEMORY_TYPE_ROCM, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_ATTACHED_HOST_STAGING));
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_CUDA, sys_dev2,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));

    set_scope(UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_SYS_DEVICE,
              UCS_SYS_DEVICE_ID_UNKNOWN);
    EXPECT_EQ(2u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));

    set_scope(UCT_PERF_ATTR_BANDWIDTH_SHARED_SCOPE_NODE,
              UCS_SYS_DEVICE_ID_UNKNOWN);
    EXPECT_EQ(0u, ucp_proto_init_memtype_copy_shared_divisor(
            worker(), &perf_attr, UCS_MEMORY_TYPE_CUDA, sys_dev0,
            UCS_MEMORY_TYPE_CUDA, sys_dev1,
            UCP_PROTO_INIT_BUFFER_COPY_FLAG_NONE));

    context()->config.est_num_ppn = orig_ppn;
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

class test_ucp_proto_cuda_async_non_reg : public test_ucp_proto {
protected:
    /* Async CUDA memory that is CUDA_MANAGED but not registrable. Verifies UCP
     * keeps it off the IB/HCA MDs at the registration and protocol layers. */
    class scoped_reg_md_map {
    public:
        scoped_reg_md_map(ucp_context_h context, ucs_memory_type_t mem_type,
                          ucp_md_map_t md_map) :
            m_context(context),
            m_mem_type(mem_type),
            m_orig_reg_md_map(context->reg_md_map[mem_type])
        {
            context->reg_md_map[mem_type] = m_orig_reg_md_map | md_map;
        }

        ~scoped_reg_md_map()
        {
            m_context->reg_md_map[m_mem_type] = m_orig_reg_md_map;
        }

    private:
        ucp_context_h m_context;
        ucs_memory_type_t m_mem_type;
        ucp_md_map_t m_orig_reg_md_map;
    };

    static ucp_md_map_t
    get_required_mem_flags_md_map(ucp_context_h context,
                                  ucs_memory_type_t mem_type,
                                  uint8_t mem_flags);

    static const ucp_proto_config_t *
    get_rma_get_rndv_remote_proto_config(
            const ucp_proto_select_elem_t *select_elem, size_t length);

    const ucp_proto_config_t *
    select_get_rndv_remote_proto_config(ucp_proto_select_t *proto_select,
                                        ucp_worker_cfg_index_t ep_cfg_index,
                                        ucp_worker_cfg_index_t rkey_cfg_index,
                                        const ucp_memory_info_t *mem_info,
                                        size_t length);
};

ucp_md_map_t test_ucp_proto_cuda_async_non_reg::get_required_mem_flags_md_map(
        ucp_context_h context, ucs_memory_type_t mem_type, uint8_t mem_flags)
{
    ucp_md_map_t md_map = context->reg_md_map[mem_type] &
                          context->cache_md_map[mem_type];
    ucp_md_map_t result = 0;
    ucp_md_index_t md_index;

    ucs_for_each_bit(md_index, md_map) {
        if (context->tl_mds[md_index].attr.required_mem_flags & mem_flags) {
            result |= UCS_BIT(md_index);
        }
    }

    return result;
}

const ucp_proto_config_t *
test_ucp_proto_cuda_async_non_reg::get_rma_get_rndv_remote_proto_config(
        const ucp_proto_select_elem_t *select_elem, size_t length)
{
    const ucp_proto_threshold_elem_t *thresh =
            ucp_proto_thresholds_search_slow(select_elem->thresholds, length);
    const ucp_proto_config_t *proto_config = &thresh->proto_config;
    const ucp_proto_rndv_ctrl_priv_t *rpriv;

    if (std::strcmp(proto_config->proto->name, "get/rndv") != 0) {
        return nullptr;
    }

    rpriv = static_cast<const ucp_proto_rndv_ctrl_priv_t*>(
            proto_config->priv);
    return &rpriv->remote_proto_config;
}

const ucp_proto_config_t *
test_ucp_proto_cuda_async_non_reg::select_get_rndv_remote_proto_config(
        ucp_proto_select_t *proto_select, ucp_worker_cfg_index_t ep_cfg_index,
        ucp_worker_cfg_index_t rkey_cfg_index,
        const ucp_memory_info_t *mem_info, size_t length)
{
    ucp_proto_select_param_t select_param;
    const ucp_proto_select_elem_t *select_elem;

    ucp_proto_select_param_init(&select_param, UCP_OP_ID_GET, 0, 0,
                                UCP_DATATYPE_CONTIG, mem_info, 1);
    select_elem = ucp_proto_select_lookup_slow(worker(), proto_select, 0,
                                               ep_cfg_index, rkey_cfg_index,
                                               &select_param);
    if (select_elem == nullptr) {
        return nullptr;
    }

    return get_rma_get_rndv_remote_proto_config(select_elem, length);
}

UCS_TEST_P(test_ucp_proto_cuda_async_non_reg, cuda_async_registrable_filter)
{
    constexpr size_t buffer_size  = 8192;
    ucp_request_param_t param     = {};
    uct_md_mem_attr_t v1_mem_attr = {};
    int v1_queried                = 0;
    ucp_md_map_t hca_md_map;
    ucp_datatype_iter_t dt_iter;
    ucs_memory_type_t mem_type;
    ucs_status_t status;
    ucp_md_index_t i, md_index;
    uint8_t sg_count;

    if (!mem_buffer::is_async_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("CUDA async allocation is not supported");
    }

    scoped_async_cuda_buffer buffer(buffer_size);

    for (i = 0; i < context()->num_mem_type_detect_mds; ++i) {
        md_index               = context()->mem_type_detect_mds[i];
        v1_mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;
        status = uct_md_mem_query(context()->tl_mds[md_index].md, buffer.ptr(),
                                  buffer_size, &v1_mem_attr);
        if ((status == UCS_OK) &&
            ((v1_mem_attr.mem_type == UCS_MEMORY_TYPE_CUDA) ||
             (v1_mem_attr.mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED))) {
            v1_queried = 1;
            break;
        }
    }
    ASSERT_TRUE(v1_queried);

    ASSERT_UCS_OK(ucp_datatype_iter_init(context(), buffer.ptr(), buffer_size,
                                         UCP_DATATYPE_CONTIG, buffer_size, 1,
                                         &dt_iter, &sg_count, &param));

    mem_type = static_cast<ucs_memory_type_t>(dt_iter.mem_info.type);
    if (mem_type != UCS_MEMORY_TYPE_CUDA_MANAGED) {
        UCS_TEST_SKIP_R("CUDA async memory is not classified as CUDA managed");
    }

    hca_md_map = get_required_mem_flags_md_map(context(), UCS_MEMORY_TYPE_CUDA,
                                               UCS_MEM_FLAG_REGISTRABLE);
    hca_md_map &= context()->cache_md_map[mem_type];
    if (hca_md_map == 0) {
        UCS_TEST_SKIP_R("no CUDA-registering IB/HCA MDs");
    }

    /* Precondition: the async CUDA buffer is detected as non-registrable. */
    ASSERT_EQ(0, dt_iter.mem_info.flags & UCS_MEM_FLAG_REGISTRABLE);

    /* Even with the IB/HCA MDs force-enabled, the REGISTRABLE filter must keep
     * them out of memh->md_map. */
    scoped_reg_md_map reg_md_map(context(), mem_type, hca_md_map);
    status = ucp_datatype_iter_mem_reg(context(), &dt_iter, hca_md_map,
                                       UCT_MD_MEM_ACCESS_RMA, UCP_DT_MASK_ALL);
    ASSERT_UCS_OK(status);
    ASSERT_NE(nullptr, dt_iter.type.contig.memh);
    EXPECT_EQ(static_cast<ucp_md_map_t>(0),
              dt_iter.type.contig.memh->md_map & hca_md_map);
    ucp_datatype_iter_mem_dereg(&dt_iter, UCP_DT_MASK_ALL);
}

UCS_TEST_P(test_ucp_proto_cuda_async_non_reg,
           cuda_async_rndv_get_zcopy_proto_filter, "RNDV_THRESH=0",
           "RNDV_SCHEME=get_zcopy")
{
    /* Keep the real CUDA allocation small, but inspect a large protocol range
     * where RMA GET/RNDV is selected. */
    constexpr size_t buffer_size = 8192;
    constexpr size_t select_size = UCS_GBYTE;
    ucp_request_param_t param    = {};
    ucp_datatype_iter_t dt_iter;
    ucp_memory_info_t mem_info;
    const ucp_ep_config_t *ep_config;
    ucp_proto_select_t *proto_select;
    const uct_iface_attr_t *iface_attr;
    ucp_worker_cfg_index_t ep_cfg_index, rkey_cfg_index;
    ucp_md_map_t hca_md_map, lane_md_map, rkey_md_map;
    ucp_lane_index_t lane;
    ucp_md_index_t md_index;
    uint8_t sg_count;
    const ucp_proto_config_t *no_flag_remote_proto_config;
    const ucp_proto_config_t *can_reg_remote_proto_config;

    if (!mem_buffer::is_async_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("CUDA async allocation is not supported");
    }

    scoped_async_cuda_buffer buffer(buffer_size);

    ASSERT_UCS_OK(ucp_datatype_iter_init(context(), buffer.ptr(), buffer_size,
                                         UCP_DATATYPE_CONTIG, buffer_size, 1,
                                         &dt_iter, &sg_count, &param));

    mem_info = dt_iter.mem_info;
    if (mem_info.type != UCS_MEMORY_TYPE_CUDA_MANAGED) {
        UCS_TEST_SKIP_R("CUDA async memory is not classified as CUDA managed");
    }

    hca_md_map = get_required_mem_flags_md_map(context(), UCS_MEMORY_TYPE_CUDA,
                                               UCS_MEM_FLAG_REGISTRABLE);
    hca_md_map &= context()->cache_md_map[mem_info.type];
    if (hca_md_map == 0) {
        UCS_TEST_SKIP_R("no CUDA-registering IB/HCA MDs");
    }

    ep_config   = &ucs_array_elem(&worker()->ep_config,
                                  sender().ep()->cfg_index);
    lane_md_map = 0;
    rkey_md_map = 0;
    for (lane = 0; lane < ep_config->key.num_lanes; ++lane) {
        md_index = context()->tl_rscs[
                ep_config->key.lanes[lane].rsc_index].md_index;
        iface_attr = ucp_worker_iface_get_attr(
                worker(), ep_config->key.lanes[lane].rsc_index);
        if ((hca_md_map & UCS_BIT(md_index)) &&
            (context()->tl_mds[md_index].attr.flags & UCT_MD_FLAG_NEED_RKEY) &&
            (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY)) {
            lane_md_map |= UCS_BIT(md_index);
            rkey_md_map |= UCS_BIT(ep_config->key.lanes[lane].dst_md_index);
        }
    }

    if (lane_md_map == 0) {
        UCS_TEST_SKIP_R("no endpoint lanes support IB/HCA get zcopy");
    }

    ASSERT_EQ(0, mem_info.flags & UCS_MEM_FLAG_REGISTRABLE);

    ep_cfg_index = sender().ep()->cfg_index;

    ucp_rkey_config_key_t rkey_config_key = create_rkey_config_key(rkey_md_map);
    rkey_config_key.ep_cfg_index = ep_cfg_index;
    ASSERT_UCS_OK(ucp_worker_rkey_config_get(worker(), &rkey_config_key, NULL,
                                             &rkey_cfg_index));

    proto_select =
            &ucs_array_elem(&worker()->ep_config, ep_cfg_index).proto_select;

    scoped_reg_md_map reg_md_map(context(),
                                 static_cast<ucs_memory_type_t>(mem_info.type),
                                 lane_md_map);

    /* Select the GET/RNDV proto twice, toggling only REGISTRABLE, and check
     * that direct GET zcopy is selected only for registrable memory. */
    no_flag_remote_proto_config = select_get_rndv_remote_proto_config(
            proto_select, ep_cfg_index, rkey_cfg_index, &mem_info, select_size);
    ASSERT_NE(nullptr, no_flag_remote_proto_config);

    mem_info.flags |= UCS_MEM_FLAG_REGISTRABLE;
    can_reg_remote_proto_config = select_get_rndv_remote_proto_config(
            proto_select, ep_cfg_index, rkey_cfg_index, &mem_info, select_size);
    ASSERT_NE(nullptr, can_reg_remote_proto_config);

    EXPECT_STRNE("rndv/get/zcopy",
                 no_flag_remote_proto_config->proto->name);
    EXPECT_STREQ("rndv/get/zcopy",
                 can_reg_remote_proto_config->proto->name);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_cuda_async_non_reg, rcx,
                              "rc_x,cuda_copy")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_cuda_async_non_reg, rcv,
                              "rc_v,cuda_copy")

class test_perf_node : public test_ucp_proto {
};

UCS_TEST_P(test_perf_node, basic)
{
    static const std::string nullstr = "(null)";

    EXPECT_EQ(nullstr, ucp_proto_perf_node_name(NULL));
    EXPECT_EQ(nullstr, ucp_proto_perf_node_desc(NULL));

    ucp_proto_perf_node_t *n1 = ucp_proto_perf_node_new_data("n1", "node%d", 1);
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

    ucp_proto_perf_node_t *tmp = NULL;
    ucp_proto_perf_node_own_child(n1, &tmp);

    /* NULL parent is ignored */
    ucp_proto_perf_node_add_child(NULL, n1);
    ucp_proto_perf_node_add_child(NULL, n2);
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

static std::ostream &operator<<(std::ostream &os, const ucp_proto_perf_t *perf)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    ucp_proto_perf_str(perf, &strb);
    auto &ret = os << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);
    return ret;
}

static std::ostream &
operator<<(std::ostream &os, const ucp_proto_flat_perf_t *flat_perf)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    ucp_proto_flat_perf_str(flat_perf, &strb);
    auto &ret = os << ucs_string_buffer_cstr(&strb);
    ucs_string_buffer_cleanup(&strb);
    return ret;
}

static std::ostream &operator<<(std::ostream &os, const ucs_linear_func_t &func)
{
    char buffer[128];

    ucs_snprintf_safe(buffer, sizeof(buffer), UCP_PROTO_PERF_FUNC_FMT,
                      UCP_PROTO_PERF_FUNC_ARG(&func));
    return os << buffer;
}

class test_proto_perf : public ucs::test {
public:
    using perf_ptr_t         = std::shared_ptr<ucp_proto_perf_t>;
    using flat_perf_ptr_t    = std::shared_ptr<ucp_proto_flat_perf_t>;
    using perf_factors_map_t = std::unordered_map<int, ucs_linear_func_t>;
protected:
    virtual void cleanup() override
    {
        m_envelope_flat_perf.reset();
        m_sum_flat_perf.reset();
        m_perf.reset();
    }

    static perf_ptr_t create()
    {
        ucp_proto_perf_t *perf;
        ASSERT_UCS_OK(ucp_proto_perf_create("test", &perf));
        return make_perf_ptr(perf);
    }

    perf_ptr_t aggregate(const std::vector<perf_ptr_t> &perfs)
    {
        std::vector<ucp_proto_perf_t*> perfs_ptr_vec;
        std::transform(perfs.begin(), perfs.end(),
                       std::back_inserter(perfs_ptr_vec),
                       [](const perf_ptr_t &p) { return p.get(); });

        ucp_proto_perf_t *perf;
        ASSERT_UCS_OK(ucp_proto_perf_aggregate("aggregate",
                                               perfs_ptr_vec.data(),
                                               perfs_ptr_vec.size(), &perf));
        return make_perf_ptr(perf);
    }

    static void add_funcs(perf_ptr_t perf, size_t start, size_t end,
                          const perf_factors_map_t &factors)
    {
        ucp_proto_perf_factors_t perf_factors =
                UCP_PROTO_PERF_FACTORS_INITIALIZER;
        ucp_proto_perf_node_t *perf_node;

        for (auto &f : factors) {
            perf_factors[f.first] = f.second;
        }

        perf_node = ucp_proto_perf_node_new_data("test", "");
        ASSERT_UCS_OK(ucp_proto_perf_add_funcs(perf.get(), start, end,
                                               perf_factors, perf_node, NULL));
    }

    static void add_func(perf_ptr_t perf, size_t start, size_t end,
                         ucp_proto_perf_factor_id_t factor_id,
                         ucs_linear_func_t func)
    {
        add_funcs(perf, start, end, {{factor_id, func}});
    }

    void add_func(size_t start, size_t end,
                  ucp_proto_perf_factor_id_t factor_id, ucs_linear_func_t func)
    {
        add_func(m_perf, start, end, factor_id, func);
    }

    static ucp_proto_perf_segment_t *find_lb(perf_ptr_t perf, size_t start)
    {
        return ucp_proto_perf_find_segment_lb(perf.get(), start);
    }

    static const ucp_proto_flat_perf_range_t *
    find_lb(flat_perf_ptr_t flat_perf, size_t start)
    {
        return ucp_proto_flat_perf_find_lb(flat_perf.get(), start);
    }

    void expect_empty_range(size_t start, size_t end) const
    {
        expect_perf_empty_range(m_perf, start, end);
        expect_perf_empty_range(m_sum_flat_perf, start, end);
        expect_perf_empty_range(m_envelope_flat_perf, start, end);
    }

    void expect_perf(size_t start, size_t end,
                     const perf_factors_map_t &factors) const
    {
        ASSERT_FALSE(ucp_proto_perf_is_empty(m_perf.get()));

        const ucp_proto_perf_segment_t *seg = find_lb(m_perf, start);
        ASSERT_NE(nullptr, seg) << "start=" << start;

        EXPECT_GE(start, ucp_proto_perf_segment_start(seg));
        EXPECT_LE(end, ucp_proto_perf_segment_end(seg));

        int factor_id = UCP_PROTO_PERF_FACTOR_LOCAL_CPU;
        for (; factor_id < UCP_PROTO_PERF_FACTOR_LAST; ++factor_id) {
            ucs_linear_func_t expected_func = UCS_LINEAR_FUNC_ZERO;
            auto entry                      = factors.find(factor_id);
            if (entry != std::end(factors)) {
                expected_func = entry->second;
            }

            auto segment_func = ucp_proto_perf_segment_func(
                    seg, (ucp_proto_perf_factor_id_t)factor_id);
            EXPECT_TRUE(
                    ucs_linear_func_is_equal(expected_func, segment_func, 1e-9))
                    << "start=" << start << " end=" << end
                    << " factor_id=" << factor_id
                    << " expected_func=" << expected_func
                    << " segment_func=" << segment_func;
        }

        expect_flat_perf(start, end, factors);
    }

    void make_flat_perf()
    {
        ucp_proto_flat_perf_t *envelope_perf, *sum_perf;

        ASSERT_NE(m_perf.get(), nullptr) << "Perf should be initialized";
        ASSERT_UCS_OK(ucp_proto_perf_envelope(m_perf.get(), 0, &envelope_perf));
        ASSERT_UCS_OK(ucp_proto_perf_sum(m_perf.get(), &sum_perf));

        m_envelope_flat_perf = make_flat_perf_ptr(envelope_perf);
        m_sum_flat_perf      = make_flat_perf_ptr(sum_perf);
    }

    static ucs_linear_func_t perf_func(double overhead_nsec, double bw_mbs)
    {
        return {overhead_nsec * 1e-9, 1.0 / (bw_mbs * UCS_MBYTE)};
    }

    void print_perf() const
    {
        UCS_TEST_MESSAGE << "perf:     " << m_perf.get();
        UCS_TEST_MESSAGE << "envelope: " << m_envelope_flat_perf.get();
        UCS_TEST_MESSAGE << "sum:      " << m_sum_flat_perf.get();
    }

    static perf_ptr_t make_perf_ptr(ucp_proto_perf_t *perf)
    {
        return {perf, ucp_proto_perf_destroy};
    }

private:
    static size_t get_start(ucp_proto_perf_segment_t *seg)
    {
        return ucp_proto_perf_segment_start(seg);
    }

    static size_t get_start(const ucp_proto_flat_perf_range_t *range)
    {
        return range->start;
    }

    template<typename PerfPtrType>
    void
    expect_perf_empty_range(PerfPtrType perf, size_t start, size_t end) const
    {
        ASSERT_NE(perf.get(), nullptr);
        auto *seg = find_lb(perf, start);

        EXPECT_TRUE((seg == nullptr) || (end < get_start(seg)))
                << "perf=" << perf.get() << " start=" << start
                << " end=" << end;
    }

    void expect_flat_range(flat_perf_ptr_t flat_perf, size_t start, size_t end,
                           ucs_linear_func_t exp_func) const
    {
        auto *range     = find_lb(flat_perf, start);
        size_t midpoint = start + (end - start) / 2;

        std::stringstream info;
        info << "expected=[" << start << ", " << end << ", " << exp_func
             << "] actual=[" << range->start << ", " << range->end << ", "
             << range->value << "]";

        EXPECT_NE(range, nullptr) << info.str();
        EXPECT_GE(start, range->start) << info.str();
        EXPECT_LE(end, range->end) << info.str();

        // Cannot compare functions directly since on big sizes (e.g. SIZE_MAX)
        // small latency difference can be overhelmed by floating point math
        // innacurracy after addition with huge BW*SIZE_MAX.
        for (size_t point : {start, midpoint, end}) {
            double expected_result = ucs_linear_func_apply(exp_func, point);
            double range_result = ucs_linear_func_apply(range->value, point);
            EXPECT_NEAR(expected_result, range_result, 10e-9) << info.str();
        }
    }

    void expect_envelope_flat_perf(size_t start, size_t end,
                                   const perf_factors_map_t &factors) const
    {
        ucp_proto_perf_envelope_t envelope = UCS_ARRAY_DYNAMIC_INITIALIZER;
        ucp_proto_perf_envelope_elem_t *envelope_elem;
        std::vector<ucs_linear_func_t> factors_vec;

        for (const auto &factor : factors) {
            if (factor.first == UCP_PROTO_PERF_FACTOR_LATENCY) {
                // Latency shouldn't be considered during envelope building
                factors_vec.emplace_back(UCS_LINEAR_FUNC_ZERO);
            } else {
                factors_vec.emplace_back(factor.second);
            }
        }
        ASSERT_FALSE(factors_vec.empty());

        ASSERT_UCS_OK(ucp_proto_perf_envelope_make(&factors_vec.front(),
                                                   factors_vec.size(), start,
                                                   end, 0, &envelope));

        ucs_array_for_each(envelope_elem, &envelope) {
            ucs_assert(envelope_elem != NULL); /* For coverity */
            expect_flat_range(m_envelope_flat_perf, start,
                              envelope_elem->max_length,
                              factors_vec[envelope_elem->index]);
            start = envelope_elem->max_length + 1;
        }

        ucs_array_cleanup_dynamic(&envelope);
    }

    void expect_sum_flat_perf(size_t start, size_t end,
                              const perf_factors_map_t &factors) const
    {
        ucs_linear_func_t expected_func = UCS_LINEAR_FUNC_ZERO;
        for (const auto &factor : factors) {
            ucs_linear_func_add_inplace(&expected_func, factor.second);
        }

        expect_flat_range(m_sum_flat_perf, start, end, expected_func);
    }

    void expect_flat_perf(size_t start, size_t end,
                          const perf_factors_map_t &factors) const
    {
        // Reduce testing range to `max_envelope_check_size` since precise
        // testing of big sizes is impossible due to floating point loss
        if (start > max_envelope_check_size) {
            return;
        }
        end = std::min(end, max_envelope_check_size);

        expect_envelope_flat_perf(start, end, factors);
        expect_sum_flat_perf(start, end, factors);
    }

    static flat_perf_ptr_t make_flat_perf_ptr(ucp_proto_flat_perf_t *flat_perf)
    {
        return {flat_perf, ucp_proto_flat_perf_destroy};
    }

protected:
    static const size_t            max_envelope_check_size;
    static const ucs_linear_func_t local_tl_func;
    static const ucs_linear_func_t remote_tl_func;
    static const ucs_linear_func_t local_cpu_func;
    flat_perf_ptr_t                m_envelope_flat_perf;
    flat_perf_ptr_t                m_sum_flat_perf;
    perf_ptr_t                     m_perf;
};

const ucs_linear_func_t test_proto_perf::local_tl_func  = perf_func(10, 1000);
const ucs_linear_func_t test_proto_perf::remote_tl_func = perf_func(20, 2000);
const ucs_linear_func_t test_proto_perf::local_cpu_func = perf_func(30, 3000);
const size_t test_proto_perf::max_envelope_check_size   = UCS_GBYTE;

UCS_TEST_F(test_proto_perf, empty)
{
    m_perf = create();

    make_flat_perf();
    print_perf();

    expect_empty_range(0, SIZE_MAX);

    ASSERT_TRUE(ucp_proto_perf_is_empty(m_perf.get()));
}

UCS_TEST_F(test_proto_perf, single_func)
{
    m_perf = create();
    add_func(1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 999);
    expect_perf(1000, 1999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(2000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, staged_pipeline_counts_recurring_fragments)
{
    const size_t frag_size = 1024;
    ucp_proto_perf_stage_t stages[1] = {};
    ucs_linear_func_t expected_two_frags;
    ucs_linear_func_t expected_three_frags;
    ucs_linear_func_t expected_four_frags;

    m_perf = create();

    stages[0].name    = "copy";
    stages[0].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_PARALLEL;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] = local_tl_func;
    stages[0].frag_size = frag_size;

    ASSERT_UCS_OK(ucp_proto_perf_add_staged_pipeline(
            m_perf.get(), frag_size + 1, 4 * frag_size, stages,
            ucs_static_array_size(stages), frag_size, NULL));

    make_flat_perf();
    print_perf();

    expected_two_frags   = ucs_linear_func_make(2 * local_tl_func.c,
                                                local_tl_func.m);
    expected_three_frags = ucs_linear_func_make(3 * local_tl_func.c,
                                                local_tl_func.m);
    expected_four_frags  = ucs_linear_func_make(4 * local_tl_func.c,
                                                local_tl_func.m);

    expect_perf(frag_size + 1, 2 * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, expected_two_frags}});
    expect_perf((2 * frag_size) + 1, 3 * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, expected_three_frags}});
    expect_perf((3 * frag_size) + 1, 4 * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, expected_four_frags}});
}

UCS_TEST_F(test_proto_perf, staged_pipeline_bounds_unlimited_tail)
{
    const size_t frag_size = 1024;
    const size_t exact_frags =
            UCP_PROTO_PERF_STAGED_PIPELINE_MAX_EXACT_FRAGS;
    const size_t tail_start = (exact_frags * frag_size) + 1;
    ucp_proto_perf_stage_t stages[1] = {};
    ucs_linear_func_t expected_exact;
    ucs_linear_func_t expected_tail;
    double fixed_slope;

    m_perf = create();

    stages[0].name    = "copy";
    stages[0].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_PARALLEL;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] = local_tl_func;
    stages[0].frag_size = frag_size;

    ASSERT_UCS_OK(ucp_proto_perf_add_staged_pipeline(
            m_perf.get(), frag_size + 1, SIZE_MAX, stages,
            ucs_static_array_size(stages), frag_size, NULL));

    make_flat_perf();
    print_perf();

    expected_exact = ucs_linear_func_make(exact_frags * local_tl_func.c,
                                          local_tl_func.m);
    expect_perf(((exact_frags - 1) * frag_size) + 1,
                exact_frags * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, expected_exact}});

    expected_tail = ucs_linear_func_make((exact_frags + 1) * local_tl_func.c,
                                         local_tl_func.m);
    fixed_slope   = local_tl_func.c / frag_size;
    expected_tail.m += fixed_slope;
    expected_tail.c -= fixed_slope * tail_start;

    expect_perf(tail_start, SIZE_MAX,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, expected_tail}});

    EXPECT_TRUE(ucs_linear_func_is_equal(
            expected_tail,
            ucp_proto_perf_segment_func(find_lb(m_perf, tail_start),
                                        UCP_PROTO_PERF_FACTOR_LOCAL_TL),
            1e-9));
}

UCS_TEST_F(test_proto_perf, staged_pipeline_parallel_recurring_stages)
{
    const size_t frag_size = 1024;
    ucp_proto_perf_stage_t stages[2] = {};
    ucs_linear_func_t expected_local_tl;
    ucs_linear_func_t expected_mtype_copy;

    m_perf = create();

    stages[0].name    = "host transport";
    stages[0].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_PARALLEL;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] = local_tl_func;
    stages[0].frag_size = frag_size;

    stages[1].name    = "memory copy";
    stages[1].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[1].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_PARALLEL;
    stages[1].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] =
            remote_tl_func;
    stages[1].frag_size = frag_size;

    ASSERT_UCS_OK(ucp_proto_perf_add_staged_pipeline(
            m_perf.get(), frag_size + 1, 2 * frag_size, stages,
            ucs_static_array_size(stages), frag_size, NULL));

    make_flat_perf();
    print_perf();

    expected_local_tl = ucs_linear_func_make(2 * local_tl_func.c,
                                             local_tl_func.m);
    expected_mtype_copy = ucs_linear_func_make(2 * remote_tl_func.c,
                                               remote_tl_func.m);

    expect_perf(frag_size + 1, 2 * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, expected_local_tl},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY,
                  expected_mtype_copy}});
}

UCS_TEST_F(test_proto_perf, staged_proto_flat_perf_envelopes_stage_costs_only)
{
    const size_t frag_size = 1024;
    const ucs_linear_func_t copy_func = perf_func(10, 50000);
    const ucs_linear_func_t control_func = perf_func(30, 100000);
    perf_ptr_t staged_perf = create();
    perf_ptr_t control_perf = create();
    ucp_proto_select_init_protocols_t proto_init;
    ucp_proto_select_param_t select_param = {};
    ucp_proto_init_params_t init_params = {};
    ucp_ep_config_key_t ep_config_key = {};
    ucp_proto_perf_stage_t stages[2] = {};
    ucp_proto_init_elem_t *init_elem;
    const ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_t *perf;
    ucs_linear_func_t expected;

    stages[0].name        = "local copy";
    stages[0].role        = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap     = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] = copy_func;
    stages[0].frag_size   = frag_size;
    stages[0].resource_id = 1;

    stages[1].name        = "remote copy";
    stages[1].role        = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[1].overlap     = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    stages[1].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY] = copy_func;
    stages[1].frag_size   = frag_size;
    stages[1].resource_id = 2;

    ASSERT_UCS_OK(ucp_proto_perf_add_staged_pipeline(
            staged_perf.get(), frag_size + 1, 2 * frag_size, stages,
            ucs_static_array_size(stages), frag_size, NULL));

    add_func(control_perf, frag_size + 1, 2 * frag_size,
             UCP_PROTO_PERF_FACTOR_LOCAL_TL, control_func);
    ASSERT_UCS_OK(ucp_proto_perf_aggregate2("staged+control",
                                            staged_perf.get(),
                                            control_perf.get(), &perf));

    ucs_array_init_dynamic(&proto_init.protocols);
    ucs_array_init_dynamic(&proto_init.priv_buf);

    init_params.select_param  = &select_param;
    init_params.ep_config_key = &ep_config_key;
    init_params.proto_id      = 0;
    init_params.ctx           = &proto_init;

    ucp_proto_select_add_proto_staged(
            &init_params, 0, 0, perf, NULL, 0, stages,
            ucs_static_array_size(stages));

    ASSERT_EQ(1u, ucs_array_length(&proto_init.protocols));
    init_elem = &ucs_array_elem(&proto_init.protocols, 0);
    range     = ucp_proto_flat_perf_find_lb(init_elem->flat_perf,
                                            frag_size + 1);
    ASSERT_NE(nullptr, range);

    expected = ucs_linear_func_make(2 * copy_func.c, copy_func.m);
    ucs_linear_func_add_inplace(&expected, control_func);
    EXPECT_NEAR(ucs_linear_func_apply(expected, 2 * frag_size),
                ucs_linear_func_apply(range->value, 2 * frag_size), 1e-9);

    ucp_proto_flat_perf_destroy(init_elem->flat_perf);
    ucp_proto_perf_destroy(init_elem->perf);
    ucs_array_cleanup_dynamic(&proto_init.priv_buf);
    ucs_array_cleanup_dynamic(&proto_init.protocols);
}

UCS_TEST_F(test_proto_perf, staged_pipeline_resource_serial_stages_are_summed)
{
    const size_t frag_size = 1024;
    const ucs_linear_func_t copy_func = perf_func(40, 4000);
    ucp_proto_perf_stage_t stages[2] = {};
    ucs_linear_func_t expected;

    m_perf = create();

    stages[0].name    = "copy 0";
    stages[0].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] =
            local_tl_func;
    stages[0].frag_size   = frag_size;
    stages[0].resource_id = 7;

    stages[1].name    = "copy 1";
    stages[1].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[1].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    stages[1].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] = copy_func;
    stages[1].frag_size   = frag_size;
    stages[1].resource_id = 7;

    ASSERT_UCS_OK(ucp_proto_perf_add_staged_pipeline(
            m_perf.get(), frag_size + 1, 2 * frag_size, stages,
            ucs_static_array_size(stages), frag_size, NULL));

    make_flat_perf();
    print_perf();

    expected = ucs_linear_func_make(2 * (local_tl_func.c + copy_func.c),
                                    local_tl_func.m + copy_func.m);
    expect_perf(frag_size + 1, 2 * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY, expected}});
}

UCS_TEST_F(test_proto_perf, segment_make_stages_groups_side_resources)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_perf_stage_t stages[UCP_PROTO_PERF_FACTOR_LAST] = {};
    unsigned num_stages;

    add_funcs(frag_perf, 0, frag_size,
              {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
               {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func},
               {UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY, remote_tl_func}});

    seg = find_lb(frag_perf, 0);
    ASSERT_NE(nullptr, seg);

    ASSERT_UCS_OK(ucp_proto_perf_segment_make_stages(
            seg, frag_size, stages, ucs_static_array_size(stages),
            &num_stages));

    ASSERT_EQ(3u, num_stages);
    EXPECT_EQ(UCP_PROTO_PERF_STAGE_ROLE_RECURRING, stages[0].role);
    EXPECT_EQ(UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL,
              stages[0].overlap);
    EXPECT_EQ(frag_size, stages[0].frag_size);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            local_cpu_func,
            stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU], 1e-9));

    EXPECT_EQ(UCP_PROTO_PERF_STAGE_ROLE_RECURRING, stages[1].role);
    EXPECT_EQ(UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL,
              stages[1].overlap);
    EXPECT_EQ(stages[0].resource_id, stages[1].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            local_tl_func, stages[1].factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL],
            1e-9));

    EXPECT_EQ(UCP_PROTO_PERF_STAGE_ROLE_RECURRING, stages[2].role);
    EXPECT_EQ(UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL,
              stages[2].overlap);
    EXPECT_NE(stages[0].resource_id, stages[2].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            remote_tl_func,
            stages[2].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY],
            1e-9));
}

UCS_TEST_F(test_proto_perf, stage_apply_func_updates_declared_factors)
{
    ucp_proto_perf_stage_t stages[1] = {};
    ucs_linear_func_t func           = ucs_linear_func_make(7.0, 0.9);
    ucs_linear_func_t expected;

    stages[0].name      = "copy";
    stages[0].role      = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap   = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    stages[0].frag_size = 1024;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] =
            local_tl_func;

    ucp_proto_perf_stages_apply_func(stages, ucs_static_array_size(stages),
                                     func);

    expected = ucs_linear_func_compose(func, local_tl_func);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            expected, stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
            1e-9));
    EXPECT_TRUE(ucs_linear_func_is_zero(
            stages[0].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY], 1e-9));
}

UCS_TEST_F(test_proto_perf, rndv_perf_make_stages_requires_finite_fragment)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    ucp_proto_perf_stage_t stages[UCP_PROTO_PERF_FACTOR_LAST] = {};
    bool found_local_copy = false;
    bool found_remote_tl  = false;
    unsigned num_stages;

    add_funcs(frag_perf, 0, (frag_size / 2) - 1,
              {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    add_funcs(frag_perf, frag_size / 2, frag_size,
              {{UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY, local_tl_func},
               {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});

    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_stages(
                          frag_perf.get(), 0, stages,
                          ucs_static_array_size(stages)));
    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_stages(
                          frag_perf.get(), SIZE_MAX, stages,
                          ucs_static_array_size(stages)));
    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_stages(
                          frag_perf.get(), frag_size - 1, stages,
                          ucs_static_array_size(stages)));

    num_stages = ucp_proto_rndv_perf_make_stages(
            frag_perf.get(), frag_size, stages, ucs_static_array_size(stages));

    ASSERT_EQ(2u, num_stages);
    for (unsigned i = 0; i < num_stages; ++i) {
        EXPECT_EQ(UCP_PROTO_PERF_STAGE_ROLE_RECURRING, stages[i].role);
        EXPECT_EQ(UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL,
                  stages[i].overlap);
        EXPECT_EQ(frag_size, stages[i].frag_size);

        found_local_copy |= ucs_linear_func_is_equal(
                local_tl_func,
                stages[i].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
                1e-9);
        found_remote_tl |= ucs_linear_func_is_equal(
                remote_tl_func,
                stages[i].factors[UCP_PROTO_PERF_FACTOR_REMOTE_TL],
                1e-9);
    }

    EXPECT_TRUE(found_local_copy);
    EXPECT_TRUE(found_remote_tl);
}

UCS_TEST_F(test_proto_perf, rndv_mtype_copy_stages_exclude_tl_factors)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    ucp_proto_perf_stage_t stages[UCP_PROTO_PERF_FACTOR_LAST] = {};
    bool found_local_copy  = false;
    bool found_remote_copy = false;
    bool found_local_tl    = false;
    bool found_remote_tl   = false;
    unsigned num_stages;

    add_funcs(frag_perf, 0, frag_size,
              {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func},
               {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func},
               {UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY, local_cpu_func},
               {UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY, local_tl_func}});

    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_mtype_copy_stages(
                          frag_perf.get(), 0, stages,
                          ucs_static_array_size(stages)));
    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_mtype_copy_stages(
                          frag_perf.get(), SIZE_MAX, stages,
                          ucs_static_array_size(stages)));
    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_mtype_copy_stages(
                          frag_perf.get(), frag_size - 1, stages,
                          ucs_static_array_size(stages)));

    num_stages = ucp_proto_rndv_perf_make_mtype_copy_stages(
            frag_perf.get(), frag_size, stages, ucs_static_array_size(stages));

    ASSERT_EQ(2u, num_stages);
    for (unsigned i = 0; i < num_stages; ++i) {
        EXPECT_EQ(UCP_PROTO_PERF_STAGE_ROLE_RECURRING, stages[i].role);
        EXPECT_EQ(UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL,
                  stages[i].overlap);
        EXPECT_EQ(frag_size, stages[i].frag_size);

        found_local_copy |= ucs_linear_func_is_equal(
                local_cpu_func,
                stages[i].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
                1e-9);
        found_remote_copy |= ucs_linear_func_is_equal(
                local_tl_func,
                stages[i].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY],
                1e-9);
        found_local_tl |= !ucs_linear_func_is_zero(
                stages[i].factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL], 1e-9);
        found_remote_tl |= !ucs_linear_func_is_zero(
                stages[i].factors[UCP_PROTO_PERF_FACTOR_REMOTE_TL], 1e-9);
    }

    EXPECT_TRUE(found_local_copy);
    EXPECT_TRUE(found_remote_copy);
    EXPECT_FALSE(found_local_tl);
    EXPECT_FALSE(found_remote_tl);
}

UCS_TEST_F(test_proto_perf, rndv_mtype_copy_stages_require_mtype_factor)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    ucp_proto_perf_stage_t stages[UCP_PROTO_PERF_FACTOR_LAST] = {};

    add_funcs(frag_perf, 0, frag_size,
              {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func},
               {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});

    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_mtype_copy_stages(
                          frag_perf.get(), frag_size, stages,
                          ucs_static_array_size(stages)));

    add_func(frag_perf, 1, frag_size,
             UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY, local_cpu_func);

    EXPECT_EQ(1u, ucp_proto_rndv_perf_make_mtype_copy_stages(
                          frag_perf.get(), frag_size, stages,
                          ucs_static_array_size(stages)));
    EXPECT_TRUE(ucs_linear_func_is_equal(
            local_cpu_func,
            stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY], 1e-9));
}

UCS_TEST_F(test_proto_perf, rndv_mtype_attached_clears_access_tl_payload)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    perf_ptr_t staged_perf = create();
    perf_ptr_t access_perf = create();
    ucp_proto_perf_stage_t stages[UCP_PROTO_PERF_FACTOR_LAST] = {};
    ucp_proto_flat_perf_t *flat_perf;
    flat_perf_ptr_t staged_flat_perf;
    const ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_t *perf;
    ucs_linear_func_t expected;
    unsigned num_stages;

    add_funcs(frag_perf, 0, frag_size,
              {{UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func},
               {UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY, local_tl_func}});

    num_stages = ucp_proto_rndv_perf_make_mtype_copy_stages(
            frag_perf.get(), frag_size, stages,
            ucs_static_array_size(stages));

    ASSERT_EQ(1u, num_stages);

    ASSERT_UCS_OK(ucp_proto_perf_add_staged_pipeline(
            staged_perf.get(), frag_size + 1, 2 * frag_size, stages,
            num_stages, frag_size, NULL));
    add_funcs(access_perf, frag_size + 1, 2 * frag_size,
              {{UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func},
               {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_cpu_func},
               {UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    ASSERT_UCS_OK(ucp_proto_perf_aggregate2("staged+access",
                                            staged_perf.get(),
                                            access_perf.get(), &perf));
    frag_perf = make_perf_ptr(perf);

    ucp_proto_perf_clear_factor_slopes(
            frag_perf.get(), UCS_BIT(UCP_PROTO_PERF_FACTOR_LOCAL_TL) |
                             UCS_BIT(UCP_PROTO_PERF_FACTOR_REMOTE_TL));

    ASSERT_UCS_OK(ucp_proto_perf_staged_pipeline_flat(
            frag_perf.get(), stages, num_stages, &flat_perf));
    staged_flat_perf = flat_perf_ptr_t(flat_perf,
                                       ucp_proto_flat_perf_destroy);

    range = find_lb(staged_flat_perf, frag_size + 1);
    ASSERT_NE(nullptr, range);

    expected = local_cpu_func;
    ucs_linear_func_add_inplace(&expected,
                                ucs_linear_func_make(local_cpu_func.c, 0));
    ucs_linear_func_add_inplace(&expected,
                                ucs_linear_func_make(remote_tl_func.c, 0));
    ucs_linear_func_add_inplace(
            &expected, ucs_linear_func_make(2 * local_tl_func.c,
                                            local_tl_func.m));

    EXPECT_TRUE(ucs_linear_func_is_equal(
            ucs_linear_func_make(local_cpu_func.c, 0),
            ucp_proto_perf_segment_func(find_lb(frag_perf, frag_size + 1),
                                        UCP_PROTO_PERF_FACTOR_LOCAL_TL),
            1e-9));
    EXPECT_TRUE(ucs_linear_func_is_equal(
            ucs_linear_func_make(remote_tl_func.c, 0),
            ucp_proto_perf_segment_func(find_lb(frag_perf, frag_size + 1),
                                        UCP_PROTO_PERF_FACTOR_REMOTE_TL),
            1e-9));
    EXPECT_NEAR(ucs_linear_func_apply(expected, 2 * frag_size),
                ucs_linear_func_apply(range->value, 2 * frag_size), 1e-9);
}

UCS_TEST_F(test_proto_perf, rndv_mtype_copy_remote_stages_flip_side_resources)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    ucp_proto_perf_stage_t local_stages[2]  = {};
    ucp_proto_perf_stage_t remote_stages[2] = {};
    uint64_t local_resource_id              = 0;
    uint64_t remote_resource_id             = 0;
    unsigned num_stages;

    add_funcs(frag_perf, 0, frag_size,
              {{UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY, local_cpu_func},
               {UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY, local_tl_func}});

    num_stages = ucp_proto_rndv_perf_make_mtype_copy_stages(
            frag_perf.get(), frag_size, local_stages,
            ucs_static_array_size(local_stages));

    ASSERT_EQ(2u, num_stages);
    for (unsigned i = 0; i < num_stages; ++i) {
        if (!ucs_linear_func_is_zero(
                    local_stages[i].factors
                            [UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
                    1e-9)) {
            local_resource_id = local_stages[i].resource_id;
        }

        if (!ucs_linear_func_is_zero(
                    local_stages[i].factors
                            [UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY],
                    1e-9)) {
            remote_resource_id = local_stages[i].resource_id;
        }
    }

    ASSERT_NE(0u, local_resource_id);
    ASSERT_NE(0u, remote_resource_id);
    EXPECT_NE(local_resource_id, remote_resource_id);

    num_stages = ucp_proto_rndv_perf_make_remote_stages(
            local_stages, ucs_static_array_size(local_stages), remote_stages,
            ucs_static_array_size(remote_stages));

    ASSERT_EQ(ucs_static_array_size(local_stages), num_stages);
    EXPECT_EQ(remote_resource_id, remote_stages[0].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            local_cpu_func,
            remote_stages[0].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY],
            1e-9));
    EXPECT_EQ(local_resource_id, remote_stages[1].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            local_tl_func,
            remote_stages[1].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
            1e-9));
}

UCS_TEST_F(test_proto_perf, rndv_remote_stages_preserve_declared_plan)
{
    const size_t frag_size = 1024;
    ucp_proto_perf_stage_t declared[2] = {};
    ucp_proto_perf_stage_t stages[2]   = {};
    unsigned num_stages;

    declared[0].name        = "local-copy";
    declared[0].role        = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    declared[0].overlap     = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    declared[0].frag_size   = frag_size;
    declared[0].resource_id = 1;
    declared[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] =
            local_tl_func;

    declared[1].name        = "remote-copy";
    declared[1].role        = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    declared[1].overlap     = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    declared[1].frag_size   = frag_size;
    declared[1].resource_id = 2;
    declared[1].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY] =
            remote_tl_func;

    EXPECT_EQ(0u, ucp_proto_rndv_perf_make_remote_stages(
                          declared, ucs_static_array_size(declared), stages,
                          1));

    num_stages = ucp_proto_rndv_perf_make_remote_stages(
            declared, ucs_static_array_size(declared), stages,
            ucs_static_array_size(stages));

    ASSERT_EQ(ucs_static_array_size(declared), num_stages);
    EXPECT_EQ(declared[0].name, stages[0].name);
    EXPECT_EQ(declared[0].role, stages[0].role);
    EXPECT_EQ(declared[0].overlap, stages[0].overlap);
    EXPECT_EQ(declared[0].frag_size, stages[0].frag_size);
    EXPECT_EQ(declared[0].resource_id, stages[0].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            local_tl_func,
            stages[0].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY],
            1e-9));
    EXPECT_TRUE(ucs_linear_func_is_zero(
            stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
            1e-9));

    EXPECT_EQ(declared[1].name, stages[1].name);
    EXPECT_EQ(declared[1].role, stages[1].role);
    EXPECT_EQ(declared[1].overlap, stages[1].overlap);
    EXPECT_EQ(declared[1].frag_size, stages[1].frag_size);
    EXPECT_EQ(declared[1].resource_id, stages[1].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            remote_tl_func,
            stages[1].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY],
            1e-9));
    EXPECT_TRUE(ucs_linear_func_is_zero(
            stages[1].factors[UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY],
            1e-9));
}

UCS_TEST_F(test_proto_perf, rndv_ctrl_add_perf_stages_preserves_on_overflow)
{
    const size_t frag_size = 1024;
    perf_ptr_t unpack_perf = create();
    ucp_proto_perf_stage_t stages
            [UCP_PROTO_INIT_ELEM_MAX_STAGED_PIPELINE_STAGES] = {};
    unsigned num_stages = ucs_static_array_size(stages);

    stages[0].name        = "existing";
    stages[0].role        = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap     = UCP_PROTO_PERF_STAGE_OVERLAP_RESOURCE_SERIAL;
    stages[0].frag_size   = frag_size;
    stages[0].resource_id = 7;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_REMOTE_TL] = remote_tl_func;

    add_func(unpack_perf, 0, frag_size, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);

    EXPECT_EQ(UCS_ERR_EXCEEDS_LIMIT,
              ucp_proto_rndv_ctrl_add_perf_stages(unpack_perf.get(), stages,
                                                  &num_stages));
    EXPECT_EQ(ucs_static_array_size(stages), num_stages);
    EXPECT_EQ(std::string("existing"), stages[0].name);
    EXPECT_EQ(frag_size, stages[0].frag_size);
    EXPECT_EQ(7u, stages[0].resource_id);
    EXPECT_TRUE(ucs_linear_func_is_equal(
            remote_tl_func, stages[0].factors[UCP_PROTO_PERF_FACTOR_REMOTE_TL],
            1e-9));
}

UCS_TEST_F(test_proto_perf, staged_ppln_uses_declared_stages)
{
    const size_t frag_size = 1024;
    perf_ptr_t frag_perf  = create();
    ucp_proto_perf_stage_t stages[1] = {};
    ucs_linear_func_t expected_two_frags;

    add_func(frag_perf, 0, frag_size, UCP_PROTO_PERF_FACTOR_LOCAL_TL,
             local_tl_func);
    m_perf = create();

    stages[0].name    = "copy";
    stages[0].role    = UCP_PROTO_PERF_STAGE_ROLE_RECURRING;
    stages[0].overlap = UCP_PROTO_PERF_STAGE_OVERLAP_PARALLEL;
    stages[0].factors[UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY] =
            remote_tl_func;
    stages[0].frag_size = frag_size;

    ASSERT_NE(nullptr, ucp_proto_perf_add_ppln_staged(
                               frag_perf.get(), m_perf.get(), 2 * frag_size,
                               stages, ucs_static_array_size(stages)));

    make_flat_perf();
    print_perf();

    expected_two_frags = ucs_linear_func_make(2 * remote_tl_func.c,
                                              remote_tl_func.m);
    expect_perf(frag_size + 1, 2 * frag_size,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY,
                  expected_two_frags}});
}

UCS_TEST_F(test_proto_perf, to_inf) {
    /*
     *  0    172                   SIZE_MAX
     *  |     |                       |
     *  |     +------ local_tl -------+
     *  |                             |
     *  |                             |
     *  +----------- local_cpu -------+
     */
    m_perf = create();
    add_func(172, SIZE_MAX, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(0, SIZE_MAX, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_perf(0, 171, {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(172, SIZE_MAX,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
}

UCS_TEST_F(test_proto_perf, intersect_first)
{
    /*
     * 500   1000          1999    3000          3999  4999
     *  |     |              |      |              |    |
     *  +-----+-- local_tl --+      +- remote_tl --+    |
     *        |                                         |
     *        |                                         |
     *        +------ local_cpu ------------------------+
     */
    m_perf = create();
    add_func(500, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(3000, 3999, UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func);
    add_func(1000, 4999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 499);
    expect_perf(500, 999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(2000, 2999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(3000, 3999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});
    expect_perf(4000, 4999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_empty_range(5000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, intersect_last)
{
    /*
     * 500   1000         1999    3000  3499    3999
     *  |     |             |      |      |       |
     *  |     +- local_tl +-+      +- remote_tl --+
     *  |                                  |
     *  |                                  |
     *  +----- local_cpu ------------------+
     */
    m_perf = create();
    add_func(1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(3000, 3999, UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func);
    add_func(500, 3499, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 499);
    expect_perf(500, 999, {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(2000, 2999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func}});
    expect_perf(3000, 3499,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});
    expect_perf(3500, 3999,
                {{UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_func}});
    expect_empty_range(4000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, intersect_middle)
{
    /*
     * 500   1000                  1999   2999
     *  |     |                      |     |
     *  +-----+------ local_tl ------+-----+
     *        |                      |
     *        |                      |
     *        +------ local_cpu -----+
     */
    m_perf = create();
    add_func(500, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    add_func(1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func);

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 499);
    expect_perf(500, 999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_perf(2000, 2999, {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(3000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, agg2)
{
    /*
     * 500   1000                  1999   2999
     *  |     |                      |     |
     *  +-----+------ local_tl ------+-----+
     *        |                      |
     *        |                      |
     *        +------ local_cpu -----+
     */
    perf_ptr_t perf1 = create();
    add_func(perf1, 500, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    UCS_TEST_MESSAGE << perf1.get();

    perf_ptr_t perf2 = create();
    add_func(perf2, 1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);
    UCS_TEST_MESSAGE << perf2.get();

    m_perf = aggregate({perf1, perf2});

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 999);
    expect_perf(1000, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU, local_cpu_func},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(2000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, agg3)
{
    /*
     * 500   1000  1200              1999       2999
     *  |     |      |                 |          |
     *  +-----+------+---1.local_tl ---+----------+
     *        |      |                 |          |
     *        |      |                 |          |
     *        +------+--2.local_cpu1 --+          |
     *               |                            |
     *               |                            |
     *               +--3.local_cpu2 -------------+
     *
     */
    perf_ptr_t perf1 = create();
    add_func(perf1, 500, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func);
    UCS_TEST_MESSAGE << perf1.get();

    perf_ptr_t perf2 = create();
    add_func(perf2, 1000, 1999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);
    UCS_TEST_MESSAGE << perf2.get();

    perf_ptr_t perf3 = create();
    add_func(perf3, 1200, 2999, UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
             local_cpu_func);
    UCS_TEST_MESSAGE << perf3.get();

    m_perf = aggregate({perf1, perf2, perf3});

    make_flat_perf();
    print_perf();

    expect_empty_range(0, 1199);
    expect_perf(1200, 1999,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
                  ucs_linear_func_add(local_cpu_func, local_cpu_func)},
                 {UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_func}});
    expect_empty_range(2000, SIZE_MAX);
}

UCS_TEST_F(test_proto_perf, envelope_intersect) {
    size_t overhead_per_byte = 10;
    size_t fixed_overhead    = 1000;
    auto local_tl_factor     = ucs_linear_func_make(0, overhead_per_byte);
    auto remote_tl_factor    = ucs_linear_func_make(fixed_overhead, 0);

    /*
     * Case for testing intersection of two factors:
     *           //
     * envelope //
     *         //
     * _______//
     * -------+-----------------------------
     *       /                   REMOTE_TL
     *      /
     *     / LOCAL_TL
     */
    m_perf = create();
    add_func(0, SIZE_MAX, UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_factor);
    add_func(0, SIZE_MAX, UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_factor);

    make_flat_perf();
    print_perf();

    expect_perf(0, SIZE_MAX,
                {{UCP_PROTO_PERF_FACTOR_LOCAL_TL, local_tl_factor},
                 {UCP_PROTO_PERF_FACTOR_REMOTE_TL, remote_tl_factor}});

    // `fixed_overhead` should be divisible by `overhead_per_byte` to have
    // accurate intersection point
    ASSERT_GE(fixed_overhead, overhead_per_byte);
    ASSERT_EQ(fixed_overhead % overhead_per_byte, 0);
    auto intersection     = fixed_overhead / overhead_per_byte;
    auto *remote_tl_range = find_lb(m_envelope_flat_perf, intersection);
    auto *local_tl_range  = find_lb(m_envelope_flat_perf, intersection + 1);

    ASSERT_NE(remote_tl_range, nullptr);
    ASSERT_NE(local_tl_range, nullptr);

    ASSERT_TRUE(ucs_linear_func_is_equal(remote_tl_factor,
                                         remote_tl_range->value, 1e-9));
    ASSERT_TRUE(ucs_linear_func_is_equal(local_tl_factor,
                                         local_tl_range->value, 1e-9));
}

class test_proto_perf_random : public test_proto_perf {
protected:
    // Generate a random perf structure
    static perf_ptr_t generate_random_perf(unsigned max_segments);

private:
    // Generate a random ascending sequence of numbers, ending with SIZE_MAX
    static std::vector<size_t> generate_random_sequence(unsigned length);
};

std::vector<size_t>
test_proto_perf_random::generate_random_sequence(unsigned length)
{
    std::vector<size_t> sequence;
    size_t value = 0;
    std::generate_n(std::back_inserter(sequence), length, [&value]() {
        value += ucs::rand_range(10000) + 1;
        return value;
    });
    sequence.emplace_back(SIZE_MAX);
    return sequence;
}

test_proto_perf::perf_ptr_t
test_proto_perf_random::generate_random_perf(unsigned max_segments)
{
    perf_ptr_t perf = create();
    size_t start    = 0;
    auto points     = generate_random_sequence(max_segments);
    for (auto point : points) {
        if (ucs::rand_range(4) > 0) {
            size_t num_factors = ucs::rand_range(UCP_PROTO_PERF_FACTOR_LAST);
            perf_factors_map_t factors_map;
            for (int i = 0; i < num_factors; ++i) {
                auto factor_id   = ucs::rand_range(UCP_PROTO_PERF_FACTOR_LAST);
                /* Avoid setting 0 BW since it can cause INF estimated time */
                auto factor_func = perf_func(1.0 * ucs::rand_range(4),
                                             1000.0 * (ucs::rand_range(4) + 1));
                factors_map.emplace(factor_id, factor_func);
            }
            add_funcs(perf, start, point, factors_map);
        }
        start = point + 1;
    }
    return perf;
}

class test_proto_perf_remote : public test_proto_perf_random {
protected:
    static perf_ptr_t create_remote(perf_ptr_t perf)
    {
        ucp_proto_perf_t *remote_perf;
        ASSERT_UCS_OK(ucp_proto_perf_remote(perf.get(), &remote_perf));
        return test_proto_perf::make_perf_ptr(remote_perf);
    }
};

UCS_TEST_F(test_proto_perf_remote, compare_with_local) {
    using factor_id_t = ucp_proto_perf_factor_id_t;
    std::vector<std::pair<factor_id_t, factor_id_t>> convert_map = {
        {UCP_PROTO_PERF_FACTOR_LOCAL_CPU, UCP_PROTO_PERF_FACTOR_REMOTE_CPU},
        {UCP_PROTO_PERF_FACTOR_LOCAL_TL, UCP_PROTO_PERF_FACTOR_REMOTE_TL},
        {UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY,
         UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY},
        {UCP_PROTO_PERF_FACTOR_LATENCY, UCP_PROTO_PERF_FACTOR_LATENCY},
        {UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY,
         UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY},
        {UCP_PROTO_PERF_FACTOR_REMOTE_TL, UCP_PROTO_PERF_FACTOR_LOCAL_TL},
        {UCP_PROTO_PERF_FACTOR_REMOTE_CPU, UCP_PROTO_PERF_FACTOR_LOCAL_CPU},
    };
    m_perf                 = generate_random_perf(100);
    perf_ptr_t remote_perf = create_remote(m_perf);

    std::vector<const ucp_proto_perf_segment_t*> segments, remote_segments;
    const ucp_proto_perf_segment_t *seg, *rseg;
    size_t seg_start, seg_end;
    ucp_proto_perf_segment_foreach_range(seg, seg_start, seg_end, m_perf.get(),
                                         0, SIZE_MAX) {
        segments.emplace_back(seg);
    }
    ucp_proto_perf_segment_foreach_range(seg, seg_start, seg_end,
                                         remote_perf.get(), 0, SIZE_MAX) {
        remote_segments.emplace_back(seg);
    }

    ASSERT_EQ(segments.size(), remote_segments.size());
    for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
        rseg = remote_segments[seg_idx];
        seg  = segments[seg_idx];
        for (const auto &convert_pair : convert_map) {
            auto func  = ucp_proto_perf_segment_func(seg, convert_pair.first);
            auto rfunc = ucp_proto_perf_segment_func(rseg, convert_pair.second);
            ASSERT_TRUE(ucs_linear_func_is_equal(func, rfunc, 1e-9));
        }
    }
}

class test_proto_perf_aggregate : public test_proto_perf_random {
protected:
    // Test aggregation of random collection of perf structures, by sampling
    // the result at specific points
    void test_random_funcs(unsigned num_perfs, unsigned num_segments);

private:
    // Collect points of sampling based on perf structure segments
    static std::vector<size_t> get_sample_points(perf_ptr_t perf);

    // Calculate the expected aggregation result at a given point. If the
    // expected result is undefined, return nullptr.
    std::unique_ptr<perf_factors_map_t>
    expected_aggregate_result(const std::vector<perf_ptr_t> &perfs,
                              size_t point);
};

std::vector<size_t>
test_proto_perf_aggregate::get_sample_points(perf_ptr_t perf)
{
    std::vector<size_t> points;

    auto seg = find_lb(perf, 0);
    while (seg != NULL) {
        auto start = ucp_proto_perf_segment_start(seg);
        auto end   = ucp_proto_perf_segment_end(seg);
        points.push_back(start);
        points.push_back(end);
        points.push_back((start + end) / 2);
        if (start > 0) {
            points.push_back(start - 1);
        }
        if (end == SIZE_MAX) {
            break;
        }

        points.push_back(end + 1);
        seg = find_lb(perf, end + 1);
    }

    return points;
}

std::unique_ptr<test_proto_perf::perf_factors_map_t>
test_proto_perf_aggregate::expected_aggregate_result(
        const std::vector<perf_ptr_t> &perfs, size_t point)
{
    perf_factors_map_t agg;
    for (auto perf : perfs) {
        ucp_proto_perf_segment_t *seg = find_lb(perf, point);
        if ((seg == NULL) || (point < ucp_proto_perf_segment_start(seg))) {
            return nullptr; // Point not found in one of the perf objects
        }

        for (int factor_id = 0; factor_id < UCP_PROTO_PERF_FACTOR_LAST;
             ++factor_id) {
            auto func = ucp_proto_perf_segment_func(
                    seg, (ucp_proto_perf_factor_id_t)factor_id);
            if (!ucs_linear_func_is_zero(func, UCP_PROTO_PERF_EPSILON)) {
                if (agg.find(factor_id) == agg.end()) {
                    agg[factor_id] = func;
                } else {
                    ucs_linear_func_add_inplace(&agg[factor_id], func);
                }
            }
        }
    }

    return std::unique_ptr<perf_factors_map_t>(new perf_factors_map_t(agg));
}

void test_proto_perf_aggregate::test_random_funcs(unsigned num_perfs,
                                                  unsigned num_segments)
{
    std::vector<perf_ptr_t> perfs;
    std::generate_n(std::back_inserter(perfs), num_perfs,
                    [num_segments]() -> perf_ptr_t {
                        return generate_random_perf(num_segments);
                    });

    std::set<size_t> sample_points;
    for (auto perf : perfs) {
        auto perf_points = get_sample_points(perf);
        std::copy(perf_points.begin(), perf_points.end(),
                  std::inserter(sample_points, sample_points.begin()));
    }

    m_perf = aggregate(perfs);

    make_flat_perf();

    for (auto point : sample_points) {
        auto expected_result = expected_aggregate_result(perfs, point);
        if (expected_result) {
            expect_perf(point, point, *expected_result.get());
        } else {
            expect_empty_range(point, point);
        }
    }

    if (testing::Test::HasFailure()) {
        for (size_t i = 0; i < perfs.size(); ++i) {
            UCS_TEST_MESSAGE << "perf " << i << ": " << perfs[i].get();
        }
        UCS_TEST_MESSAGE << "result:";
        print_perf();
    }

    m_perf.reset();
}

UCS_TEST_F(test_proto_perf_aggregate, random)
{
    for (int iter = 0; (iter < (1000 / ucs::test_time_multiplier())) &&
                       !::testing::Test::HasFailure();
         ++iter) {
        test_random_funcs(10, 10);
    }
}
