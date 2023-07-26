/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_rkey.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/dt/dt.h>
}

class test_ucp_mmap : public ucp_test {
public:
    test_ucp_mmap()
    {
        m_always_equal_md_map = 0;
    }

    enum {
        VARIANT_DEFAULT,
        VARIANT_MAP_NONBLOCK,
        VARIANT_PROTO_ENABLE,
        VARIANT_NO_RCACHE
    };

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants,
                      uint64_t extra_features)
    {
        add_variant_with_value(variants, UCP_FEATURE_RMA | extra_features, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_RMA |extra_features,
                               VARIANT_MAP_NONBLOCK, "map_nb");
        add_variant_with_value(variants, UCP_FEATURE_RMA | extra_features,
                               VARIANT_PROTO_ENABLE, "proto");
        add_variant_with_value(variants, UCP_FEATURE_RMA | extra_features,
                               VARIANT_NO_RCACHE, "no_rcache");
    }

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        get_test_variants(variants, 0);
    }

    void always_equal_md_map_init() {
        ucs_status_t status;
        ucp_mem_h memh1, memh2;
        char dummy[2 * ucs_get_page_size()];

        ucp_mem_map_params_t params;
        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.length     = sizeof(char);

        params.address = &dummy[0];
        status         = ucp_mem_map(sender().ucph(), &params, &memh1);
        ASSERT_UCS_OK(status);

        params.address = &dummy[2 * ucs_get_page_size() - 1];
        status         = ucp_mem_map(sender().ucph(), &params, &memh2);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(memh1->md_map, memh2->md_map);
        ucp_md_index_t md_index;

        m_always_equal_md_map = 0;
        ucs_for_each_bit(md_index, memh1->md_map) {
            if (memh1->uct[md_index] == memh2->uct[md_index]) {
                m_always_equal_md_map |= UCS_BIT(md_index);
            }
        }

        status = ucp_mem_unmap(sender().ucph(), memh1);
        ASSERT_UCS_OK(status);

        status = ucp_mem_unmap(sender().ucph(), memh2);
        ASSERT_UCS_OK(status);
    }

    virtual void init() {
        ucs::skip_on_address_sanitizer();
        if (enable_proto()) {
            modify_config("PROTO_ENABLE", "y");
        }

        if (get_variant_value() == VARIANT_NO_RCACHE) {
            modify_config("RCACHE_ENABLE", "n");
            ucs::scoped_setenv ib_reg_methods_env("UCX_IB_REG_METHODS",
                                                  "direct");
            ucs::scoped_setenv knem_rcache_env("UCX_KNEM_RCACHE", "no");
            ucp_test::init(); // Init UCP with rcache disabled
        } else {
            ucp_test::init();
        }

        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }

        always_equal_md_map_init();
    }

    unsigned mem_map_flags() const {
        return (get_variant_value() == VARIANT_MAP_NONBLOCK) ?
                       UCP_MEM_MAP_NONBLOCK :
                       0;
    }

    bool is_tl_rdma() {
        /* Return true if the selected transport is expected to have remote
         * registered memory access capabilities. If we have both shared memory
         * and rdma options, it's possible that only shared memory is actually
         * used, so can't assume it.
         */
        return (has_transport("dc_x") || has_transport("rc_x") ||
                has_transport("rc_v") || has_transport("ib")) &&
               !is_tl_shm();
    }

    bool is_tl_shm() {
        return has_transport("shm");
    }

    void compare_uct_memhs(const ucp_mem_h memh1, const ucp_mem_h memh2,
                           bool equal = true)
    {
        ucp_md_map_t md_map = memh1->md_map &
                              sender().ucph()->cache_md_map[memh1->mem_type];
        ucp_md_index_t md_index;

        EXPECT_NE(memh1->md_map, 0);

        ucs_for_each_bit(md_index, md_map) {
            if (equal || (m_always_equal_md_map & UCS_BIT(md_index))) {
                EXPECT_EQ(memh2->uct[md_index], memh1->uct[md_index]);
            } else {
                EXPECT_NE(memh2->uct[md_index], memh1->uct[md_index]);
            }
        }
    }

    void compare_memhs(const ucp_mem_h memh1, const ucp_mem_h memh2)
    {
        EXPECT_NE(memh1, memh2);
        EXPECT_EQ(memh1->reg_id, memh2->reg_id);

        compare_uct_memhs(memh1, memh2);
    }

protected:
    bool resolve_rma(entity *e, ucp_rkey_h rkey);
    bool resolve_amo(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw_get_zcopy(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw_put_zcopy(entity *e, ucp_rkey_h rkey);
    void test_length0(unsigned flags);
    void test_rereg(unsigned map_flags = 0, uint64_t memh_pack_flags = 0,
                    bool import_mem = false);
    void test_rkey_management(ucp_mem_h memh, bool is_dummy,
                              bool expect_rma_offload);
    bool enable_proto() const;

private:
    void expect_same_distance(const ucs_sys_dev_distance_t &dist1,
                              const ucs_sys_dev_distance_t &dist2);
    void test_rkey_proto(ucp_mem_h memh);
    void test_rereg_local_mem(ucp_mem_h memh, void *ptr, size_t size,
                              unsigned map_flags);
    static ucs_log_func_rc_t
    import_no_md_error_handler(const char *file, unsigned line,
                               const char *function, ucs_log_level_t level,
                               const ucs_log_component_config_t *comp_conf,
                               const char *message, va_list ap);
    void import_memh(void *exported_memh_buf, ucp_mem_h *memh_p);
    void release_exported_memh_buf(void *exported_memh_buf);
    void test_rereg_imported_mem(ucp_mem_h memh, uint64_t memh_pack_flags,
                                 size_t size);

protected:
    ucp_md_map_t m_always_equal_md_map;
};

bool test_ucp_mmap::resolve_rma(entity *e, ucp_rkey_h rkey)
{
    ucs_status_t status;

    {
        scoped_log_handler slh(hide_errors_logger);
        status = UCP_RKEY_RESOLVE(rkey, e->ep(), rma);
    }

    if (status == UCS_OK) {
        EXPECT_NE(UCP_NULL_LANE, rkey->cache.rma_lane);
        return true;
    } else if (status == UCS_ERR_UNREACHABLE) {
        EXPECT_EQ(UCP_NULL_LANE, rkey->cache.rma_lane);
        return false;
    } else {
        UCS_TEST_ABORT("Invalid status from UCP_RKEY_RESOLVE");
    }
}

bool test_ucp_mmap::resolve_amo(entity *e, ucp_rkey_h rkey)
{
    ucs_status_t status;

    {
        scoped_log_handler slh(hide_errors_logger);
        status = UCP_RKEY_RESOLVE(rkey, e->ep(), amo);
    }

    if (status == UCS_OK) {
        EXPECT_NE(UCP_NULL_LANE, rkey->cache.amo_lane);
        return true;
    } else if (status == UCS_ERR_UNREACHABLE) {
        EXPECT_EQ(UCP_NULL_LANE, rkey->cache.amo_lane);
        return false;
    } else {
        UCS_TEST_ABORT("Invalid status from UCP_RKEY_RESOLVE");
    }
}

bool test_ucp_mmap::resolve_rma_bw_get_zcopy(entity *e, ucp_rkey_h rkey)
{
    ucp_ep_config_t *ep_config = ucp_ep_config(e->ep());
    ucp_lane_index_t lane;
    uct_rkey_t uct_rkey;

    lane = ucp_rkey_find_rma_lane(e->ucph(), ep_config, UCS_MEMORY_TYPE_HOST,
                                  ep_config->rndv.get_zcopy.lanes, rkey, 0,
                                  &uct_rkey);
    if (lane != UCP_NULL_LANE) {
        return true;
    } else {
        return false;
    }
}

bool test_ucp_mmap::resolve_rma_bw_put_zcopy(entity *e, ucp_rkey_h rkey)
{
    ucp_ep_config_t *ep_config = ucp_ep_config(e->ep());
    ucp_lane_index_t lane;
    uct_rkey_t uct_rkey;

    lane = ucp_rkey_find_rma_lane(e->ucph(), ep_config, UCS_MEMORY_TYPE_HOST,
                                  ep_config->rndv.put_zcopy.lanes, rkey, 0,
                                  &uct_rkey);
    if (lane != UCP_NULL_LANE) {
        return true;
    } else {
        return false;
    }
}

void test_ucp_mmap::test_rkey_management(ucp_mem_h memh, bool is_dummy,
                                         bool expect_rma_offload)
{
    size_t rkey_size;
    void *rkey_buffer;
    ucs_status_t status;

    /* Some transports don't support memory registration, so the memory
     * can be inaccessible remotely. But it should always be possible
     * to pack/unpack a key, even if empty. */
    status = ucp_rkey_pack(sender().ucph(), memh, &rkey_buffer, &rkey_size);
    if ((status == UCS_ERR_UNSUPPORTED) && !is_dummy) {
        return;
    }
    ASSERT_UCS_OK(status);

    EXPECT_EQ(ucp_rkey_packed_size(sender().ucph(), memh->md_map,
                                   UCS_SYS_DEVICE_ID_UNKNOWN, 0),
              rkey_size);

    /* Unpack remote key buffer */
    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(receiver().ep(), rkey_buffer, &rkey);
    if ((status == UCS_ERR_UNREACHABLE) && !is_dummy) {
        ucp_rkey_buffer_release(rkey_buffer);
        return;
    }
    ASSERT_UCS_OK(status);

    /* Test ucp_rkey_packed_md_map() */
    EXPECT_EQ(memh->md_map, ucp_rkey_packed_md_map(rkey_buffer));

    /* rkey->md_map is a subset of all possible keys */
    EXPECT_TRUE(ucs_test_all_flags(memh->md_map, rkey->md_map));

    /* Test remote key protocols selection */
    if (m_ucp_config->ctx.proto_enable) {
        test_rkey_proto(memh);
    } else {
        bool have_rma              = resolve_rma(&receiver(), rkey);
        bool have_amo              = resolve_amo(&receiver(), rkey);
        bool have_rma_bw_get_zcopy = resolve_rma_bw_get_zcopy(&receiver(),
                                                              rkey);
        bool have_rma_bw_put_zcopy = resolve_rma_bw_put_zcopy(&receiver(),
                                                              rkey);

        /* Test that lane resolution on the remote key returns consistent results */
        for (int i = 0; i < 10; ++i) {
            switch (ucs::rand() % 4) {
            case 0:
                EXPECT_EQ(have_rma, resolve_rma(&receiver(), rkey));
                break;
            case 1:
                EXPECT_EQ(have_amo, resolve_amo(&receiver(), rkey));
                break;
            case 2:
                EXPECT_EQ(have_rma_bw_get_zcopy,
                          resolve_rma_bw_get_zcopy(&receiver(), rkey));
                break;
            case 3:
                EXPECT_EQ(have_rma_bw_put_zcopy,
                          resolve_rma_bw_put_zcopy(&receiver(), rkey));
                break;
            }
        }

        if (expect_rma_offload) {
            if (is_dummy) {
                EXPECT_EQ(&ucp_rma_sw_proto,
                          UCP_RKEY_RMA_PROTO(rkey->cache.rma_proto_index));
            } else {
                ucs_assert(&ucp_rma_basic_proto ==
                           UCP_RKEY_RMA_PROTO(rkey->cache.rma_proto_index));
                EXPECT_EQ(&ucp_rma_basic_proto,
                          UCP_RKEY_RMA_PROTO(rkey->cache.rma_proto_index));
            }
        }
    }

    /* Test obtaining direct-access pointer */
    void *ptr;
    status = ucp_rkey_ptr(rkey, (uint64_t)ucp_memh_address(memh), &ptr);
    if (status == UCS_OK) {
        EXPECT_EQ(0, memcmp(ucp_memh_address(memh), ptr, ucp_memh_length(memh)));
    } else {
        EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
    }

    ucp_rkey_destroy(rkey);
    ucp_rkey_buffer_release(rkey_buffer);
}

bool test_ucp_mmap::enable_proto() const
{
    return get_variant_value() == VARIANT_PROTO_ENABLE;
}

void test_ucp_mmap::expect_same_distance(const ucs_sys_dev_distance_t &dist1,
                                         const ucs_sys_dev_distance_t &dist2)
{
    /* Expect the implementation to always provide a reasonable precision w.r.t.
     * real-world bandwidth and latency ballpark numbers.
     */
    EXPECT_NEAR(dist1.bandwidth, dist2.bandwidth, 600e6); /* 600 MBs accuracy */
    EXPECT_NEAR(dist1.latency, dist2.latency, 20e-9); /* 20 nsec accuracy */
}

void test_ucp_mmap::test_rkey_proto(ucp_mem_h memh)
{
    ucs_status_t status;

    /* Detect system device of the allocated memory */
    ucp_memory_info_t mem_info;
    ucp_memory_detect(sender().ucph(), ucp_memh_address(memh),
                      ucp_memh_length(memh), &mem_info);
    EXPECT_EQ(memh->mem_type, mem_info.type);

    /* Collect distances from all devices in the system */
    ucp_sys_dev_map_t sys_dev_map = UCS_MASK(ucs_topo_num_devices());
    std::vector<ucs_sys_dev_distance_t> sys_distance(ucs_topo_num_devices());
    for (unsigned i = 0; i < sys_distance.size(); ++i) {
        if (std::string(ucs_topo_sys_device_get_name(i)).find("test") == 0) {
            /* Dummy device created by test */
            continue;
        }

        status = ucs_topo_get_distance(mem_info.sys_dev, i, &sys_distance[i]);
        ASSERT_UCS_OK(status);
    }

    /* Allocate buffer for packed rkey */
    size_t rkey_size = ucp_rkey_packed_size(sender().ucph(), memh->md_map,
                                            mem_info.sys_dev, sys_dev_map);
    std::string rkey_buffer(rkey_size, '0');

    /* Pack the rkey and validate packed size */
    ssize_t packed_size = ucp_rkey_pack_memh(sender().ucph(), memh->md_map,
                                             memh, &mem_info, sys_dev_map,
                                             &sys_distance[0], &rkey_buffer[0]);
    ASSERT_EQ((ssize_t)rkey_size, packed_size);

    /* Unpack remote key buffer */
    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack_reachable(receiver().ep(), &rkey_buffer[0],
                                          rkey_size, &rkey);
    ASSERT_UCS_OK(status);

    /* Check rkey configuration */
    if (enable_proto()) {
        ucp_rkey_config_t *rkey_config = ucp_rkey_config(receiver().worker(),
                                                         rkey);
        ucp_ep_config_t *ep_config     = ucp_ep_config(receiver().ep());

        EXPECT_EQ(receiver().ep()->cfg_index, rkey_config->key.ep_cfg_index);
        EXPECT_EQ(mem_info.sys_dev, rkey_config->key.sys_dev);
        EXPECT_EQ(mem_info.type, rkey_config->key.mem_type);

        /* Compare original system distance and unpacked rkey system distance */
        for (ucp_lane_index_t lane = 0; lane < ep_config->key.num_lanes;
             ++lane) {
            ucs_sys_device_t sys_dev = ep_config->key.lanes[lane].dst_sys_dev;
            expect_same_distance(rkey_config->lanes_distance[lane],
                                 (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) ?
                                         ucs_topo_default_distance :
                                         sys_distance[sys_dev]);
        }
    }

    ucp_rkey_destroy(rkey);
}

UCS_TEST_P(test_ucp_mmap, alloc_mem_type) {
    const std::vector<ucs_memory_type_t> &mem_types =
            mem_buffer::supported_mem_types();
    ucs_status_t status;
    bool is_dummy;
    bool expect_rma_offload;

    for (auto mem_type : mem_types) {
        for (int i = 0; i < (100 / ucs::test_time_multiplier()); ++i) {
            size_t size = ucs::rand() % (UCS_MBYTE);

            ucp_mem_h memh;
            ucp_mem_map_params_t params;
            params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                 UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                                 UCP_MEM_MAP_PARAM_FIELD_FLAGS   |
                                 UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            params.address     = NULL;
            params.memory_type = mem_type;
            params.length      = size;
            params.flags       = UCP_MEM_MAP_ALLOCATE;

            status = ucp_mem_map(sender().ucph(), &params, &memh);

            ASSERT_UCS_OK(status);

            is_dummy           = (size == 0);
            expect_rma_offload = !UCP_MEM_IS_CUDA_MANAGED(mem_type) &&
                                 (is_tl_rdma() || is_tl_shm()) &&
                                 check_reg_mem_types(sender(), mem_type);
            test_rkey_management(memh, is_dummy, expect_rma_offload);

            status = ucp_mem_unmap(sender().ucph(), memh);
            ASSERT_UCS_OK(status);
        }
    }
}

UCS_TEST_P(test_ucp_mmap, reg_mem_type) {
    const std::vector<ucs_memory_type_t> &mem_types =
            mem_buffer::supported_mem_types();
    ucs_status_t status;
    bool is_dummy;
    bool expect_rma_offload;
    ucs_memory_type_t alloc_mem_type;

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size    = ucs::rand() % UCS_MBYTE;
        alloc_mem_type = mem_types.at(ucs::rand() % mem_types.size());
        mem_buffer buf(size, alloc_mem_type);
        mem_buffer::pattern_fill(buf.ptr(), size, 0, alloc_mem_type);

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        params.address     = buf.ptr();
        params.length      = size;
        params.memory_type = alloc_mem_type;
        params.flags       = mem_map_flags();

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);

        is_dummy = (size == 0);
        if (!is_dummy) {
            EXPECT_EQ(alloc_mem_type, memh->mem_type);
        }

        expect_rma_offload = !UCP_MEM_IS_CUDA_MANAGED(alloc_mem_type) &&
                             !UCP_MEM_IS_ROCM_MANAGED(alloc_mem_type) &&
                             is_tl_rdma() &&
                             check_reg_mem_types(sender(), alloc_mem_type);
        test_rkey_management(memh, is_dummy, expect_rma_offload);

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

void test_ucp_mmap::test_rereg_local_mem(ucp_mem_h memh, void *ptr,
                                         size_t size, unsigned map_flags)
{
    ucs_status_t status;
    const int num_iters     = 4;
    const void *end_address = UCS_PTR_BYTE_OFFSET(ptr, size);

    for (int i = 0; i < num_iters; ++i) {
        size_t offset       = (size != 0) ? ucs::rand() % size : 0;
        void *start_address = UCS_PTR_BYTE_OFFSET(ptr, offset);
        ucp_mem_h test_memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.flags      = mem_map_flags() | map_flags;
        params.address    = start_address;
        params.length     = UCS_PTR_BYTE_DIFF(start_address, end_address);
        status            = ucp_mem_map(sender().ucph(), &params, &test_memh);
        ASSERT_UCS_OK(status);

        if (size == 0) {
            EXPECT_EQ(memh, test_memh);
            EXPECT_EQ(&ucp_mem_dummy_handle.memh, test_memh);
        } else if (get_variant_value() == VARIANT_NO_RCACHE) {
            EXPECT_NE(test_memh->reg_id, memh->reg_id);
            compare_uct_memhs(test_memh, memh, false);
        } else {
            compare_memhs(test_memh, memh);
        }

        status = ucp_mem_unmap(sender().ucph(), test_memh);
        ASSERT_UCS_OK(status);
    }
}

ucs_log_func_rc_t
test_ucp_mmap::import_no_md_error_handler(const char *file, unsigned line,
                                          const char *function,
                                          ucs_log_level_t level,
                                          const ucs_log_component_config_t *comp_conf,
                                          const char *message, va_list ap)
{
    // Ignore errors that no suitable MDs for import as it is expected
    if (level == UCS_LOG_LEVEL_ERROR) {
        std::string err_str = format_message(message, ap);
        if (err_str.find("no suitable UCT memory domains to perform importing"
                         " on") != std::string::npos) {
            UCS_TEST_MESSAGE << err_str;
            return UCS_LOG_FUNC_RC_STOP;
        }
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

void test_ucp_mmap::release_exported_memh_buf(void *exported_memh_buf)
{
    const ucp_memh_buffer_release_params_t release_params = { 0 };
    ucp_memh_buffer_release(exported_memh_buf, &release_params);
}

void test_ucp_mmap::import_memh(void *exported_memh_buf, ucp_mem_h *memh_p)
{
    ucp_mem_map_params_t params;

    params.field_mask           =
            UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = exported_memh_buf;

    {
        scoped_log_handler warn_slh(import_no_md_error_handler);
        ucs_status_t status = ucp_mem_map(receiver().ucph(), &params, memh_p);
        if (status == UCS_ERR_UNREACHABLE) {
            release_exported_memh_buf(exported_memh_buf);
            UCS_TEST_SKIP_R("memory importing is unsupported");
        }
        ASSERT_UCS_OK(status);
    }
}

void test_ucp_mmap::test_rereg_imported_mem(ucp_mem_h memh,
                                            uint64_t memh_pack_flags,
                                            size_t size)
{
    ucp_memh_pack_params_t pack_params;
    ucs_status_t status;
    void *exported_memh_buf;
    size_t exported_memh_buf_size;

    pack_params.field_mask = UCP_MEMH_PACK_PARAM_FIELD_FLAGS;
    pack_params.flags      = memh_pack_flags;

    status = ucp_memh_pack(memh, &pack_params, &exported_memh_buf,
                           &exported_memh_buf_size);
    if ((status == UCS_ERR_UNSUPPORTED) &&
        (pack_params.flags & UCP_MEMH_PACK_FLAG_EXPORT)) {
        UCS_TEST_SKIP_R("memory exporting is unsupported");
    }
    ASSERT_UCS_OK(status);

    ucp_mem_h imp_memh;
    import_memh(exported_memh_buf, &imp_memh);

    ucp_mem_h test_imp_memh;
    import_memh(exported_memh_buf, &test_imp_memh);

    release_exported_memh_buf(exported_memh_buf);

    if (size == 0) {
        EXPECT_EQ(memh, test_imp_memh);
        EXPECT_EQ(&ucp_mem_dummy_handle.memh, test_imp_memh);
    } else if (get_variant_value() == VARIANT_NO_RCACHE) {
        EXPECT_EQ(test_imp_memh->reg_id, memh->reg_id);
        compare_uct_memhs(test_imp_memh, imp_memh, false);
    } else {
        compare_memhs(test_imp_memh, imp_memh);
    }

    status = ucp_mem_unmap(receiver().ucph(), test_imp_memh);
    ASSERT_UCS_OK(status);

    status = ucp_mem_unmap(receiver().ucph(), imp_memh);
    ASSERT_UCS_OK(status);
}

void test_ucp_mmap::test_rereg(unsigned map_flags,
                               uint64_t memh_pack_flags,
                               bool import_mem)
{
    ucs_status_t status;

    for (int i = 0; i < (100 / ucs::test_time_multiplier()); ++i) {
        size_t size = ucs::rand() % UCS_MBYTE;
        mem_buffer *buf;
        void *ptr;

        if (map_flags & UCP_MEM_MAP_ALLOCATE) {
            buf = NULL;
            ptr = NULL;
        } else {
            buf = new mem_buffer(size, UCS_MEMORY_TYPE_HOST);
            ptr = buf->ptr();
        }

        ucp_mem_h memh;
        ucp_mem_map_params_t params;
        params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address     = ptr;
        params.length      = size;
        params.flags       = mem_map_flags() | map_flags;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);

        ucp_mem_attr_t memh_attr;
        memh_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
        status               = ucp_mem_query(memh, &memh_attr);
        ASSERT_UCS_OK(status);
        ptr = memh_attr.address;

        if (import_mem) {
            try {
                test_rereg_imported_mem(memh, memh_pack_flags, size);
            } catch (ucs::test_skip_exception &e) {
                status = ucp_mem_unmap(sender().ucph(), memh);
                ASSERT_UCS_OK(status);
                delete buf;
                throw e;
            }
        } else {
            test_rereg_local_mem(memh, ptr, size,
                                 map_flags & ~UCP_MEM_MAP_ALLOCATE);
        }

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);

        delete buf;
    }
}

UCS_TEST_P(test_ucp_mmap, rereg)
{
    test_rereg();
}

UCS_TEST_P(test_ucp_mmap, alloc_rereg)
{
    test_rereg(UCP_MEM_MAP_ALLOCATE);
}

void test_ucp_mmap::test_length0(unsigned flags)
{
    ucs_status_t status;
    int buf_num = 2;
    ucp_mem_h memh[buf_num];
    int dummy[1];
    ucp_mem_map_params_t params;
    int i;

    /* Check that ucp_mem_map accepts any value for buffer if size is 0 and
     * UCP_MEM_FLAG_ZERO_REG flag is passed to it. */

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = NULL;
    params.length     = 0;
    params.flags      = mem_map_flags() | flags;

    status = ucp_mem_map(sender().ucph(), &params, &memh[0]);
    ASSERT_UCS_OK(status);

    params.address = dummy;
    status = ucp_mem_map(sender().ucph(), &params, &memh[1]);
    ASSERT_UCS_OK(status);

    bool expect_rma_offload = is_tl_rdma() ||
                              ((flags & UCP_MEM_MAP_ALLOCATE) &&
                               is_tl_shm());

    for (i = 0; i < buf_num; i++) {
        test_rkey_management(memh[i], true, expect_rma_offload);
        test_rkey_proto(memh[i]);
        status = ucp_mem_unmap(sender().ucph(), memh[i]);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_P(test_ucp_mmap, reg0) {
    test_length0(0);
}

UCS_TEST_P(test_ucp_mmap, alloc0) {
    test_length0(UCP_MEM_MAP_ALLOCATE);
}

UCS_TEST_P(test_ucp_mmap, alloc_advise) {
    ucs_status_t status;
    bool is_dummy;

    const size_t size = ucs_max(UCS_KBYTE,
                                128 * UCS_MBYTE / ucs::test_time_multiplier());

    ucp_mem_h memh;
    ucp_mem_map_params_t params;
    ucp_mem_attr_t attr;
    ucp_mem_advise_params_t advise_params;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = NULL;
    params.length     = size;
    params.flags      = UCP_MEM_MAP_NONBLOCK | UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(sender().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH |
                      UCP_MEM_ATTR_FIELD_MEM_TYPE;
    status = ucp_mem_query(memh, &attr);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(attr.mem_type, UCS_MEMORY_TYPE_HOST);
    EXPECT_GE(attr.length, size);

    advise_params.field_mask = UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS |
                               UCP_MEM_ADVISE_PARAM_FIELD_LENGTH |
                               UCP_MEM_ADVISE_PARAM_FIELD_ADVICE;
    advise_params.address    = attr.address;
    advise_params.length     = size;
    advise_params.advice     = UCP_MADV_WILLNEED;
    status = ucp_mem_advise(sender().ucph(), memh, &advise_params);
    ASSERT_UCS_OK(status);

    is_dummy = (size == 0);
    test_rkey_management(memh, is_dummy, is_tl_rdma() || is_tl_shm());

    status = ucp_mem_unmap(sender().ucph(), memh);
    ASSERT_UCS_OK(status);
}

UCS_TEST_P(test_ucp_mmap, reg_advise) {
    ucs_status_t status;
    bool is_dummy;

    const size_t size = ucs_max(UCS_KBYTE,
                                128 * UCS_MBYTE / ucs::test_time_multiplier());
    void *ptr         = malloc(size);
    ucs::fill_random(ptr, size);

    ucp_mem_h               memh;
    ucp_mem_map_params_t    params;
    ucp_mem_attr_t          mem_attr;
    ucp_mem_advise_params_t advise_params;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = ptr;
    params.length     = size;
    params.flags      = UCP_MEM_MAP_NONBLOCK;

    status = ucp_mem_map(sender().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(memh, &mem_attr);
    ASSERT_UCS_OK(status);

    advise_params.field_mask = UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS |
                               UCP_MEM_ADVISE_PARAM_FIELD_LENGTH |
                               UCP_MEM_ADVISE_PARAM_FIELD_ADVICE;
    advise_params.address    = mem_attr.address;
    advise_params.length     = size;
    advise_params.advice     = UCP_MADV_WILLNEED;
    status = ucp_mem_advise(sender().ucph(), memh, &advise_params);
    ASSERT_UCS_OK(status);
    is_dummy = (size == 0);
    test_rkey_management(memh, is_dummy, is_tl_rdma());

    status = ucp_mem_unmap(sender().ucph(), memh);
    ASSERT_UCS_OK(status);

    free(ptr);
}

UCS_TEST_P(test_ucp_mmap, fixed) {
    ucs_status_t status;
    bool         is_dummy;

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = (i + 1) * ((i % 2) ? 1000 : 1);
        void *ptr   = ucs::mmap_fixed_address(size);
        if (ptr == nullptr) {
            UCS_TEST_ABORT("mmap failed to allocate memory region");
        }

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = ptr;
        params.length     = size;
        params.flags      = UCP_MEM_MAP_FIXED | UCP_MEM_MAP_ALLOCATE;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ(ucp_memh_address(memh), ptr);
        EXPECT_GE(ucp_memh_length(memh), size);

        is_dummy = (size == 0);
        test_rkey_management(memh, is_dummy, is_tl_rdma());

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_mmap)


class test_ucp_mmap_export : public test_ucp_mmap {
public:
    static void
    get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        test_ucp_mmap::get_test_variants(variants, UCP_FEATURE_EXPORTED_MEMH);
    }
};

UCS_TEST_P(test_ucp_mmap_export, reg_export_and_reimport)
{
    test_rereg(0, UCP_MEMH_PACK_FLAG_EXPORT, true);
}

UCS_TEST_P(test_ucp_mmap_export, alloc_reg_export_and_reimport)
{
    test_rereg(UCP_MEM_MAP_ALLOCATE, UCP_MEMH_PACK_FLAG_EXPORT, true);
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_mmap_export)
