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
#include <ucs/type/float8.h>
}

#include <cmath>
#include <list>

class test_ucp_mmap : public ucp_test {
public:
    test_ucp_mmap()
    {
        m_always_equal_md_map = 0;
    }

    enum {
        VARIANT_DEFAULT,
        VARIANT_MAP_NONBLOCK,
        VARIANT_PROTO_DISABLE,
        VARIANT_NO_RCACHE
    };

    struct mem_chunk {
        ucp_context_h           context;
        ucp_mem_h               memh;
        std::vector<ucp_rkey_h> rkeys;

        mem_chunk(ucp_context_h);
        ~mem_chunk();
        ucp_rkey_h unpack(ucp_ep_h, ucp_md_map_t md_map = 0);
    };

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants,
                      uint64_t extra_features)
    {
        add_variant_with_value(variants, UCP_FEATURE_RMA | extra_features, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_RMA |extra_features,
                               VARIANT_MAP_NONBLOCK, "map_nb");
        if (!RUNNING_ON_VALGRIND) {
            add_variant_with_value(variants, UCP_FEATURE_RMA | extra_features,
                                   VARIANT_PROTO_DISABLE, "proto_v1");
        }
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
        if (get_variant_value() == VARIANT_PROTO_DISABLE) {
            modify_config("PROTO_ENABLE", "n");
        }

        if (get_variant_value() == VARIANT_MAP_NONBLOCK) {
            // ODPv1 cannot interact with DEVX objects
            modify_config("IB_MLX5_DEVX_OBJECTS", "", SETENV_IF_NOT_EXIST);
        }

        if (get_variant_value() == VARIANT_NO_RCACHE) {
            modify_config("RCACHE_ENABLE", "n");
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

    ucp_mem_h import_memh(ucp_mem_h exported_memh);

protected:
    bool resolve_rma(entity *e, ucp_rkey_h rkey);
    bool resolve_amo(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw_get_zcopy(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw_put_zcopy(entity *e, ucp_rkey_h rkey);
    void test_length0(unsigned flags);
    void test_rereg(unsigned map_flags = 0, bool import_mem = false);
    void test_rkey_management(ucp_mem_h memh, bool is_dummy,
                              bool expect_rma_offload);

private:
    void check_distance_precision(double rkey_value, double topo_value,
                                  size_t pack_min, size_t pack_max);
    void test_rkey_proto(ucp_mem_h memh);
    void test_rereg_local_mem(ucp_mem_h memh, void *ptr, size_t size,
                              unsigned map_flags);
    static ucs_log_func_rc_t
    import_no_md_error_handler(const char *file, unsigned line,
                               const char *function, ucs_log_level_t level,
                               const ucs_log_component_config_t *comp_conf,
                               const char *message, va_list ap);
    void release_exported_memh_buf(void *exported_memh_buf);
    void test_rereg_imported_mem(ucp_mem_h memh, size_t size);

protected:
    ucp_md_map_t m_always_equal_md_map;
};

test_ucp_mmap::mem_chunk::mem_chunk(ucp_context_h ctx) : context(ctx)
{
    ucp_mem_map_params_t params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                      UCP_MEM_MAP_PARAM_FIELD_FLAGS,
        .address    = NULL,
        .length     = 4096,
        .flags      = UCP_MEM_MAP_ALLOCATE,
    };

    ASSERT_UCS_OK(ucp_mem_map(context, &params, &memh));
}

test_ucp_mmap::mem_chunk::~mem_chunk()
{
    for (auto &rkey : rkeys) {
        ucp_rkey_destroy(rkey);
    }

    EXPECT_UCS_OK(ucp_mem_unmap(context, memh));
}

ucp_rkey_h test_ucp_mmap::mem_chunk::unpack(ucp_ep_h ep, ucp_md_map_t md_map)
{
    ucp_rkey_h rkey;
    void *rkey_buffer;
    size_t rkey_size;

    ASSERT_UCS_OK(ucp_rkey_pack(context, memh, &rkey_buffer, &rkey_size));
    if (md_map == 0) {
        ASSERT_UCS_OK(ucp_ep_rkey_unpack(ep, rkey_buffer, &rkey));
    } else {
        // Different MD map means different config index on proto v2
        ASSERT_UCS_OK(ucp_ep_rkey_unpack_internal(ep, rkey_buffer, rkey_size,
                                                  md_map, 0, &rkey));
    }

    ucp_rkey_buffer_release(rkey_buffer);
    rkeys.push_back(rkey);
    return rkey;
}

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
    if (is_proto_enabled()) {
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

void test_ucp_mmap::check_distance_precision(double rkey_value,
                                             double topo_value,
                                             size_t pack_min,
                                             size_t pack_max)
{
    /* Expect the implementation to always provide a reasonable precision w.r.t.
     * real-world bandwidth and latency ballpark numbers.
     */
    double allowed_diff_ratio = 1 - UCS_FP8_PRECISION;

    if (rkey_value == pack_min) {
        /* Capped by pack_min, no cache entry */
        EXPECT_LE(std::lround(topo_value), pack_min);
    } else if (rkey_value == pack_max) {
        /* Capped by pack_max, no cache entry */
        EXPECT_GE(std::lround(topo_value), pack_max);
    } else if (topo_value == INFINITY) {
        /* Infinity values can be packed without loss */
        EXPECT_EQ(topo_value, rkey_value);
    } else {
        /* Inside the borders or cache entry */
        EXPECT_NEAR(rkey_value, topo_value, topo_value * allowed_diff_ratio);
    }
}

void test_ucp_mmap::test_rkey_proto(ucp_mem_h memh)
{
    ucs_sys_dev_distance_t rkey_dist, topo_dist;
    ucs_sys_device_t sys_dev;
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
                                             memh, ucp_memh_address(memh),
                                             ucp_memh_length(memh), &mem_info,
                                             sys_dev_map, &sys_distance[0], 0,
                                             &rkey_buffer[0]);
    ASSERT_EQ((ssize_t)rkey_size, packed_size);

    /* Unpack remote key buffer */
    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack_reachable(receiver().ep(), &rkey_buffer[0],
                                          rkey_size, &rkey);
    ASSERT_UCS_OK(status);

    /* Check rkey configuration */
    if (is_proto_enabled()) {
        ucp_rkey_config_t *rkey_config = ucp_rkey_config(receiver().worker(),
                                                         rkey);
        ucp_ep_config_t *ep_config     = ucp_ep_config(receiver().ep());

        EXPECT_EQ(receiver().ep()->cfg_index, rkey_config->key.ep_cfg_index);
        EXPECT_EQ(mem_info.sys_dev, rkey_config->key.sys_dev);
        EXPECT_EQ(mem_info.type, rkey_config->key.mem_type);

        /* Compare original system distance and unpacked rkey system distance */
        for (ucp_lane_index_t lane = 0; lane < ep_config->key.num_lanes;
             ++lane) {
            sys_dev   = ep_config->key.lanes[lane].dst_sys_dev;
            rkey_dist = rkey_config->lanes_distance[lane];
            topo_dist = (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) ?
                        ucs_topo_default_distance : sys_distance[sys_dev];

            check_distance_precision(rkey_dist.bandwidth, topo_dist.bandwidth,
                                     UCS_FP8_MIN_BW, UCS_FP8_MAX_BW);
            check_distance_precision(rkey_dist.latency, topo_dist.latency,
                                     UCS_FP8_MIN_LAT, UCS_FP8_MAX_LAT);
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
            expect_rma_offload = (is_tl_rdma() || is_tl_shm()) &&
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
        auto flags     = mem_map_flags();
        /* Test ODP registration with host buffer only */
        alloc_mem_type = (flags & VARIANT_MAP_NONBLOCK) ?
                UCS_MEMORY_TYPE_HOST :
                mem_types.at(ucs::rand() % mem_types.size());
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
        params.flags       = flags;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);

        is_dummy = (size == 0);
        if (!is_dummy) {
            EXPECT_EQ(alloc_mem_type, memh->mem_type);
        }

        expect_rma_offload = !UCP_MEM_IS_ROCM_MANAGED(alloc_mem_type) &&
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

ucp_mem_h test_ucp_mmap::import_memh(ucp_mem_h exported_memh)
{
    ucp_memh_pack_params_t pack_params;
    ucs_status_t status;
    void *exported_memh_buf;
    size_t exported_memh_buf_size;

    pack_params.field_mask = UCP_MEMH_PACK_PARAM_FIELD_FLAGS;
    pack_params.flags      = UCP_MEMH_PACK_FLAG_EXPORT;

    status = ucp_memh_pack(exported_memh, &pack_params, &exported_memh_buf,
                           &exported_memh_buf_size);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("memory exporting is unsupported");
    }
    ASSERT_UCS_OK(status);

    ucp_mem_map_params_t params;
    params.field_mask           =
            UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = exported_memh_buf;

    ucp_mem_h imported_memh;
    scoped_log_handler warn_slh(import_no_md_error_handler);
    status = ucp_mem_map(receiver().ucph(), &params, &imported_memh);
    if (status == UCS_ERR_UNREACHABLE) {
        release_exported_memh_buf(exported_memh_buf);
        UCS_TEST_SKIP_R("memory importing is unsupported");
    }

    release_exported_memh_buf(exported_memh_buf);
    ASSERT_UCS_OK(status);
    return imported_memh;
}

void test_ucp_mmap::test_rereg_imported_mem(ucp_mem_h memh, size_t size)
{
    ucp_mem_h imp_memh      = import_memh(memh);
    ucp_mem_h test_imp_memh = import_memh(memh);

    if (size == 0) {
        EXPECT_EQ(memh, test_imp_memh);
        EXPECT_EQ(&ucp_mem_dummy_handle.memh, test_imp_memh);
    } else if (get_variant_value() == VARIANT_NO_RCACHE) {
        EXPECT_EQ(test_imp_memh->reg_id, memh->reg_id);
        compare_uct_memhs(test_imp_memh, imp_memh, false);
    } else {
        compare_memhs(test_imp_memh, imp_memh);
    }

    ASSERT_UCS_OK(ucp_mem_unmap(receiver().ucph(), test_imp_memh));
    ASSERT_UCS_OK(ucp_mem_unmap(receiver().ucph(), imp_memh));
}

void test_ucp_mmap::test_rereg(unsigned map_flags, bool import_mem)
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
                test_rereg_imported_mem(memh, size);
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

    /* Make sure eps are connected, because UCX async thread may add some
     * progress callbacks to worker callback queue
     * (e.g. ucp_worker_iface_check_events_progress) and mmap some memory
     * for it (see ucs_callbackq_array_grow->ucs_sys_realloc). This mapped
     * address may conflict with the one used in this test, because
     * ucs::mmap_fixed_address() does mmap/munmap to obtain a pointer for
     * ucp_mem_map() with UCP_MEM_MAP_FIXED which creates a race with async
     * thread.
     */
    flush_workers();

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size = (i + 1) * ((i % 2) ? 1000 : 1);
        ucs::mmap_fixed_address ptr(size);

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = *ptr;
        params.length     = size;
        params.flags      = UCP_MEM_MAP_FIXED | UCP_MEM_MAP_ALLOCATE;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);
        EXPECT_EQ(ucp_memh_address(memh), *ptr);
        EXPECT_GE(ucp_memh_length(memh), size);

        is_dummy = (size == 0);
        test_rkey_management(memh, is_dummy, is_tl_rdma());

        ptr.detach();
        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);
    }
}

UCS_TEST_P(test_ucp_mmap, gva_allocate, "GVA_ENABLE=y")
{
    for (auto mem_type : mem_buffer::supported_mem_types()) {
        ucp_md_map_t md_map = sender().ucph()->gva_md_map[mem_type];
        if (md_map == 0) {
            continue;
        }

        ucp_mem_h memh;
        ucp_mem_map_params_t params;
        params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        params.address     = NULL;
        params.memory_type = mem_type;
        params.length      = 1 * UCS_MBYTE;
        params.flags       = UCP_MEM_MAP_ALLOCATE;

        ASSERT_UCS_OK(ucp_mem_map(sender().ucph(), &params, &memh));
        EXPECT_TRUE(ucs_test_all_flags(memh->md_map, md_map));
        ASSERT_UCS_OK(ucp_mem_unmap(sender().ucph(), memh));
    }
}

UCS_TEST_P(test_ucp_mmap, gva, "GVA_ENABLE=y")
{
    std::list<void*> bufs;
    ucp_mem_h first     = NULL;
    ucp_md_map_t md_map = 0;

    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {
        size_t size  = (i + 1) * ((i % 2) ? 1000 : 1);
        void *buffer = ucs_malloc(size, "gva");
        bufs.push_back(buffer);

        ucp_mem_h memh;
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.address    = buffer;
        params.length     = size;

        ASSERT_UCS_OK(ucp_mem_map(sender().ucph(), &params, &memh));

        if (i == 0) {
            first  = memh;
            md_map = memh->md_map & sender().ucph()->gva_md_map[memh->mem_type];
            if (md_map == 0) {
                UCS_TEST_MESSAGE << "no GVA";
                break;
            }
        } else {
            ucp_md_index_t md_index;

            ucs_for_each_bit(md_index, md_map) {
                EXPECT_EQ(memh->uct[md_index], first->uct[md_index]);
            }

            ASSERT_UCS_OK(ucp_mem_unmap(sender().ucph(), memh));
        }
    }

    ASSERT_UCS_OK(ucp_mem_unmap(sender().ucph(), first));
    for (auto *buffer : bufs) {
        ucs_free(buffer);
    }
}


UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_mmap)

class test_ucp_mmap_atomic : public test_ucp_mmap {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        test_ucp_mmap::get_test_variants(variants,
                                         UCP_FEATURE_TAG | UCP_FEATURE_AMO64);
    }
};

/* Use a buffer for send/recv, and then reuse it for atomic operations */
UCS_TEST_P(test_ucp_mmap_atomic, reuse_buffer)
{
    mem_buffer sbuf(UCS_MBYTE, UCS_MEMORY_TYPE_HOST, 1);
    mem_buffer rbuf(UCS_MBYTE, UCS_MEMORY_TYPE_HOST);

    /* Send/receive from buffers to trigger adding them to registration cache */
    {
        static constexpr uint64_t TAG = 0xdeadbeef;
        ucp_request_param_t param;

        param.op_attr_mask = 0;
        auto sreq = ucp_tag_send_nbx(sender().ep(), sbuf.ptr(), sbuf.size(),
                                     TAG, &param);
        auto rreq = ucp_tag_recv_nbx(receiver().worker(), rbuf.ptr(),
                                     rbuf.size(), TAG, 0, &param);

        ASSERT_UCS_OK(requests_wait({sreq, rreq}));
    }

    /* Map the receive buffer for atomic operations */
    ucp_mem_h memh;
    ucp_rkey_h rkey;
    {
        ucp_mem_map_params_t params;
        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = rbuf.ptr();
        params.length     = rbuf.size();
        params.flags      = mem_map_flags();

        ASSERT_UCS_OK(ucp_mem_map(receiver().ucph(), &params, &memh));

        void *rkey_buffer;
        size_t rkey_size;
        ASSERT_UCS_OK(ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer,
                                    &rkey_size));
        ASSERT_UCS_OK(ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey));

        ucp_rkey_buffer_release(rkey_buffer);
    }

    /* Perform atomic operation */
    {
        uint64_t value = 1;
        ucp_request_param_t param;

        param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype     = ucp_dt_make_contig(sizeof(value));
        auto sreq = ucp_atomic_op_nbx(sender().ep(), UCP_ATOMIC_OP_ADD, &value,
                                      1, (uintptr_t)rbuf.ptr(), rkey, &param);

        param.op_attr_mask = 0;
        auto freq          = ucp_ep_flush_nbx(sender().ep(), &param);

        ASSERT_UCS_OK(requests_wait({sreq, freq}));
    }

    /* Unmap the buffer */
    ucp_rkey_destroy(rkey);
    ASSERT_UCS_OK(ucp_mem_unmap(receiver().ucph(), memh));
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_mmap_atomic)

class test_ucp_rkey_compare : public test_ucp_mmap {
public:
    void init() override
    {
        const size_t count = 5;
        test_ucp_mmap::init();

        for (int i = 0; i < count; i++) {
            m_chunks.emplace_back(new mem_chunk(sender().ucph()));
        }
    }

    void cleanup() override
    {
        m_chunks.clear();

        test_ucp_mmap::cleanup();
    }

protected:
    std::vector<std::unique_ptr<mem_chunk>> m_chunks;
};

UCS_TEST_P(test_ucp_rkey_compare, rkey_compare_errors)
{
    ucp_rkey_compare_params_t params = {};
    ucp_rkey_h rkey = m_chunks.front()->unpack(receiver().ep());
    ucs_status_t status;
    int result;

    scoped_log_handler err_handler(wrap_errors_logger);

    status = ucp_rkey_compare(receiver().worker(), rkey, rkey, &params, NULL);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    params.field_mask = 1;
    status = ucp_rkey_compare(receiver().worker(), rkey, rkey, &params,
                              &result);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_ucp_rkey_compare, rkey_compare_different_config)
{
    int result                       = 1;
    ucp_rkey_compare_params_t params = {};
    ucp_rkey_h rkey1, rkey2;
    ucp_md_map_t md_map;

    rkey1  = m_chunks.front()->unpack(receiver().ep());
    md_map = rkey1->md_map & (rkey1->md_map - 1);

    if (md_map == 0) {
        UCS_TEST_SKIP_R("cannot remove last memory domain");
    }

    rkey2 = m_chunks.front()->unpack(receiver().ep(), md_map);
    EXPECT_EQ(md_map, rkey2->md_map);

    EXPECT_UCS_OK(ucp_rkey_compare(receiver().worker(), rkey1, rkey2, &params,
                                   &result));
    EXPECT_NE(0, result);
}

UCS_TEST_P(test_ucp_rkey_compare, rkey_compare)
{
    ucp_rkey_compare_params_t params = {};
    ucp_worker_h worker              = receiver().worker();
    std::vector<ucp_rkey_h> rkeys;
    ucs_status_t status;
    int result;

    for (auto &c : m_chunks) {
        rkeys.push_back(c->unpack(receiver().ep()));
    }

    std::sort(rkeys.begin(), rkeys.end(),
              [worker](const ucp_rkey_h &rkey1, const ucp_rkey_h &rkey2) {
                  ucp_rkey_compare_params_t params = {};
                  int result;
                  ucs_status_t status = ucp_rkey_compare(worker, rkey1, rkey2,
                                                         &params, &result);
                  ASSERT_UCS_OK(status);
                  return result < 0;
              });

    for (int i = 0; i < rkeys.size(); i++) {
        for (int j = 0; j < rkeys.size(); j++) {
            status = ucp_rkey_compare(worker, rkeys[i], rkeys[j], &params,
                                      &result);
            ASSERT_UCS_OK(status);

            if (i < j) {
                ASSERT_GE(0, result);
            } else if (i > j) {
                ASSERT_LE(0, result);
            } else {
                ASSERT_EQ(0, result);
            }
        }
    }
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_rkey_compare)


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
    test_rereg(0, true);
}

UCS_TEST_P(test_ucp_mmap_export, alloc_reg_export_and_reimport)
{
    test_rereg(UCP_MEM_MAP_ALLOCATE, true);
}

UCS_TEST_P(test_ucp_mmap_export, export_import) {
    mem_chunk mem(sender().ucph());
    EXPECT_FALSE(mem.memh->flags & UCP_MEMH_FLAG_IMPORTED);

    ucp_mem_h imported_memh = import_memh(mem.memh);
    EXPECT_TRUE(imported_memh->flags & UCP_MEMH_FLAG_IMPORTED);

    ASSERT_UCS_OK(ucp_mem_unmap(receiver().ucph(), imported_memh));
}

UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(test_ucp_mmap_export)
