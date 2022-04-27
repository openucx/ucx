/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
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
    enum {
        VARIANT_DEFAULT,
        VARIANT_MAP_NONBLOCK,
        VARIANT_PROTO_ENABLE,
        VARIANT_NO_RCACHE
    };

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_RMA, VARIANT_DEFAULT, "");
        add_variant_with_value(variants, UCP_FEATURE_RMA, VARIANT_MAP_NONBLOCK,
                               "map_nb");
        add_variant_with_value(variants, UCP_FEATURE_RMA, VARIANT_PROTO_ENABLE,
                               "proto");
        add_variant_with_value(variants, UCP_FEATURE_RMA, VARIANT_NO_RCACHE,
                               "no_rcache");
    }

    virtual void init() {
        ucs::skip_on_address_sanitizer();
        if (enable_proto()) {
            modify_config("PROTO_ENABLE", "y");
        }
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }

        if (get_variant_value() == VARIANT_NO_RCACHE) {
            modify_config("RCACHE_ENABLE", "n");
        }
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

protected:
    bool resolve_rma(entity *e, ucp_rkey_h rkey);
    bool resolve_amo(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw_get_zcopy(entity *e, ucp_rkey_h rkey);
    bool resolve_rma_bw_put_zcopy(entity *e, ucp_rkey_h rkey);
    void test_length0(unsigned flags);
    void test_rkey_management(ucp_mem_h memh, bool is_dummy,
                              bool expect_rma_offload);
    bool enable_proto() const;

private:
    void expect_same_distance(const ucs_sys_dev_distance_t &dist1,
                              const ucs_sys_dev_distance_t &dist2);
    void test_rkey_proto(ucp_mem_h memh);
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
    if (enable_proto()) {
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
        void *ptr = ucs::mmap_fixed_address();

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
