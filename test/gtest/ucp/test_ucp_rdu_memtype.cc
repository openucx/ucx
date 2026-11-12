/**
 * Copyright (C) SambaNova Systems, 2025-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/mem_buffer.h>
#include <common/test.h>

extern "C" {
#include <ucp/api/ucp.h>
#include <ucs/memory/memtype_cache.h>
}


/*
 * Memory type detection of RDU buffers mapped without a memory type, whether
 * they were allocated before or after UCP created the memory type cache.
 */
class test_ucp_rdu_memtype : public ucs::test {
protected:
    static const size_t BUF_LEN = UCS_MBYTE;

    virtual void init()
    {
        ucs::test::init();

        if (!mem_buffer::is_rdu_supported()) {
            UCS_TEST_SKIP_R("RDU memory is not supported");
        }
    }

    virtual void cleanup()
    {
        if (m_context != NULL) {
            ucp_cleanup(m_context);
        }
        ucs::test::cleanup();
    }

    void create_context()
    {
        ucp_params_t params;
        ucp_config_t *config;
        ucs_status_t status;

        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features   = UCP_FEATURE_RMA;

        ASSERT_UCS_OK(ucp_config_read(NULL, NULL, &config));
        /* Only memory type detection is tested, so do not use network
         * transports, which may be unable to register RDU memory */
        ASSERT_UCS_OK(ucp_config_modify(config, "TLS", "self,rdu"));
        status = ucp_init(&params, config, &m_context);
        ucp_config_release(config);
        ASSERT_UCS_OK(status);
    }

    /* Map a buffer without a memory type, and return the detected type */
    ucs_memory_type_t map_detect(void *buffer, size_t length)
    {
        ucp_mem_map_params_t params;
        ucp_mem_attr_t attr;
        ucp_mem_h memh;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.address    = buffer;
        params.length     = length;
        EXPECT_UCS_OK(ucp_mem_map(m_context, &params, &memh));
        if (HasFailure()) {
            return UCS_MEMORY_TYPE_UNKNOWN;
        }

        attr.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
        EXPECT_UCS_OK(ucp_mem_query(memh, &attr));
        EXPECT_UCS_OK(ucp_mem_unmap(m_context, memh));
        return attr.mem_type;
    }

    /* Make sure the memory type cache exists and was looked up */
    void lookup_host_buffer()
    {
        std::vector<char> host_buffer(BUF_LEN);

        EXPECT_EQ(UCS_MEMORY_TYPE_HOST,
                  map_detect(host_buffer.data(), host_buffer.size()));
    }

    void test_alloc_after_init(mem_buffer::rdu_mem_kind kind)
    {
        create_context();
        lookup_host_buffer();

        void *buffer = mem_buffer::allocate_rdu(BUF_LEN, kind);
        EXPECT_EQ(UCS_MEMORY_TYPE_RDU, map_detect(buffer, BUF_LEN));
        /* Part of the buffer */
        EXPECT_EQ(UCS_MEMORY_TYPE_RDU,
                  map_detect(UCS_PTR_BYTE_OFFSET(buffer, BUF_LEN / 2),
                             BUF_LEN / 4));
        mem_buffer::release(buffer, UCS_MEMORY_TYPE_RDU);
    }

    ucp_context_h m_context = NULL;
};

UCS_TEST_F(test_ucp_rdu_memtype, alloc_after_init_hbm) {
    test_alloc_after_init(mem_buffer::rdu_mem_kind::HBM);
}

UCS_TEST_F(test_ucp_rdu_memtype, alloc_after_init_ddr) {
    test_alloc_after_init(mem_buffer::rdu_mem_kind::DDR);
}

UCS_TEST_F(test_ucp_rdu_memtype, alloc_before_init) {
    void *buffer = mem_buffer::allocate_rdu(BUF_LEN,
                                            mem_buffer::rdu_mem_kind::HBM);

    create_context();
    lookup_host_buffer();
    EXPECT_EQ(UCS_MEMORY_TYPE_RDU, map_detect(buffer, BUF_LEN));

    mem_buffer::release(buffer, UCS_MEMORY_TYPE_RDU);
}

/* The runtime reuses the address range of a released buffer, so a buffer
 * allocated after another one was released must still be detected as RDU */
UCS_TEST_F(test_ucp_rdu_memtype, realloc) {
    create_context();
    lookup_host_buffer();

    for (int i = 0; i < 4; ++i) {
        void *buffer = mem_buffer::allocate_rdu(BUF_LEN,
                                                mem_buffer::rdu_mem_kind::HBM);
        EXPECT_EQ(UCS_MEMORY_TYPE_RDU, map_detect(buffer, BUF_LEN));
        mem_buffer::release(buffer, UCS_MEMORY_TYPE_RDU);
    }
}

/* A released buffer must not be reported as RDU memory */
UCS_TEST_F(test_ucp_rdu_memtype, free) {
    ucs_memory_info_t mem_info;

    create_context();
    lookup_host_buffer();

    void *buffer = mem_buffer::allocate_rdu(BUF_LEN,
                                            mem_buffer::rdu_mem_kind::HBM);
    EXPECT_EQ(UCS_MEMORY_TYPE_RDU, map_detect(buffer, BUF_LEN));
    mem_buffer::release(buffer, UCS_MEMORY_TYPE_RDU);

    if (ucs_memtype_cache_lookup(buffer, BUF_LEN, &mem_info) == UCS_OK) {
        EXPECT_NE(UCS_MEMORY_TYPE_RDU, mem_info.type);
    }
}

/* The first detection of a buffer marks its memory type cache entry, which the
 * UCM hooks add as unknown memory, as RDU memory with the RDU's system device,
 * so later lookups don't query the memory domain again */
UCS_TEST_F(test_ucp_rdu_memtype, detect_caches_rdu) {
    ucs_memory_info_t mem_info;

    create_context();
    lookup_host_buffer();

    mem_buffer mem_buf(BUF_LEN, UCS_MEMORY_TYPE_RDU);
    EXPECT_EQ(UCS_MEMORY_TYPE_RDU, map_detect(mem_buf.ptr(), BUF_LEN));
    ASSERT_UCS_OK(ucs_memtype_cache_lookup(mem_buf.ptr(), BUF_LEN, &mem_info));
    EXPECT_EQ(UCS_MEMORY_TYPE_RDU, mem_info.type)
            << "memory type " << ucs_memory_type_names[mem_info.type];
    EXPECT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, mem_info.sys_dev);
}
