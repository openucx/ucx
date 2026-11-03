/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/mem_buffer.h>

extern "C" {
#include <ucm/event/event.h>
#include <ucs/config/global_opts.h>
#include <ucs/memory/memtype_cache.h>
#include <uct/api/uct.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
}


class test_ucp_mem_type : public ucp_test {
public:
    static void get_test_variants_base(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG);
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_memtypes(variants, get_test_variants_base);
    }

protected:
    ucs_memory_type_t mem_type() const {
        return static_cast<ucs_memory_type_t>(get_variant_value());
    }
};

UCS_TEST_P(test_ucp_mem_type, detect) {

    const size_t size                      = 256;
    const ucs_memory_type_t alloc_mem_type = mem_type();
    ucp_memory_info_t mem_info;

    mem_buffer b(size, alloc_mem_type);

    ucp_memory_detect(sender().ucph(), b.ptr(), size, &mem_info);
    EXPECT_EQ(alloc_mem_type, mem_info.type);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type, all, "all")

class test_ucp_mem_type_alloc_before_init : public test_ucp_mem_type {
public:
    test_ucp_mem_type_alloc_before_init() {
        m_size = 10000;
    }

    virtual void init() {
        m_send_buffer.reset(new mem_buffer(m_size, mem_type()));
        m_recv_buffer.reset(new mem_buffer(m_size, mem_type()));
        test_ucp_mem_type::init();
    }

    virtual void cleanup() {
        test_ucp_mem_type::cleanup();
        m_send_buffer.reset();
        m_recv_buffer.reset();
    }

    static const uint64_t SEED = 0x1111111111111111lu;
protected:
    size_t                     m_size;
    ucs::auto_ptr<mem_buffer>  m_send_buffer, m_recv_buffer;
};

UCS_TEST_P(test_ucp_mem_type_alloc_before_init, xfer) {
    sender().connect(&receiver(), get_ep_params());

    ucp_memory_info_t mem_info;
    ucp_memory_detect(sender().ucph(), m_send_buffer->ptr(), m_size, &mem_info);
    EXPECT_EQ(mem_type(), mem_info.type) << "send buffer";
    ucp_memory_detect(receiver().ucph(), m_recv_buffer->ptr(), m_size,
                      &mem_info);
    EXPECT_EQ(mem_type(), mem_info.type) << "receive buffer";

    mem_buffer::pattern_fill(m_send_buffer->ptr(), m_size, SEED, mem_type());

    for (int i = 0; i < 3; ++i) {
        mem_buffer::pattern_fill(m_recv_buffer->ptr(), m_size, 0, mem_type());

        void *sreq = ucp_tag_send_nb(sender().ep(), m_send_buffer->ptr(), m_size,
                                     ucp_dt_make_contig(1), 1,
                                     (ucp_send_callback_t)ucs_empty_function);
        void *rreq = ucp_tag_recv_nb(receiver().worker(), m_recv_buffer->ptr(),
                                     m_size, ucp_dt_make_contig(1), 1, 1,
                                     (ucp_tag_recv_callback_t)ucs_empty_function);
        request_wait(sreq);
        request_wait(rreq);

        mem_buffer::pattern_check(m_recv_buffer->ptr(), m_size, SEED, mem_type());
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_alloc_before_init, all, "all")

/* Memory type cache without UCM events: only ranges mapped by ucp_mem_map() are
 * cached, and they are removed by ucp_mem_unmap() */
class test_ucp_memtype_explicit : public ucp_test {
public:
    static void get_test_variants_base(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_TAG);
    }

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_memtypes(variants, get_test_variants_base);
    }

    virtual void init() {
        m_enable_events        = ucm_global_opts.enable_events;
        m_enable_memtype_cache = ucs_global_opts.enable_memtype_cache;

        ucm_global_opts.enable_events        = 0;
        ucs_global_opts.enable_memtype_cache = UCS_YES;
        ucs_memtype_cache_cleanup();
        ucs_memtype_cache_global_init();
        ucp_test::init();
    }

    virtual void cleanup() {
        ucp_test::cleanup();
        ucs_memtype_cache_cleanup();
        ucs_memtype_cache_global_init();
        ucm_global_opts.enable_events        = m_enable_events;
        ucs_global_opts.enable_memtype_cache = m_enable_memtype_cache;
    }

protected:
    ucs_memory_type_t mem_type() const {
        return static_cast<ucs_memory_type_t>(get_variant_value());
    }

    void check_mem_type() const {
        if (mem_type() == UCS_MEMORY_TYPE_HOST) {
            UCS_TEST_SKIP_R("host memory is never cached");
        }
    }

    ucs_status_t lookup(const void *address, size_t size,
                        ucs_memory_info_t *info) const {
        return ucs_memtype_cache_lookup(address, size, info);
    }

    ucs_status_t map(void *address, size_t size, ucp_mem_h *memh_p,
                     const ucs_memory_type_t *hint = NULL) {
        ucp_mem_map_params_t params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.address    = address;
        params.length     = size;
        if (hint != NULL) {
            params.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            params.memory_type = *hint;
        }

        return ucp_mem_map(sender().ucph(), &params, memh_p);
    }

    void unmap(ucp_mem_h memh) {
        ASSERT_UCS_OK(ucp_mem_unmap(sender().ucph(), memh));
    }

private:
    int          m_enable_events;
    ucs_ternary_auto_value_t m_enable_memtype_cache;
};

UCS_TEST_P(test_ucp_memtype_explicit, map_unmap) {
    const size_t size = 4096;
    ucs_memory_info_t info;
    ucp_mem_h memh;

    check_mem_type();
    mem_buffer b(size, mem_type());

    /* Memory which was not mapped is not known to the cache */
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, lookup(b.ptr(), size, &info));

    ASSERT_UCS_OK(map(b.ptr(), size, &memh));
    ASSERT_UCS_OK(lookup(b.ptr(), size, &info));
    EXPECT_EQ(mem_type(), info.type);

    unmap(memh);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, lookup(b.ptr(), size, &info));
}

UCS_TEST_P(test_ucp_memtype_explicit, repeated_map) {
    const size_t size = 4096;
    ucs_memory_info_t info;
    ucp_mem_h memh1, memh2;

    check_mem_type();
    mem_buffer b(size, mem_type());

    ASSERT_UCS_OK(map(b.ptr(), size, &memh1));
    ASSERT_UCS_OK(map(b.ptr(), size, &memh2));

    /* The range stays cached until the last mapping is released */
    unmap(memh1);
    ASSERT_UCS_OK(lookup(b.ptr(), size, &info));
    EXPECT_EQ(mem_type(), info.type);

    unmap(memh2);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, lookup(b.ptr(), size, &info));
}

UCS_TEST_P(test_ucp_memtype_explicit, map_with_hint) {
    const size_t size                = 4096;
    const ucs_memory_type_t wrong_hint = UCS_MEMORY_TYPE_HOST;
    ucs_memory_info_t info;
    ucp_mem_h memh;
    ucs_status_t status;

    check_mem_type();
    mem_buffer b(size, mem_type());

    status = map(b.ptr(), size, &memh, &wrong_hint);
    if (status != UCS_OK) {
        UCS_TEST_SKIP_R("mapping with a different memory type hint failed");
    }

    /* The cache holds the detected type, not the user-provided hint */
    ASSERT_UCS_OK(lookup(b.ptr(), size, &info));
    EXPECT_EQ(mem_type(), info.type);

    unmap(memh);
}

UCS_TEST_P(test_ucp_memtype_explicit, map_allocate) {
    const size_t size = 4096;
    ucp_mem_map_params_t params;
    ucp_mem_attr_t attr;
    ucs_memory_info_t info;
    ucp_mem_h memh;

    check_mem_type();

    params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS     |
                         UCP_MEM_MAP_PARAM_FIELD_LENGTH      |
                         UCP_MEM_MAP_PARAM_FIELD_FLAGS       |
                         UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    params.address     = NULL;
    params.length      = size;
    params.flags       = UCP_MEM_MAP_ALLOCATE;
    params.memory_type = mem_type();
    ASSERT_UCS_OK(ucp_mem_map(sender().ucph(), &params, &memh));

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    ASSERT_UCS_OK(ucp_mem_query(memh, &attr));

    ASSERT_UCS_OK(lookup(attr.address, size, &info));
    EXPECT_EQ(mem_type(), info.type);

    unmap(memh);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, lookup(attr.address, size, &info));
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_memtype_explicit, self_cuda,
                              "self,cuda_copy")

class test_ucp_cuda : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_AM);
    }
};

UCS_TEST_P(test_ucp_cuda, sparse_regions) {
    const ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_CUDA;
    const size_t size = 4096;
    const size_t count = 5;
    ucs_status_t status;
    void *ptr[count];
    ucp_mem_h memh[count];

    if (!mem_buffer::is_mem_type_supported(mem_type)) {
        UCS_TEST_SKIP_R("CUDA is not supported");
    }

    /* create contiguous CUDA registrations list */
    for (int i = 0; i < count; i++) {
        ptr[i] = mem_buffer::allocate(size, mem_type);

        if ((i > 0) && (UCS_PTR_BYTE_OFFSET(ptr[i - 1], size) != ptr[i])) {
            for (int j = 0; j < i; j++) {
                mem_buffer::release(ptr[j], mem_type);
            }
            UCS_TEST_SKIP_R("failed to create contiguous CUDA registrations list");
        }
    }

    /* make CUDA registrations list sparse */
    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            mem_buffer::release(ptr[i], mem_type);
        }
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            continue;
        }

        ucp_mem_map_params_t params;
        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        params.address    = ptr[i];
        params.length     = size;

        status = ucp_mem_map(sender().ucph(), &params, &memh[i]);
        ASSERT_UCS_OK(status);
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            continue;
        }

        void* rkey_buffer;
        size_t rkey_buffer_size;
        status = ucp_rkey_pack(sender().ucph(), memh[i], &rkey_buffer,
                               &rkey_buffer_size);
        ASSERT_UCS_OK(status);
        ucp_rkey_buffer_release(rkey_buffer);
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 0) {
            continue;
        }

        status = ucp_mem_unmap(sender().ucph(), memh[i]);
        ASSERT_UCS_OK(status);
    }

    for (int i = 0; i < count; i++) {
        if ((i & 1) == 1) {
            mem_buffer::release(ptr[i], mem_type);
        }
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_cuda, all, "all")
