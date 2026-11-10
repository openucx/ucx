/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
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
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/dt/dt.h>
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

/* Disables UCM events and resets the global memory type cache, and restores
 * both on destruction, also when the test fails or is skipped in init() */
class memtype_explicit_guard {
public:
    memtype_explicit_guard() : m_enable_events(0), m_active(false) {
    }

    ~memtype_explicit_guard() {
        if (m_active) {
            reset_cache();
            ucm_global_opts.enable_events = m_enable_events;
        }
    }

    void enable() {
        m_enable_events               = ucm_global_opts.enable_events;
        m_active                      = true;
        ucm_global_opts.enable_events = 0;
        reset_cache();
    }

private:
    static void reset_cache() {
        ucs_memtype_cache_cleanup();
        ucs_memtype_cache_global_init();
    }

    int  m_enable_events;
    bool m_active;
};

/* Memory type cache without UCM events: only ranges mapped by ucp_mem_map() are
 * cached, and they are removed by ucp_mem_unmap() */
class test_ucp_memtype_explicit : public test_ucp_mem_type {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_memtypes(variants, get_test_variants_base,
                             UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                             UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED));
    }

    virtual void init() {
        modify_config("MEMTYPE_CACHE", "y");
        m_guard.enable();
        ucp_test::init();
    }

protected:
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
    memtype_explicit_guard m_guard;
};

UCS_TEST_P(test_ucp_memtype_explicit, map_unmap) {
    const size_t size = 4096;
    ucs_memory_info_t info;
    ucp_mem_h memh;

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
    const size_t size                  = 4096;
    const ucs_memory_type_t wrong_hint = UCS_MEMORY_TYPE_HOST;
    ucs_memory_info_t info;
    ucp_mem_h memh;
    ucs_status_t status;

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

/*
 * Stream-ordered CUDA memory is not GDR capable, so gdr_copy cannot
 * register it. Pack/unpack through the CUDA memory type endpoint has to fall
 * back from the preferred gdr_copy lane to cuda_copy.
 */
class test_ucp_mem_type_non_gdr_capable : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_AM);
    }

    virtual void init() override
    {
        /* Type the stream-ordered allocations as plain CUDA memory, so they
         * reach the CUDA memory type endpoint where gdr_copy is selected */
        modify_config("CUDA_COPY_ASYNC_MEM_TYPE", "cuda", SETENV_IF_NOT_EXIST);
        ucp_test::init();
    }

protected:
    static constexpr size_t BUF_SIZE = 65536;
    static constexpr uint64_t SEED   = 0xdeadbeefcafebabeull;

    /* The datatype iterator passes the resolved memory flags, while the
     * protocols which do not carry them pass NULL */
    void test_pack_unpack(int pass_mem_info)
    {
        std::vector<uint8_t> host_buf(BUF_SIZE);
        ucp_memory_info_t mem_info;
        ucp_ep_h mem_type_ep;

        if (!mem_buffer::is_async_supported(UCS_MEMORY_TYPE_CUDA)) {
            UCS_TEST_SKIP_R("CUDA async allocation is not supported");
        }

        mem_type_ep = sender().worker()->mem_type_ep[UCS_MEMORY_TYPE_CUDA];
        if (mem_type_ep == NULL) {
            UCS_TEST_SKIP_R("no CUDA mem type endpoint");
        }

        /* Without a preferred lane which demands memory flags, pack/unpack has
         * nothing to fall back from */
        ucp_lane_index_t lane = ucp_ep_config(mem_type_ep)->key.rma_lanes[0];
        if ((lane == UCP_NULL_LANE) ||
            (ucp_ep_md_attr(mem_type_ep, lane)->required_mem_flags == 0)) {
            UCS_TEST_SKIP_R("first CUDA mem type lane requires no memory flags");
        }

        mem_buffer src(BUF_SIZE, UCS_MEMORY_TYPE_CUDA,
                       mem_buffer::alloc_mode::ASYNC);
        mem_buffer dst(BUF_SIZE, UCS_MEMORY_TYPE_CUDA,
                       mem_buffer::alloc_mode::ASYNC);

        ucp_memory_detect(sender().ucph(), src.ptr(), BUF_SIZE, &mem_info);
        if (mem_info.flags & UCS_MEM_FLAG_GDR_CAPABLE) {
            UCS_TEST_SKIP_R("CUDA async memory is GDR capable");
        }

        ASSERT_EQ(UCS_MEMORY_TYPE_CUDA, mem_info.type);

        const ucp_memory_info_t *mem_info_arg = pass_mem_info ? &mem_info :
                                                                NULL;

        mem_buffer::pattern_fill(src.ptr(), BUF_SIZE, SEED,
                                 UCS_MEMORY_TYPE_CUDA);
        mem_buffer::pattern_fill(dst.ptr(), BUF_SIZE, 0, UCS_MEMORY_TYPE_CUDA);

        ucp_mem_type_pack(sender().worker(), host_buf.data(), src.ptr(),
                          BUF_SIZE, UCS_MEMORY_TYPE_CUDA, mem_info_arg);
        mem_buffer::pattern_check(host_buf.data(), BUF_SIZE, SEED);

        ucp_mem_type_unpack(sender().worker(), dst.ptr(), host_buf.data(),
                            BUF_SIZE, UCS_MEMORY_TYPE_CUDA, mem_info_arg);
        mem_buffer::pattern_check(dst.ptr(), BUF_SIZE, SEED,
                                  UCS_MEMORY_TYPE_CUDA);
    }
};

UCS_TEST_P(test_ucp_mem_type_non_gdr_capable, pack_unpack_resolved_flags)
{
    test_pack_unpack(1);
}

/* Without the flags from the caller, they are resolved from the buffer address
 * on the first lane whose memory domain requires them */
UCS_TEST_P(test_ucp_mem_type_non_gdr_capable, pack_unpack_unknown_flags)
{
    test_pack_unpack(0);
}

/* Resolving the flags hits the memtype cache first, make sure the slow path
 * also reports them when the cache is disabled */
UCS_TEST_P(test_ucp_mem_type_non_gdr_capable, pack_unpack_no_memtype_cache,
           "MEMTYPE_CACHE=n")
{
    test_pack_unpack(0);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_non_gdr_capable, all, "all")
