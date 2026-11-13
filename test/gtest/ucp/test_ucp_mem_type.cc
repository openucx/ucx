/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
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
        return ucs_memtype_lookup(address, size, info);
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

/* Transfers from and to memory which is cached by ucp_mem_map(), where the
 * operations get no memory handle, so the protocols look the memory up */
class test_ucp_memtype_explicit_xfer : public test_ucp_memtype_explicit {
public:
    virtual void init() {
        test_ucp_memtype_explicit::init();
        sender().connect(&receiver(), get_ep_params());
    }

protected:
    static const uint64_t SEED = 0x2222222222222222lu;

    void send_recv(void *send_ptr, void *recv_ptr, size_t length) {
        ucp_request_param_t param;

        param.op_attr_mask = 0;
        mem_buffer::pattern_fill(send_ptr, length, SEED, mem_type());
        mem_buffer::pattern_fill(recv_ptr, length, 0, mem_type());

        void *rreq = ucp_tag_recv_nbx(receiver().worker(), recv_ptr, length, 1,
                                      1, &param);
        void *sreq = ucp_tag_send_nbx(sender().ep(), send_ptr, length, 1,
                                      &param);
        EXPECT_UCS_OK(request_wait(sreq));
        EXPECT_UCS_OK(request_wait(rreq));

        mem_buffer::pattern_check(recv_ptr, length, SEED, mem_type());
    }
};

/* Send a rendezvous size message from an offset inside a mapped buffer, without
 * passing the memory handle, and then send again after the buffers are
 * unmapped, which falls back to the memory type detection */
UCS_TEST_P(test_ucp_memtype_explicit_xfer, tag_send_from_mapped_buffer,
           "RNDV_THRESH=0")
{
    static const size_t size   = 256 * UCS_KBYTE;
    static const size_t offset = 4 * UCS_KBYTE;
    const size_t length        = size - (2 * offset);
    ucs_memory_info_t info;
    ucp_mem_h send_memh, recv_memh;

    mem_buffer send_buf(size, mem_type());
    mem_buffer recv_buf(size, mem_type());
    void *send_ptr = UCS_PTR_BYTE_OFFSET(send_buf.ptr(), offset);
    void *recv_ptr = UCS_PTR_BYTE_OFFSET(recv_buf.ptr(), offset);

    ASSERT_UCS_OK(map(send_buf.ptr(), size, &send_memh));
    ASSERT_UCS_OK(map(recv_buf.ptr(), size, &recv_memh));

    /* The lookup of a range inside the buffer reports the mapped range */
    ASSERT_UCS_OK(lookup(send_ptr, length, &info));
    EXPECT_EQ(mem_type(), info.type);
    EXPECT_EQ(send_buf.ptr(), info.base_address);
    EXPECT_EQ(size, info.alloc_length);

    send_recv(send_ptr, recv_ptr, length);

    unmap(send_memh);
    unmap(recv_memh);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, lookup(send_ptr, length, &info));

    send_recv(send_ptr, recv_ptr, length);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_memtype_explicit_xfer, all, "all")

/*
 * The CUDA_COPY_DMABUF, ROCM_COPY_DMABUF and ZE_COPY_DMABUF settings decide
 * whether the GPU memory domains provide a DMA-BUF that the network memory
 * domains register. uct_p2p_rma_test_dmabuf checks that the network reads the
 * memory correctly through it.
 */
class test_ucp_mem_type_dmabuf : public test_ucp_mem_type {
public:
    static constexpr uint64_t SEED = 0x1234567890abcdeflu;

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_memtypes(variants, get_test_variants_base,
                             UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                             UCS_BIT(UCS_MEMORY_TYPE_ROCM) |
                             UCS_BIT(UCS_MEMORY_TYPE_ZE_HOST) |
                             UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE));
    }

protected:
    void init() override
    {
        test_ucp_mem_type::init();
        sender().connect(&receiver(), get_ep_params());
    }

    /*
     * Send memory of the tested type to a host buffer over a rendezvous
     * protocol. 'app_owned' allocates the source without UCP and sends it
     * without a memory handle, like an application would.
     */
    void test_xfer(bool app_owned, bool exp_dmabuf);
};

void test_ucp_mem_type_dmabuf::test_xfer(bool app_owned, bool exp_dmabuf)
{
    static constexpr size_t length = 4 * UCS_MBYTE;
    ucp_context_h context          = sender().ucph();
    const char *mem_type_name      = ucs_memory_type_names[mem_type()];
    /* The network memory domains register Level Zero memory only through a
     * DMA-BUF, and CUDA or ROCm memory also through peer memory */
    bool dmabuf_only               = (mem_type() == UCS_MEMORY_TYPE_ZE_HOST) ||
                                     (mem_type() == UCS_MEMORY_TYPE_ZE_DEVICE);
    mem_buffer recv_buf(length, UCS_MEMORY_TYPE_HOST);
    ucs::handle<ucp_mem_h, ucp_context_h> memh_handle;
    ucs::auto_ptr<mem_buffer> app_buf;
    ucp_request_param_t param, recv_param;
    ucp_mem_map_params_t params;
    void *send_buf, *sreq, *rreq;
    ucp_mem_h memh;

    EXPECT_EQ(exp_dmabuf, context->dmabuf_mds[mem_type()] != UCP_NULL_RESOURCE)
            << mem_type_name << " memory DMA-BUF provider";
    if (dmabuf_only) {
        EXPECT_EQ(exp_dmabuf, (context->reg_md_map[mem_type()] &
                               context->dmabuf_reg_md_map) != 0)
                << mem_type_name << " memory registration through a DMA-BUF";
    }

    if (app_owned) {
        app_buf.reset(new mem_buffer(length, mem_type()));
        send_buf           = app_buf->ptr();
        param.op_attr_mask = 0;
    } else {
        params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        params.length      = length;
        params.flags       = UCP_MEM_MAP_ALLOCATE;
        params.memory_type = mem_type();
        ASSERT_UCS_OK(ucp_mem_map(context, &params, &memh));
        memh_handle.reset(memh, [](ucp_mem_h handle, ucp_context_h ctx) {
            EXPECT_UCS_OK(ucp_mem_unmap(ctx, handle));
        }, context);

        if (exp_dmabuf || dmabuf_only) {
            EXPECT_EQ(exp_dmabuf,
                      (memh->md_map & context->dmabuf_reg_md_map) != 0)
                    << mem_type_name << " memory handle registration on a "
                    << "DMA-BUF capable memory domain";
        }

        send_buf           = ucp_memh_address(memh);
        param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        param.memh         = memh;
    }

    mem_buffer::pattern_fill(send_buf, length, SEED, mem_type());
    mem_buffer::pattern_fill(recv_buf.ptr(), length, 0, UCS_MEMORY_TYPE_HOST);

    recv_param.op_attr_mask = 0;

    rreq = ucp_tag_recv_nbx(receiver().worker(), recv_buf.ptr(), length, 1, 1,
                            &recv_param);
    sreq = ucp_tag_send_nbx(sender().ep(), send_buf, length, 1, &param);
    EXPECT_UCS_OK(request_wait(sreq));
    EXPECT_UCS_OK(request_wait(rreq));

    mem_buffer::pattern_check(recv_buf.ptr(), length, SEED,
                              UCS_MEMORY_TYPE_HOST);
}

UCS_TEST_P(test_ucp_mem_type_dmabuf, alloc, "RNDV_THRESH=0")
{
    ucp_context_h context   = sender().ucph();
    ucp_md_index_t md_index = context->dmabuf_mds[mem_type()];
    bool exported;

    if ((md_index == UCP_NULL_RESOURCE) || (context->dmabuf_reg_md_map == 0)) {
        UCS_TEST_SKIP_R("DMA-BUF registration is not supported");
    }

    /* Only an MPS MLOPart device may refuse an export the memory domain
     * advertises, and the network cannot register the memory then */
    if ((mem_type() == UCS_MEMORY_TYPE_CUDA) &&
        !mem_buffer::cuda_mem_is_gdr_capable()) {
        ASSERT_UCS_OK(mem_buffer::query_dmabuf_export(
                context->tl_mds[md_index].md, mem_type(), &exported));
        if (!exported) {
            UCS_TEST_SKIP_R("the device refuses DMA-BUF export");
        }
    }

    test_xfer(false, true);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_mem_type_dmabuf, rcx,
                                        "rc_x,ze_copy")
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_mem_type_dmabuf, rcv,
                                        "rc_v,ze_copy")


/*
 * With DMA-BUF export disabled, the network memory domains cannot register
 * Level Zero memory: device memory is staged through host memory, and
 * CPU-accessible memory is copied. CUDA and ROCm memory may still be
 * registered through peer memory.
 */
class test_ucp_mem_type_dmabuf_off : public test_ucp_mem_type_dmabuf {
protected:
    void init() override
    {
        modify_config("CUDA_COPY_DMABUF", "n", SETENV_IF_NOT_EXIST);
        modify_config("ROCM_COPY_DMABUF", "n", SETENV_IF_NOT_EXIST);
        modify_config("ZE_COPY_DMABUF", "n", SETENV_IF_NOT_EXIST);
        test_ucp_mem_type_dmabuf::init();
    }
};

UCS_TEST_P(test_ucp_mem_type_dmabuf_off, alloc, "RNDV_THRESH=0",
           "RNDV_FRAG_MEM_TYPES=host")
{
    test_xfer(false, false);
}

UCS_TEST_P(test_ucp_mem_type_dmabuf_off, app_mem, "RNDV_THRESH=0",
           "RNDV_FRAG_MEM_TYPES=host")
{
    test_xfer(true, false);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_mem_type_dmabuf_off, rcx,
                                        "rc_x,ze_copy")
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_mem_type_dmabuf_off, rcv,
                                        "rc_v,ze_copy")

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
