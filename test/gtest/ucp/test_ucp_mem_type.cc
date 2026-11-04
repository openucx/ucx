/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/mem_buffer.h>

extern "C" {
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


/*
 * ZE_COPY_DMABUF decides whether the network memory domains register Level Zero
 * memory through a DMA-BUF. uct_p2p_rma_test_dmabuf checks that the network
 * reads the memory correctly through it.
 */
class test_ucp_mem_type_dmabuf : public test_ucp_mem_type {
public:
    static constexpr uint64_t SEED = 0x1234567890abcdeflu;

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_memtypes(variants, get_test_variants_base,
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
    mem_buffer recv_buf(length, UCS_MEMORY_TYPE_HOST);
    ucs::handle<ucp_mem_h, ucp_context_h> memh_handle;
    ucs::auto_ptr<mem_buffer> app_buf;
    ucp_request_param_t param, recv_param;
    ucp_mem_map_params_t params;
    void *send_buf, *sreq, *rreq;
    ucp_mem_h memh;

    EXPECT_EQ(exp_dmabuf, (context->reg_md_map[mem_type()] &
                           context->dmabuf_reg_md_map) != 0)
            << mem_type_name << " memory registration through a DMA-BUF";

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

        EXPECT_EQ(exp_dmabuf, (memh->md_map & context->dmabuf_reg_md_map) != 0)
                << mem_type_name << " memory handle registration through a "
                << "DMA-BUF";
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
    if ((sender().ucph()->dmabuf_mds[mem_type()] == UCP_NULL_RESOURCE) ||
        (sender().ucph()->dmabuf_reg_md_map == 0)) {
        UCS_TEST_SKIP_R("DMA-BUF registration is not supported");
    }

    test_xfer(false, true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_dmabuf, rcx_ze, "rc_x,ze_copy")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_dmabuf, rcv_ze, "rc_v,ze_copy")


/*
 * With DMA-BUF export disabled, the network memory domains cannot register
 * Level Zero memory: device memory is staged through host memory, and
 * CPU-accessible memory is copied.
 */
class test_ucp_mem_type_dmabuf_off : public test_ucp_mem_type_dmabuf {
protected:
    void init() override
    {
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

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_dmabuf_off, rcx_ze,
                              "rc_x,ze_copy")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_mem_type_dmabuf_off, rcv_ze,
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
