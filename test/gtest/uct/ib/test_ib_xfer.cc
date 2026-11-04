/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/sys/ptr_arith.h>
#include <uct/test_p2p_rma.h>
#include <uct/test_p2p_mix.h>

#include <uct/ib/base/ib_md.h>
#ifdef HAVE_MLX5_DV
#include <uct/ib/mlx5/ib_mlx5.h>
#endif


class uct_p2p_rma_test_xfer : public uct_p2p_rma_test {};

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_xfer, fence_relaxed_order,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY),
                     "IB_PCI_RELAXED_ORDERING=try") {
    size_t size = ucs_min(ucs_get_page_size(),
                          sender().iface_attr().cap.put.max_bcopy);

    mapped_buffer sendbuf(size, 0, sender());
    mapped_buffer recvbuf(size, 0, receiver(), 0, UCS_MEMORY_TYPE_HOST,
                          UCT_MD_MEM_ACCESS_RMA);

    blocking_send(static_cast<send_func_t>(&uct_p2p_rma_test::put_bcopy),
                  sender_ep(), sendbuf, recvbuf, true);

    uct_ep_fence(sender_ep(), 0);

    blocking_send(static_cast<send_func_t>(&uct_p2p_rma_test::put_bcopy),
                  sender_ep(), sendbuf, recvbuf, true);

    flush();
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_rma_test_xfer)

class uct_p2p_rma_test_inlresp : public uct_p2p_rma_test {};

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_inlresp, get_bcopy_inlresp0,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY),
                     "IB_TX_INLINE_RESP=0") {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_inlresp, get_bcopy_inlresp64,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY),
                     "IB_TX_INLINE_RESP=64") {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_inlresp, get_zcopy_inlresp0,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY),
                     "IB_TX_INLINE_RESP=0") {
    EXPECT_EQ(has_transport("srd") ? 0u : 1u,
              sender().iface_attr().cap.get.min_zcopy);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                    sender().iface_attr().cap.get.min_zcopy,
                    sender().iface_attr().cap.get.max_zcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

#if HAVE_DEVX
/* test mlx5dv_create_qp() */
UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_inlresp, get_zcopy_inlresp0_devx_no,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY),
                     "IB_TX_INLINE_RESP=0", "IB_MLX5_DEVX=n") {
    EXPECT_EQ(1u, sender().iface_attr().cap.get.min_zcopy);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                    sender().iface_attr().cap.get.min_zcopy,
                    sender().iface_attr().cap.get.max_zcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}
#endif

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_inlresp, get_zcopy_inlresp64,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY),
                     "IB_TX_INLINE_RESP=64") {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                    sender().iface_attr().cap.get.min_zcopy,
                    sender().iface_attr().cap.get.max_zcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_rma_test_inlresp)


class uct_p2p_rma_test_alloc_methods : public uct_p2p_rma_test {
protected:
    void test_put_zcopy() {
        test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                        sender().iface_attr().cap.put.min_zcopy,
                        sender().iface_attr().cap.put.max_zcopy,
                        TEST_UCT_FLAG_SEND_ZCOPY);
    }

    void test_get_zcopy() {
        test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                        sender().iface_attr().cap.get.min_zcopy,
                        sender().iface_attr().cap.get.max_zcopy,
                        TEST_UCT_FLAG_RECV_ZCOPY);
    }
};

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_alloc_methods, xfer_reg,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_GET_ZCOPY))
{
    test_put_zcopy();
    test_get_zcopy();
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_alloc_methods, xfer_reg_multithreaded,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_GET_ZCOPY),
                     "*REG_MT_THRESH=1", "*REG_MT_CHUNK=1G", "*REG_MT_BIND=y")
{
    test_put_zcopy();
    test_get_zcopy();
}

UCT_INSTANTIATE_IB_AND_GGA_TEST_CASE(uct_p2p_rma_test_alloc_methods)

class uct_p2p_mix_test_alloc_methods : public uct_p2p_mix_test {};

UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000)
{
    run(1000);
}

UCT_INSTANTIATE_IB_AND_GGA_TEST_CASE(uct_p2p_mix_test_alloc_methods)


class uct_p2p_mix_test_mt : public uct_p2p_mix_test {
protected:
    mapped_buffer alloc_buffer(const entity &entity, size_t offset) override
    {
        mapped_buffer buf = uct_p2p_mix_test::alloc_buffer(entity, offset);
        auto *ib_memh     = static_cast<uct_ib_mem_t*>(buf.memh());
        EXPECT_TRUE(ib_memh->flags & UCT_IB_MEM_MULTITHREADED);
        return buf;
    }

    bool check_md_flags()
    {
#if HAVE_DEVX
        auto *ib_md = ucs_derived_of(sender().md(), uct_ib_md_t);
        if (strcmp(ib_md->name, UCT_IB_MD_NAME(mlx5))) {
            return false;
        }

        auto *ib_mlx5_md = ucs_derived_of(sender().md(), uct_ib_mlx5_md_t);
        return (ib_mlx5_md->flags & UCT_IB_MLX5_MD_FLAG_KSM) &&
               (ib_mlx5_md->flags & UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS);
#else
        return false;
#endif
    }

    virtual void init() override
    {
        push_config();
        modify_config("*REG_MT_THRESH", ucs::to_string(reg_mt_chunk + 1));
        modify_config("*REG_MT_CHUNK", ucs::to_string(reg_mt_chunk));

        uct_p2p_mix_test::init();

        if (!check_md_flags()) {
            UCS_TEST_SKIP_R("KSM and indirect atomics are required for MT "
                            "registration");
        }

        /* Too many chunks causes MT registration failure since DEVX
         * input structure became too big */
        m_buffer_size = ucs_min(m_buffer_size, 256 * reg_mt_chunk);
        /* We need at least two chunks */
        m_buffer_size = ucs_max(m_buffer_size, reg_mt_chunk + 1);
    }

    virtual void cleanup() override
    {
        uct_p2p_mix_test::cleanup();
        pop_config();
    }

    constexpr static size_t reg_mt_chunk = 16 * UCS_KBYTE;
};

constexpr size_t uct_p2p_mix_test_mt::reg_mt_chunk;

UCS_TEST_P(uct_p2p_mix_test_mt, mix1000_alloc_methods, "*REG_MT_BIND=y")
{
    run(1000);
}

UCS_TEST_P(uct_p2p_mix_test_mt, mix1000)
{
    run(1000);
}

UCS_TEST_P(uct_p2p_mix_test_mt, mix1000_last_byte_offset)
{
    /* Alloc 2 chunks buffer, but perform the operations on the last 8 bytes */
    run(1000, (reg_mt_chunk * 2) - 8, 8);
}

UCT_INSTANTIATE_IB_AND_GGA_TEST_CASE(uct_p2p_mix_test_mt)


class uct_p2p_mix_test_indirect_atomic : public uct_p2p_mix_test {};

UCS_TEST_P(uct_p2p_mix_test_indirect_atomic, mix1000_indirect_atomic,
           "IB_INDIRECT_ATOMIC=n")
{
    run(1000);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_mix_test_indirect_atomic)



/*
 * Moves memory allocated by a DMA-BUF provider, such as a GPU copy memory
 * domain, which the IB memory domain registers through the DMA-BUF. Nothing
 * falls back at UCT, so a correct payload shows that the NIC accessed the
 * memory through the DMA-BUF.
 */
class uct_p2p_rma_test_dmabuf : public uct_p2p_rma_test {
protected:
    /* Small enough for a driver to sub-allocate from a memory pool */
    static constexpr size_t LENGTH = 64 * UCS_KBYTE;

    struct provider_mem : uct_allocated_memory_t {
        provider_mem(uct_md_h md, ucs_memory_type_t mem_type)
        {
            uct_alloc_method_t method     = UCT_ALLOC_METHOD_MD;
            uct_mem_alloc_params_t params = {};

            params.field_mask = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS |
                                UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                                UCT_MEM_ALLOC_PARAM_FIELD_MDS |
                                UCT_MEM_ALLOC_PARAM_FIELD_NAME;
            params.flags      = UCT_MD_MEM_ACCESS_ALL;
            params.mem_type   = mem_type;
            params.mds.mds    = &md;
            params.mds.count  = 1;
            params.name       = "dmabuf_test";
            ASSERT_UCS_OK(uct_mem_alloc(LENGTH, &method, 1, &params, this));
        }

        ~provider_mem()
        {
            uct_mem_free(this);
        }
    };

    void test_dmabuf_xfer(uct_md_h provider_md, ucs_memory_type_t mem_type,
                          bool is_get);

    void test_dmabuf_xfer(bool is_get);
};

void uct_p2p_rma_test_dmabuf::test_dmabuf_xfer(uct_md_h provider_md,
                                               ucs_memory_type_t mem_type,
                                               bool is_get)
{
    /* The NIC accesses the sender's memory for PUT, the receiver's for GET */
    entity &mem_entity                 = is_get ? receiver() : sender();
    uct_md_mem_attr_v2_t mem_attr      = {};
    uct_md_mem_reg_params_t reg_params = {};
    uct_allocated_memory_t reg_mem;
    uct_rkey_bundle_t rkey             = {};
    uct_mem_h uct_memh;
    ucs_status_t status;

    UCS_TEST_MESSAGE << ucs_memory_type_names[mem_type] << " memory";

    /* Keep another buffer of the same size allocated, so that a driver which
     * sub-allocates both from one memory pool places the transferred buffer at
     * a nonzero offset in the pool */
    provider_mem pad(provider_md, mem_type);
    provider_mem mem(provider_md, mem_type);
    mapped_buffer host_buf(LENGTH, 0, is_get ? sender() : receiver());
    /* Declared after the memory, so it is deregistered before being freed */
    ucs::handle<uct_mem_h, uct_md_h> memh;

    mem_attr.field_mask = UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_FD |
                          UCT_MD_MEM_ATTR_V2_FIELD_DMABUF_OFFSET;
    ASSERT_UCS_OK(uct_md_mem_query_v2(provider_md, mem.address, LENGTH,
                                      &mem_attr));
    ASSERT_NE(UCT_DMABUF_FD_INVALID, mem_attr.dmabuf_fd);

    reg_params.field_mask    = UCT_MD_MEM_REG_FIELD_FLAGS |
                               UCT_MD_MEM_REG_FIELD_DMABUF_FD |
                               UCT_MD_MEM_REG_FIELD_DMABUF_OFFSET;
    reg_params.flags         = UCT_MD_MEM_ACCESS_ALL;
    reg_params.dmabuf_fd     = mem_attr.dmabuf_fd;
    reg_params.dmabuf_offset = mem_attr.dmabuf_offset;
    status = uct_md_mem_reg_v2(mem_entity.md(), mem.address, LENGTH,
                               &reg_params, &uct_memh);
    close(mem_attr.dmabuf_fd);
    ASSERT_UCS_OK(status);
    memh.reset(uct_memh, [](uct_mem_h handle, uct_md_h md) {
        EXPECT_UCS_OK(uct_md_mem_dereg(md, handle));
    }, mem_entity.md());

    mem_buffer::pattern_fill(mem.address, LENGTH, SEED1, mem_type);

    uct_iov_t iov = {mem.address, LENGTH, memh, 0, 1};
    if (is_get) {
        reg_mem      = mem;
        reg_mem.memh = memh;
        mem_entity.rkey_unpack(&reg_mem, &rkey);
    }

    do {
        status = is_get ?
                 uct_ep_get_zcopy(sender_ep(), host_buf.iov(), 1,
                                  (uintptr_t)mem.address, rkey.rkey, NULL) :
                 uct_ep_put_zcopy(sender_ep(), &iov, 1, host_buf.addr(),
                                  host_buf.rkey(), NULL);
        progress();
    } while (status == UCS_ERR_NO_RESOURCE);
    EXPECT_FALSE(UCS_STATUS_IS_ERR(status)) << ucs_status_string(status);
    flush();

    if (is_get) {
        mem_entity.rkey_release(&rkey);
    }

    host_buf.pattern_check(SEED1);
}

void uct_p2p_rma_test_dmabuf::test_dmabuf_xfer(bool is_get)
{
    unsigned num_mem_types = 0;
    uct_md_attr_v2_t md_attr;
    unsigned mem_type;

    if (!(sender().md_attr().flags & UCT_MD_FLAG_REG_DMABUF)) {
        UCS_TEST_SKIP_R("the memory domain cannot register a DMA-BUF");
    }

    for (const md_resource &rsc : enum_md_resources()) {
        ucs::handle<uct_md_config_t*> md_config;
        ucs::handle<uct_md_h> md;

        UCS_TEST_CREATE_HANDLE(uct_md_config_t*, md_config,
                               (void (*)(uct_md_config_t*))uct_config_release,
                               uct_md_config_read, rsc.cmpt, NULL, NULL);
        UCS_TEST_CREATE_HANDLE(uct_md_h, md, uct_md_close, uct_md_open,
                               rsc.cmpt, rsc.rsc_desc.md_name, md_config);

        md_attr.field_mask = UCT_MD_ATTR_FIELD_ALLOC_MEM_TYPES |
                             UCT_MD_ATTR_FIELD_DMABUF_MEM_TYPES;
        ASSERT_UCS_OK(uct_md_query_v2(md, &md_attr));
        ucs_for_each_bit(mem_type, md_attr.alloc_mem_types &
                                   md_attr.dmabuf_mem_types) {
            if (mem_buffer::is_mem_type_supported(
                        (ucs_memory_type_t)mem_type)) {
                test_dmabuf_xfer(md, (ucs_memory_type_t)mem_type, is_get);
                ++num_mem_types;
            }
        }
    }

    if (num_mem_types == 0) {
        UCS_TEST_SKIP_R("no memory domain exports allocated memory as a "
                        "DMA-BUF");
    }
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_dmabuf, put_zcopy,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY))
{
    test_dmabuf_xfer(false);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_dmabuf, get_zcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    test_dmabuf_xfer(true);
}

UCT_INSTANTIATE_RC_TEST_CASE(uct_p2p_rma_test_dmabuf)
