/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
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
    EXPECT_EQ(1u, sender().iface_attr().cap.get.min_zcopy);
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

