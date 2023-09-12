/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/test_p2p_rma.h>
#include <uct/test_p2p_mix.h>


class uct_p2p_rma_test_xfer : public uct_p2p_rma_test {};

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_xfer, fence_relaxed_order,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY),
                     "PCI_RELAXED_ORDERING=try") {
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
                     "IB_TX_INLINE_RESP=0", "MLX5_DEVX=n") {
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
                        0, sender().iface_attr().cap.put.max_zcopy,
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
                     "REG_MT_THRESH=1", "REG_MT_CHUNK=1G", "REG_MT_BIND=y")
{
    test_put_zcopy();
    test_get_zcopy();
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_rma_test_alloc_methods)


class uct_p2p_mix_test_alloc_methods : public uct_p2p_mix_test {};

UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000)
{
    run(1000);
}

UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000_multithreaded,
           "REG_MT_THRESH=1", "REG_MT_CHUNK=1K", "REG_MT_BIND=y")
{
    run(1000);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_mix_test_alloc_methods)


class uct_p2p_mix_test_indirect_atomic : public uct_p2p_mix_test {};

UCS_TEST_P(uct_p2p_mix_test_indirect_atomic, mix1000_indirect_atomic,
           "INDIRECT_ATOMIC=n")
{
    run(1000);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_mix_test_indirect_atomic)

