/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/test_p2p_rma.h>
#include <uct/test_p2p_mix.h>


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

#ifdef IMPLICIT_ODP_FIXED
UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_alloc_methods, xfer_reg_odp,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_GET_ZCOPY),
                     "REG_METHODS=odp,direct",
                     "MLX5_DEVX_OBJECTS=dct,dcsrq")
{
    test_put_zcopy();
    test_get_zcopy();
}
#endif

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_alloc_methods, xfer_reg_rcache,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_GET_ZCOPY),
                     "REG_METHODS=rcache,direct")
{
    test_put_zcopy();
    test_get_zcopy();
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test_alloc_methods, xfer_reg_direct,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_GET_ZCOPY),
                     "REG_METHODS=direct")
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

#ifdef IMPLICIT_ODP_FIXED
UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000_odp,
           "REG_METHODS=odp,direct", "MLX5_DEVX_OBJECTS=dct,dcsrq")
{
    run(1000);
}
#endif

UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000_rcache,
           "REG_METHODS=rcache,direct")
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

