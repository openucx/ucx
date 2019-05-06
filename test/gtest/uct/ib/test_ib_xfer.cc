/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/test_p2p_rma.h>
#include <uct/test_p2p_mix.h>
#include <uct/ib/base/ib_alloc.h>


class uct_p2p_rma_test_inlresp : public uct_p2p_rma_test {};

UCS_TEST_P(uct_p2p_rma_test_inlresp, get_bcopy_inlresp0, "IB_TX_INLINE_RESP=0") {
    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_P(uct_p2p_rma_test_inlresp, get_bcopy_inlresp64, "IB_TX_INLINE_RESP=64") {
    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_rma_test_inlresp)


class uct_p2p_rma_test_alloc_methods : public uct_p2p_rma_test {
protected:
    void test_put_zcopy() {
        check_caps(UCT_IFACE_FLAG_PUT_ZCOPY);
        test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                        0, sender().iface_attr().cap.put.max_zcopy,
                        TEST_UCT_FLAG_SEND_ZCOPY);
    }

    void test_get_zcopy() {
        check_caps(UCT_IFACE_FLAG_GET_ZCOPY);
        test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                        sender().iface_attr().cap.get.min_zcopy,
                        sender().iface_attr().cap.get.max_zcopy,
                        TEST_UCT_FLAG_RECV_ZCOPY);
    }
};

UCS_TEST_P(uct_p2p_rma_test_alloc_methods, xfer_reg_odp,
           "REG_METHODS=odp,direct")
{
    test_put_zcopy();
    test_get_zcopy();
}

UCS_TEST_P(uct_p2p_rma_test_alloc_methods, xfer_reg_rcache,
           "REG_METHODS=rcache,direct")
{
    test_put_zcopy();
    test_get_zcopy();
}

UCS_TEST_P(uct_p2p_rma_test_alloc_methods, xfer_reg_direct,
           "REG_METHODS=direct")
{
    test_put_zcopy();
    test_get_zcopy();
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_rma_test_alloc_methods)


class uct_p2p_mix_test_alloc_methods : public uct_p2p_mix_test {};

UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000_odp,
           "REG_METHODS=odp,direct")
{
    run(1000);
}

UCS_TEST_P(uct_p2p_mix_test_alloc_methods, mix1000_rcache,
           "REG_METHODS=rcache,direct")
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


class uct_p2p_mix_test_dm : public uct_p2p_mix_test {
public:
    virtual void run(unsigned count) {

        check_run_conditions();

        size_t size = m_send_size;
        uct_ib_device_mem_h dev_mem;
        ucs_status_t status;
        uct_mem_h dm_memh;
        void *dm_ptr;

        status = uct_ib_md_alloc_device_mem(receiver().md(), &size, &dm_ptr,
                                            UCT_MD_MEM_ACCESS_ALL, "test DM",
                                            &dev_mem);
        if ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_ERR_UNSUPPORTED)) {
            UCS_TEST_SKIP_R("Device memory is not available");
        }
        ASSERT_UCS_OK(status);

        status = uct_md_mem_reg(receiver().md(), dm_ptr, m_send_size,
                                UCT_MD_MEM_ACCESS_ALL, &dm_memh);
        ASSERT_UCS_OK(status);

        mapped_buffer sendbuf(m_send_size, 1, sender());
        mapped_buffer recvbuf(dm_ptr, m_send_size, dm_memh, 2, receiver());

        for (unsigned i = 0; i < count; ++i) {
            random_op(sendbuf, recvbuf);
        }

        sender().flush();

        status = uct_md_mem_dereg(receiver().md(), dm_memh);
        ASSERT_UCS_OK(status);

        uct_ib_md_release_device_mem(dev_mem);
    }
};

UCS_TEST_P(uct_p2p_mix_test_dm, mix1000)
{
    run(1000);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_mix_test_dm)

