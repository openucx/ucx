/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"

#include <functional>

class uct_p2p_rma_test : public uct_p2p_test {
public:
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    static const uint64_t SEED3 = 0x3333333333333333lu;

    uct_p2p_rma_test() : uct_p2p_test(0) {
    }

    ucs_status_t put_short(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
         return uct_ep_put_short(ep, sendbuf.ptr(), sendbuf.length(),
                                 recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t put_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        ssize_t packed_len;
        packed_len = uct_ep_put_bcopy(ep, mapped_buffer::pack, (void*)&sendbuf,
                                      recvbuf.addr(), recvbuf.rkey());
        if (packed_len >= 0) {
            EXPECT_EQ(sendbuf.length(), (size_t)packed_len);
            return UCS_OK;
        } else {
            return (ucs_status_t)packed_len;
        }
    }

    ucs_status_t put_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                                sendbuf.memh(), sender().iface_attr().cap.put.max_iov, 0);

        return uct_ep_put_zcopy(ep, iov, iovcnt, recvbuf.addr(), recvbuf.rkey(), comp());
    }

    ucs_status_t get_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        return uct_ep_get_bcopy(ep, (uct_unpack_callback_t)memcpy, sendbuf.ptr(),
                                sendbuf.length(), recvbuf.addr(),
                                recvbuf.rkey(), comp());
    }

    ucs_status_t get_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                                sendbuf.memh(), sender().iface_attr().cap.get.max_iov, 0);

        return uct_ep_get_zcopy(ep, iov, iovcnt, recvbuf.addr(), recvbuf.rkey(), comp());
    }

    virtual void test_xfer(send_func_t send, size_t length,
                           direction_t direction) {
        mapped_buffer sendbuf(length, SEED1, sender(), 1);
        mapped_buffer recvbuf(length, SEED2, receiver(), 3);

        blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
        if (direction == DIRECTION_SEND_TO_RECV) {
            sendbuf.pattern_fill(SEED3);
            wait_for_remote();
            recvbuf.pattern_check(SEED1);
        } else if (direction == DIRECTION_RECV_TO_SEND) {
            recvbuf.pattern_fill(SEED3);
            sendbuf.pattern_check(SEED2);
            wait_for_remote();
        }
    }
};

UCS_TEST_P(uct_p2p_rma_test, put_short) {
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
                    0ul, sender().iface_attr().cap.put.max_short,
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_rma_test, put_bcopy) {
    check_caps(UCT_IFACE_FLAG_PUT_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_bcopy),
                    0ul, sender().iface_attr().cap.put.max_bcopy,
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_rma_test, put_zcopy) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                    0ul, sender().iface_attr().cap.put.max_zcopy,
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_rma_test, get_bcopy) {
    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    DIRECTION_RECV_TO_SEND);
}

UCS_TEST_P(uct_p2p_rma_test, get_zcopy) {
    check_caps(UCT_IFACE_FLAG_GET_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                    ucs_max(1ull, sender().iface_attr().cap.get.min_zcopy),
                    sender().iface_attr().cap.get.max_zcopy,
                    DIRECTION_RECV_TO_SEND);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_test)

class uct_p2p_rma_test_inlresp : public uct_p2p_rma_test {};

UCS_TEST_P(uct_p2p_rma_test_inlresp, get_bcopy_inlresp0, "IB_TX_INLINE_RESP=0") {
    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    DIRECTION_RECV_TO_SEND);
}

UCS_TEST_P(uct_p2p_rma_test_inlresp, get_bcopy_inlresp64, "IB_TX_INLINE_RESP=64") {
    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    DIRECTION_RECV_TO_SEND);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_p2p_rma_test_inlresp)

class uct_p2p_rma_nc_test : public uct_p2p_rma_test {
public:
    uct_p2p_rma_nc_test() : uct_p2p_rma_test() {
    }

    ucs_status_t put_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        ucs_status_t status;
        uct_iov_t iov = {0};
        iov.buffer = sendbuf.ptr();
        iov.length = sendbuf.nc_length();
        iov.memh = sendbuf.nc_memh();
        iov.count = 1;

        return uct_ep_put_zcopy(ep, &iov, 1, recvbuf.addr(), recvbuf.nc_rkey(), comp());
    }

    ucs_status_t get_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        ucs_status_t status;
        uct_iov_t iov = {0};
        iov.buffer = sendbuf.ptr();
        iov.length = sendbuf.nc_length();
        iov.memh = sendbuf.nc_memh();
        iov.count = 1;

        return uct_ep_get_zcopy(ep, &iov, 1, recvbuf.addr(), recvbuf.nc_rkey(), comp());
    }

    virtual void test_xfer(send_func_t send, size_t length,
                           direction_t direction) {
        mapped_buffer sendbuf(length, SEED1, sender(), 2, 0, sender_ep());
        mapped_buffer recvbuf(length, SEED2, receiver(), 3, 1, receiver_ep());

        blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
        if (direction == DIRECTION_SEND_TO_RECV) {
            sendbuf.pattern_fill(SEED3);
            wait_for_remote();
            recvbuf.pattern_check(SEED1);
        } else if (direction == DIRECTION_RECV_TO_SEND) {
            recvbuf.pattern_fill(SEED3);
            sendbuf.pattern_check(SEED2);
            wait_for_remote();
        }
    }
};

UCS_TEST_P(uct_p2p_rma_nc_test, put_zcopy) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_MEM_NC);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_nc_test::put_zcopy),
                    0ul, sender().iface_attr().cap.put.max_zcopy,
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_rma_nc_test, get_zcopy) {
    check_caps(UCT_IFACE_FLAG_GET_ZCOPY | UCT_IFACE_FLAG_MEM_NC);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_nc_test::get_zcopy),
                    ucs_max(1ull, sender().iface_attr().cap.get.min_zcopy),
                    sender().iface_attr().cap.get.max_zcopy,
                    DIRECTION_RECV_TO_SEND);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_nc_test)
