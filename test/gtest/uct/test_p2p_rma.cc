/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_p2p_rma.h"

#include <functional>


uct_p2p_rma_test::uct_p2p_rma_test() : uct_p2p_test(0) {
}

ucs_status_t uct_p2p_rma_test::put_short(uct_ep_h ep, const mapped_buffer &sendbuf,
                       const mapped_buffer &recvbuf)
{
     return uct_ep_put_short(ep, sendbuf.ptr(), sendbuf.length(),
                             recvbuf.addr(), recvbuf.rkey());
}

ucs_status_t uct_p2p_rma_test::put_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
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

ucs_status_t uct_p2p_rma_test::put_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                                         const mapped_buffer &recvbuf)
{
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(), sender().iface_attr().cap.put.max_iov);

    return uct_ep_put_zcopy(ep, iov, iovcnt, recvbuf.addr(), recvbuf.rkey(), comp());
}

ucs_status_t uct_p2p_rma_test::get_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                                         const mapped_buffer &recvbuf)
{
    return uct_ep_get_bcopy(ep, (uct_unpack_callback_t)memcpy, sendbuf.ptr(),
                            sendbuf.length(), recvbuf.addr(),
                            recvbuf.rkey(), comp());
}

ucs_status_t uct_p2p_rma_test::get_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                                         const mapped_buffer &recvbuf)
{
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(), sender().iface_attr().cap.get.max_iov);

    return uct_ep_get_zcopy(ep, iov, iovcnt, recvbuf.addr(), recvbuf.rkey(), comp());
}

void uct_p2p_rma_test::test_xfer(send_func_t send, size_t length,
                                 unsigned flags, uct_memory_type_t mem_type)
{
    mapped_buffer sendbuf(length, SEED1, sender(), 1);
    mapped_buffer recvbuf(length, SEED2, receiver(), 3, mem_type);

    blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
    if (flags & TEST_UCT_FLAG_SEND_ZCOPY) {
        sendbuf.pattern_fill(SEED3);
        wait_for_remote();
        recvbuf.pattern_check(SEED1);
    } else if (flags & TEST_UCT_FLAG_RECV_ZCOPY) {
        recvbuf.pattern_fill(SEED3);
        sendbuf.pattern_check(SEED2);
        wait_for_remote();
    }
}

UCS_TEST_P(uct_p2p_rma_test, put_short) {
    check_caps(UCT_IFACE_FLAG_PUT_SHORT);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
                    0ul, sender().iface_attr().cap.put.max_short,
                    TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_P(uct_p2p_rma_test, put_bcopy) {
    check_caps(UCT_IFACE_FLAG_PUT_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_bcopy),
                    0ul, sender().iface_attr().cap.put.max_bcopy,
                    TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_P(uct_p2p_rma_test, put_zcopy) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                    0ul, sender().iface_attr().cap.put.max_zcopy,
                    TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_P(uct_p2p_rma_test, get_bcopy) {
    check_caps(UCT_IFACE_FLAG_GET_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_P(uct_p2p_rma_test, get_zcopy) {
    check_caps(UCT_IFACE_FLAG_GET_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                    ucs_max(1ull, sender().iface_attr().cap.get.min_zcopy),
                    sender().iface_attr().cap.get.max_zcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_test)
