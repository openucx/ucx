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

ucs_status_t uct_p2p_rma_test::get_short(uct_ep_h ep, const mapped_buffer &sendbuf,
                                         const mapped_buffer &recvbuf)
{
     return uct_ep_get_short(ep, sendbuf.ptr(), sendbuf.length(),
                             recvbuf.addr(), recvbuf.rkey());
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
                                 unsigned flags, ucs_memory_type_t mem_type)
{
    ucs_memory_type_t src_mem_type = UCS_MEMORY_TYPE_HOST;

    if (has_transport("cuda_ipc") ||
        has_transport("rocm_copy")) {
        src_mem_type = mem_type;
    }

    mapped_buffer sendbuf(length, SEED1, sender(), 1, src_mem_type);
    mapped_buffer recvbuf(length, SEED2, receiver(), 3, mem_type);

    blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
    if (flags & TEST_UCT_FLAG_SEND_ZCOPY) {
        sendbuf.memset(0);
        wait_for_remote();
        recvbuf.pattern_check(SEED1);
    } else if (flags & TEST_UCT_FLAG_RECV_ZCOPY) {
        recvbuf.memset(0);
        sendbuf.pattern_check(SEED2);
        wait_for_remote();
    }
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test, put_short,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
                    0ul, sender().iface_attr().cap.put.max_short,
                    TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test, put_bcopy,
                     !check_caps(UCT_IFACE_FLAG_PUT_BCOPY)) {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_bcopy),
                    0ul, sender().iface_attr().cap.put.max_bcopy,
                    TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test, put_zcopy,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY)) {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                    0ul, sender().iface_attr().cap.put.max_zcopy,
                    TEST_UCT_FLAG_SEND_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test, get_short,
                     !check_caps(UCT_IFACE_FLAG_GET_SHORT)) {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_short),
                    0ul, sender().iface_attr().cap.get.max_short,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test, get_bcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_BCOPY)) {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_bcopy),
                    1ul, sender().iface_attr().cap.get.max_bcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCS_TEST_SKIP_COND_P(uct_p2p_rma_test, get_zcopy,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY)) {
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                    ucs_max(1ull, sender().iface_attr().cap.get.min_zcopy),
                    sender().iface_attr().cap.get.max_zcopy,
                    TEST_UCT_FLAG_RECV_ZCOPY);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_test)

class test_p2p_rma_madvise : private ucs::clear_dontcopy_regions,
                             public uct_p2p_rma_test
{
};

UCS_TEST_SKIP_COND_P(test_p2p_rma_madvise, madvise,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY),
                     /* Allocate with mmap to avoid pinning other heap memory */
                     "IB_ALLOC?=mmap")
{
    mapped_buffer sendbuf(4096, 0, sender());
    mapped_buffer recvbuf(4096, 0, receiver());
    char cmd_str[] = "/bin/bash -c 'exit 42'";

    blocking_send(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                  sender_ep(), sendbuf, recvbuf, true);
    flush();

    int exit_status = system(cmd_str);
    EXPECT_TRUE(WIFEXITED(exit_status));
    EXPECT_EQ(42, WEXITSTATUS(exit_status)) << ucs::exit_status_info(exit_status);

    blocking_send(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
                  sender_ep(), sendbuf, recvbuf, true);
    flush();
}

UCT_INSTANTIATE_TEST_CASE(test_p2p_rma_madvise)
