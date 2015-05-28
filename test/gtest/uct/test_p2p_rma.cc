/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
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
        return uct_ep_put_bcopy(ep,
                                (uct_pack_callback_t)memcpy,
                                sendbuf.ptr(), sendbuf.length(),
                                recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t put_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        m_completion->length = 0;
        return uct_ep_put_zcopy(ep,
                                sendbuf.ptr(), sendbuf.length(), sendbuf.memh(),
                                recvbuf.addr(), recvbuf.rkey(),
                                &m_completion->uct);
    }

    ucs_status_t get_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        m_completion->dest   = sendbuf.ptr();
        m_completion->length = sendbuf.length();
        return uct_ep_get_bcopy(ep, sendbuf.length(), recvbuf.addr(),
                                recvbuf.rkey(), &m_completion->uct);
    }

    ucs_status_t get_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        m_completion->length = 0;
        return uct_ep_get_zcopy(ep,
                                sendbuf.ptr(), sendbuf.length(), sendbuf.memh(),
                                recvbuf.addr(), recvbuf.rkey(),
                                &m_completion->uct);
    }

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction) {
        mapped_buffer sendbuf(length, SEED1, sender(), 1);
        mapped_buffer recvbuf(length, SEED2, receiver(), 3);

        blocking_send(send, sender_ep(), sendbuf, recvbuf, m_completion_count);
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
                    1ul, sender().iface_attr().cap.get.max_zcopy,
                    DIRECTION_RECV_TO_SEND);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_test)
