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

    ucs_status_t put_short(const entity& e, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
         return uct_ep_put_short(e.ep(), sendbuf.ptr(), sendbuf.length(),
                                 recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t put_bcopy(const entity& e, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        return uct_ep_put_bcopy(e.ep(),
                                (uct_pack_callback_t)memcpy,
                                sendbuf.ptr(), sendbuf.length(),
                                recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t put_zcopy(const entity& e, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        return uct_ep_put_zcopy(e.ep(),
                                sendbuf.ptr(), sendbuf.length(), sendbuf.lkey(),
                                recvbuf.addr(), recvbuf.rkey(),
                                &m_completion->uct);
    }

    virtual void test_xfer(send_func_t send, size_t length) {
        mapped_buffer sendbuf(length, 1, SEED1, get_entity(0));
        mapped_buffer recvbuf(length, 1, SEED2, get_entity(1));

        unsigned count = m_completion_count;
        ucs_status_t status = (this->*send)(get_entity(0), sendbuf, recvbuf);

        wait_for_local(status, count);
        sendbuf.pattern_fill(SEED3);

        wait_for_remote();
        recvbuf.pattern_check(SEED1);
    }
};

UCS_TEST_P(uct_p2p_rma_test, put_short) {
    if (!(get_entity(0).iface_attr().cap.flags & UCT_IFACE_FLAG_PUT_SHORT)) { 
        UCS_TEST_SKIP;
    }
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_short),
                    0ul, get_entity(0).iface_attr().cap.put.max_short);
}

UCS_TEST_P(uct_p2p_rma_test, put_bcopy) {
    if (!(get_entity(0).iface_attr().cap.flags & UCT_IFACE_FLAG_PUT_BCOPY)) { 
        UCS_TEST_SKIP;
    }
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_bcopy),
                    0ul, get_entity(0).iface_attr().cap.put.max_bcopy);
}

UCS_TEST_P(uct_p2p_rma_test, put_zcopy) {
    if (!(get_entity(0).iface_attr().cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY)) { 
        UCS_TEST_SKIP;
    }
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
                    0ul, get_entity(0).iface_attr().cap.put.max_zcopy);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_test)
