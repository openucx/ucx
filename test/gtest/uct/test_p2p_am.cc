/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_p2p_test.h"

#include <string>

class uct_p2p_am_test : public uct_p2p_test {
public:
    static const uint8_t AM_ID = 11;
    static const uint64_t SEED1 = 0xa1a1a1a1a1a1a1a1ul;
    static const uint64_t SEED2 = 0xa2a2a2a2a2a2a2a2ul;

    virtual void init() {
        uct_p2p_test::init();
        m_am_count = 0;
    }

    static ucs_status_t am_handler(void *desc, void *data, size_t length, void *arg) {
        uct_p2p_am_test *self = reinterpret_cast<uct_p2p_am_test*>(arg);
        self->am_handler(data, length);
        return UCS_OK; /* TODO test keeping data */
    }

    void am_handler(void *data, size_t length) {
        buffer::pattern_check(data, length, SEED1);
        ++m_am_count;
    }

    static void am_pack(void *dest, void *arg, size_t length) {
        memcpy(dest, arg, length);
    }

    ucs_status_t am_short(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
        uint64_t hdr = *(uint64_t*)sendbuf.ptr();
        return uct_ep_am_short(ep, AM_ID, hdr, (char*)sendbuf.ptr() + sizeof(hdr),
                               sendbuf.length() - sizeof(hdr));
    }

    ucs_status_t am_bcopy(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
        return uct_ep_am_bcopy(ep, AM_ID, am_pack, sendbuf.ptr(), sendbuf.length());
    }

    ucs_status_t am_zcopy(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
        size_t max_hdr  = ucs_min(sender().iface_attr().cap.am.max_hdr,
                                  sendbuf.length());
        size_t hdr_size = rand() % (max_hdr + 1);
        return uct_ep_am_zcopy(ep, AM_ID, sendbuf.ptr(), hdr_size,
                               (char*)sendbuf.ptr() + hdr_size, sendbuf.length() - hdr_size,
                               sendbuf.lkey(), &m_completion->uct);
    }

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction) {
        ucs_status_t status;

        m_am_count = 0;

        status = uct_set_am_handler(receiver().iface(), AM_ID, am_handler, (void*)this);
        ASSERT_UCS_OK(status);

        mapped_buffer sendbuf(length, 1, SEED1, sender());
        mapped_buffer recvbuf(0, 0, 0, sender()); /* dummy */

        blocking_send(send, sender_ep(), sendbuf, recvbuf, m_completion_count);
        sendbuf.pattern_fill(SEED2);

        while (m_am_count == 0) {
            short_progress_loop();
        }

        status = uct_set_am_handler(receiver().iface(), AM_ID, NULL, NULL);
        ASSERT_UCS_OK(status);
    }

private:
    unsigned m_am_count;

};

UCS_TEST_P(uct_p2p_am_test, am_short) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_short),
                    sizeof(uint64_t),
                    sender().iface_attr().cap.am.max_short,
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_bcopy) {
    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_bcopy),
                    0ul,
                    sender().iface_attr().cap.am.max_bcopy,
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_zcopy) {
    check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_zcopy),
                    0ul,
                    sender().iface_attr().cap.am.max_zcopy,
                    DIRECTION_SEND_TO_RECV);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_am_test)
