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
    static const uint64_t HDR   = 0xdeadbeef1337feedul;

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
        uint64_t *hdr = (uint64_t*)data;
        if (*hdr != HDR) {
            UCS_TEST_ABORT("Invalid AM header received");
        }
        buffer::pattern_check(hdr + 1, length - sizeof(*hdr), SEED1);
        ++m_am_count;
    }

    static void am_pack(void *dest, void *arg, size_t length) {
        uint64_t *hdr = (uint64_t*)dest;
        *hdr = HDR;
        memcpy(hdr + 1, arg, length - sizeof(*hdr));
    }

    ucs_status_t am_short(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
        return uct_ep_am_short(ep, AM_ID, HDR, sendbuf.ptr(), sendbuf.length());
    }

    ucs_status_t am_bcopy(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
        /* TODO test with different header sizes */
        return uct_ep_am_bcopy(ep, AM_ID, am_pack,
                               sendbuf.ptr(), sendbuf.length() + sizeof(HDR));
    }

    ucs_status_t am_zcopy(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
         /* TODO test with different header sizes */
         uint64_t hdr = HDR;
         return uct_ep_am_zcopy(ep, AM_ID, &hdr, sizeof(hdr),
                                sendbuf.ptr(), sendbuf.length(), sendbuf.lkey(),
                                &m_completion->uct);
    }

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction) {
        ucs_status_t status;

        status = uct_set_am_handler(receiver().iface(), AM_ID, am_handler, (void*)this);
        ASSERT_UCS_OK(status);

        mapped_buffer sendbuf(length, 1, SEED1, sender());
        mapped_buffer recvbuf(0, 0, 0, sender()); /* dummy */

        unsigned count = m_completion_count;
        status = (this->*send)(sender_ep(), sendbuf, recvbuf);

        wait_for_local(status, count);
        sendbuf.pattern_fill(SEED2);

        wait_for_remote();
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
                    0ul,
                    sender().iface_attr().cap.am.max_short - sizeof(uint64_t),
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_bcopy) {
    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_bcopy),
                    0ul,
                    sender().iface_attr().cap.am.max_bcopy - sizeof(HDR),
                    DIRECTION_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_zcopy) {
    check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_zcopy),
                    0ul,
                    sender().iface_attr().cap.am.max_zcopy - sizeof(HDR),
                    DIRECTION_SEND_TO_RECV);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_am_test)
