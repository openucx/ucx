/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"

#include <string>

class uct_p2p_am_test : public uct_p2p_test
{
public:
    static const uint8_t AM_ID = 11;
    static const uint64_t SEED1 = 0xa1a1a1a1a1a1a1a1ul;
    static const uint64_t SEED2 = 0xa2a2a2a2a2a2a2a2ul;
    static const uint64_t MAGIC = 0xdeadbeef12345678ul;

    typedef struct {
        uint64_t magic;
        unsigned length;
        /* data follows */
    } receive_desc_t;

    uct_p2p_am_test() :
        uct_p2p_test(sizeof(receive_desc_t)),
        m_keep_data(false),
        m_am_count(0) {
    }

    virtual void init() {
        uct_p2p_test::init();
        m_am_count = 0;
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length, void *desc) {
        uct_p2p_am_test *self = reinterpret_cast<uct_p2p_am_test*>(arg);
        return self->am_handler(data, length, desc);
    }

    ucs_status_t am_handler(void *data, size_t length, void *desc) {
        ++m_am_count;
        if (m_keep_data) {
            receive_desc_t *my_desc = (receive_desc_t *)desc;
            my_desc->magic  = MAGIC;
            my_desc->length = length;
            if (data != my_desc + 1) {
                memcpy(my_desc + 1, data, length);
            }
            m_backlog.push_back(my_desc);
            return UCS_INPROGRESS;
        }
        mapped_buffer::pattern_check(data, length, SEED1);
        return UCS_OK;
    }

    void check_backlog() {
        while (!m_backlog.empty()) {
            receive_desc_t *my_desc = m_backlog.back();
            m_backlog.pop_back();
            EXPECT_EQ(uint64_t(MAGIC), my_desc->magic);
            mapped_buffer::pattern_check(my_desc + 1, my_desc->length, SEED1);
            uct_iface_release_am_desc(my_desc);
        }
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
        return uct_ep_am_zcopy(ep,
                               AM_ID,
                               sendbuf.ptr(),
                               hdr_size,
                               (char*)sendbuf.ptr() + hdr_size,
                               sendbuf.length()     - hdr_size,
                               sendbuf.memh(),
                               comp());
    }

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction) {
        ucs_status_t status;

        m_am_count = 0;

        status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler, (void*)this);
        ASSERT_UCS_OK(status);

        mapped_buffer sendbuf(length, SEED1, sender());
        mapped_buffer recvbuf(0, 0, sender()); /* dummy */

        blocking_send(send, sender_ep(), sendbuf, recvbuf);
        sendbuf.pattern_fill(SEED2);

        while (m_am_count == 0) {
            short_progress_loop();
        }

        status = uct_iface_set_am_handler(receiver().iface(), AM_ID, NULL, NULL);
        ASSERT_UCS_OK(status);

        check_backlog();
    }

    void set_keep_data(bool keep) {
        m_keep_data = keep;
    }

private:
    bool                         m_keep_data;
    unsigned                     m_am_count;
    std::vector<receive_desc_t*> m_backlog;

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

UCS_TEST_P(uct_p2p_am_test, am_short_keep_data) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);
    set_keep_data(true);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_short),
                    sizeof(uint64_t),
                    sender().iface_attr().cap.am.max_short,
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
