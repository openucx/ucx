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

    static ucs_status_t am_handler(void *data, unsigned length, void *arg) {
        uct_p2p_am_test *self = reinterpret_cast<uct_p2p_am_test*>(arg);
        self->am_handler(data, length);
        return UCS_OK; /* TODO test keeping data */
    }

    void am_handler(void *data, unsigned length) {
        m_data.assign(reinterpret_cast<char*>(data), length);
    }

    std::string m_data;
};

UCS_TEST_P(uct_p2p_am_test, am8) {
    const uint64_t hdr = 0xdeadbeed1ee7a880;
    const std::string payload = "TestAM";
    ucs_status_t status;

    status = uct_set_am_handler(get_entity(1).iface(), AM_ID, am_handler, (void*)this);
    ASSERT_UCS_OK(status);

    m_data.clear();

    status = uct_ep_am_short(get_entity(0).ep(), AM_ID, hdr,
                             (void*)&payload[0], payload.length());
    ASSERT_UCS_OK(status);

    short_progress_loop();

    ASSERT_EQ(sizeof(hdr) + payload.length(), m_data.length());

    EXPECT_EQ(hdr, *reinterpret_cast<uint64_t*>(&m_data[0]));
    EXPECT_EQ(payload, m_data.substr(sizeof(hdr)));

    status = uct_set_am_handler(get_entity(1).iface(), AM_ID, NULL, NULL);
    ASSERT_UCS_OK(status);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_am_test)
