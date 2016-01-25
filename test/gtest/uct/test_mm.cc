/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include "uct_p2p_test.h"
#include <common/test.h>
#include "uct_test.h"

class test_uct_mm : public uct_test {
public:

    void initialize() {
        if (GetParam()->dev_name == "posix") {
            set_config("USE_SHM_OPEN=no");
        }
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_e2 = uct_test::create_entity(0);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static ucs_status_t mm_am_handler(void *arg, void *data, size_t length,
            void *desc) {
        recv_desc_t *my_desc = (recv_desc_t *) arg;
        uint64_t *test_mm_hdr = (uint64_t *) data;
        uint64_t *actual_data = (uint64_t *) test_mm_hdr + 1;
        unsigned data_length = length - sizeof(test_mm_hdr);

        my_desc->length = data_length;
        if (*test_mm_hdr == 0xbeef) {
            memcpy(my_desc + 1, actual_data, data_length);
        }

        return UCS_OK;
    }

    void short_progress_loop() {
        ucs_time_t end_time = ucs_get_time()
                + ucs_time_from_msec(100.0) * ucs::test_time_multiplier();
        while (ucs_get_time() < end_time) {
            progress();
        }
    }

    void cleanup() {
        uct_test::cleanup();
    }

protected:
    entity *m_e1, *m_e2;
};

UCS_TEST_P(test_uct_mm, open_for_posix) {
    uint64_t send_data   = 0xdeadbeef;
    uint64_t test_mm_hdr = 0xbeef;
    recv_desc_t *recv_buffer;

    initialize();
    check_caps(UCT_IFACE_FLAG_AM_SHORT);

    recv_buffer = (recv_desc_t *) malloc(sizeof(*recv_buffer) + sizeof(uint64_t));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, mm_am_handler , recv_buffer, 0);

    /* send the data */
    uct_ep_am_short(m_e1->ep(0), 0, test_mm_hdr, &send_data, sizeof(send_data));

    short_progress_loop();

    ASSERT_EQ(sizeof(send_data), recv_buffer->length);
    EXPECT_EQ(send_data, *(uint64_t*)(recv_buffer+1));

    free(recv_buffer);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_mm, mm)
