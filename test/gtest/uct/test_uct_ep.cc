/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
}
#include "uct_test.h"


class test_uct_ep : public uct_test {
protected:

    void init() {
        uct_test::init();
        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        check_skip_test();

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        uct_iface_set_am_handler(m_receiver->iface(), 1,
                                 (uct_am_callback_t)ucs_empty_function_return_success,
                                 NULL, UCT_CB_FLAG_ASYNC);
    }

    void connect() {
        m_sender->connect(0, *m_receiver, 0);
        short_progress_loop(10); /* Some transports need time to become ready */
    }

    void disconnect() {
        flush();
        if (m_receiver->iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            m_receiver->destroy_ep(0);
        }
        m_sender->destroy_ep(0);
    }

    bool skip_on_ib_dc() {
#ifdef HAVE_DC_DV /* skip due to DCI stuck bug */
        return has_transport("dc_mlx5");
#else
        return false;
#endif
    }

    entity * m_sender;
    entity * m_receiver;
};

UCS_TEST_SKIP_COND_P(test_uct_ep, disconnect_after_send,
                     (!check_caps(UCT_IFACE_FLAG_AM_ZCOPY) ||
                      skip_on_ib_dc())) {
    ucs_status_t status;

    mapped_buffer buffer(256, 0, *m_sender);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buffer.ptr(),
                            (ucs_min(buffer.length(), m_sender->iface_attr().cap.am.max_zcopy)),
                            buffer.memh(),
                            m_sender->iface_attr().cap.am.max_iov);

    unsigned max_iter = 300 / ucs::test_time_multiplier();
    for (unsigned i = 0; i < max_iter; ++i) {
        connect();
        for (unsigned count = 0; count < max_iter; ) {
            status = uct_ep_am_zcopy(m_sender->ep(0), 1, NULL, 0, iov, iovcnt,
                                     0, NULL);
            if (status == UCS_ERR_NO_RESOURCE) {
                if (count > 0) {
                    break;
                }
                progress();
            } else {
                ASSERT_UCS_OK_OR_INPROGRESS(status);
                ++count;
            }
        }
        disconnect();
        short_progress_loop();
    }
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_ep)
