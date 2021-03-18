/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
}
#include "uct_test.h"

#include <new>


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

    void connect(bool should_flush = true)
    {
        m_sender->connect(0, *m_receiver, 0);

        /* Some transports need time to become ready */
        if (should_flush) {
            flush();
        }
    }

    void disconnect(bool should_flush = true)
    {
        if (should_flush) {
            flush();
        }

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

    struct test_ep_comp_t {
        uct_completion_t comp;
        test_uct_ep      *test;
        uct_ep_h         ep;
        mapped_buffer    *buffer;

        test_ep_comp_t(test_uct_ep *_test, uct_ep_h _ep,
                       mapped_buffer *_buffer,
                       uct_completion_callback_t _func) :
            test(_test), ep(_ep), buffer(_buffer)
        {
            comp.count  = 0;
            comp.status = UCS_OK;
            comp.func   = _func;
        }

    private:
        test_ep_comp_t() {}
    };

    static void completion_cb(uct_completion_t *comp)
    {
        test_ep_comp_t *ep_comp = ucs_container_of(comp, test_ep_comp_t, comp);

        EXPECT_TRUE(ep_comp->ep != NULL);
        /* Check that completion callback was invoked not after EP destroy */
        EXPECT_EQ(ep_comp->test->m_sender->ep(0), ep_comp->ep);

        if (ep_comp->buffer != NULL) {
            ep_comp->buffer->~mapped_buffer();
        }
    }

    static ucs_log_func_rc_t
    detect_uncomp_op_logger(const char *file, unsigned line,
                            const char *function, ucs_log_level_t level,
                            const ucs_log_component_config_t *comp_conf,
                            const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_WARN) {
            static std::vector<std::string> stop_list;
            if (stop_list.empty()) {
                stop_list.push_back("with uncompleted operation");
            }

            std::string err_str = format_message(message, ap);
            for (size_t i = 0; i < stop_list.size(); ++i) {
                if (err_str.find(stop_list[i]) != std::string::npos) {
                    return UCS_LOG_FUNC_RC_STOP;
                }
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
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

UCS_TEST_SKIP_COND_P(test_uct_ep, destroy_entity_after_send,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY))
{
    const unsigned max_iter = 300 / ucs::test_time_multiplier();
    char *buffer_mem        = new char[sizeof(mapped_buffer)];

    for (unsigned i = 0; i < max_iter; ++i) {
        connect(false);

        mapped_buffer *buffer = new (buffer_mem)
                mapped_buffer(256 * UCS_KBYTE, 0, *m_sender);
        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buffer->ptr(),
                                ucs_min(buffer->length(),
                                        m_sender->iface_attr().cap.am.max_zcopy),
                                buffer->memh(),
                                m_sender->iface_attr().cap.am.max_iov);

        ucs_status_t status;

        test_ep_comp_t am_zcopy_comp(this, m_sender->ep(0), buffer,
                                     completion_cb);
        test_ep_comp_t flush_comp(this, m_sender->ep(0), NULL, completion_cb);

        for (unsigned count = 0; count < max_iter;) {
            status = uct_ep_am_zcopy(m_sender->ep(0), 1, NULL, 0, iov, iovcnt,
                                     0, &am_zcopy_comp.comp);
            if (status == UCS_INPROGRESS) {
                ++am_zcopy_comp.comp.count;
            } else if (status != UCS_OK) {
                EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
                if (count == 0) {
                    progress();
                } else {
                    status = uct_ep_flush(m_sender->ep(0), UCT_FLUSH_FLAG_LOCAL,
                                          &flush_comp.comp);
                    if (status == UCS_INPROGRESS) {
                        ++flush_comp.comp.count;
                    } else {
                        EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
                    }
                    break;
                }
            }
            ++count;
        }

        if (am_zcopy_comp.comp.count == 0) {
            /* No AM Zcopy operations scheduled, just release buffer */
            buffer->~mapped_buffer();
        }

        if ((am_zcopy_comp.comp.count == 0) && (flush_comp.comp.count == 0)) {
            /* Neither AM Zcopy nor flush operatiosn scheduled */
            continue; /* retry */
        }

        {
            scoped_log_handler slh(detect_uncomp_op_logger);
            disconnect(false);
            EXPECT_EQ(0, am_zcopy_comp.comp.count);
            EXPECT_EQ(0, flush_comp.comp.count);

            m_entities.remove(m_sender);
        }

        /* Create new sender for new iteration or it will be destroyed in
         * cleanup() */
        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);
    }

    delete[] buffer_mem;
}

UCT_INSTANTIATE_NO_SELF_TEST_CASE(test_uct_ep)
