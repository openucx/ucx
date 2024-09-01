/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
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

        m_sender   = NULL;
        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        check_skip_test();

        uct_iface_set_am_handler(m_receiver->iface(), 1,
                                 (uct_am_callback_t)ucs_empty_function_return_success,
                                 NULL, UCT_CB_FLAG_ASYNC);
    }

    void create_sender()
    {
        /* Self/Memtype transports must connect to the same ucp_worker */
        if ((GetParam()->dev_type == UCT_DEVICE_TYPE_SELF) ||
            (GetParam()->dev_type == UCT_DEVICE_TYPE_ACC)) {
            m_sender = m_receiver;
        } else {
            m_sender = uct_test::create_entity(0);
            m_entities.push_back(m_sender);
        }
    }

    void connect()
    {
        m_sender->connect(0, *m_receiver, 0);

        /* Some transports need time to become ready */
        flush();
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

    struct test_ep_comp_t {
        uct_completion_t comp;
        test_uct_ep      *test;
        uct_ep_h         ep;
    };

    static void completion_cb(uct_completion_t *comp)
    {
        test_ep_comp_t *ep_comp = ucs_container_of(comp, test_ep_comp_t, comp);

        EXPECT_TRUE(ep_comp->ep != NULL);
        /* Check that completion callback was invoked not after EP destroy */
        EXPECT_EQ(ep_comp->test->m_sender->ep(0), ep_comp->ep);
    }

    static ucs_log_func_rc_t
    detect_uncomp_op_logger(const char *file, unsigned line,
                            const char *function, ucs_log_level_t level,
                            const ucs_log_component_config_t *comp_conf,
                            const char *message, va_list ap)
    {
        if (level == UCS_LOG_LEVEL_WARN) {
            std::string err_str = format_message(message, ap);
            if (err_str.find("with uncompleted operation") !=
                std::string::npos) {
                return UCS_LOG_FUNC_RC_STOP;
            }
        }
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    void handle_status(ucs_status_t status, test_ep_comp_t &comp)
    {
        if (status == UCS_INPROGRESS) {
            ++comp.comp.count;
        } else if (status != UCS_OK) {
            EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
        }
    }

    void modify_remote_id(entity &e) const;
    bool is_connected_to_sender(const entity &e) const;

    entity * m_sender;
    entity * m_receiver;
};

bool test_uct_ep::is_connected_to_sender(const entity &e) const
{
    uct_ep_is_connected_params_t params;
    uct_iface_attr_t iface_attr;
    std::string dev_addr, ep_addr, iface_addr;

    ASSERT_UCS_OK(uct_iface_query(e.iface(), &iface_attr));
    dev_addr.resize(iface_attr.device_addr_len);
    iface_addr.resize(iface_attr.iface_addr_len);

    ASSERT_UCS_OK(uct_iface_get_address(e.iface(),
                                        (uct_iface_addr_t*)iface_addr.data()));
    ASSERT_UCS_OK(
            uct_iface_get_device_address(e.iface(),
                                         (uct_device_addr_t*)dev_addr.data()));

    params.iface_addr  = (uct_iface_addr_t*)iface_addr.data();
    params.device_addr = (uct_device_addr_t*)dev_addr.data();
    params.field_mask  = UCT_EP_IS_CONNECTED_FIELD_DEVICE_ADDR |
                         UCT_EP_IS_CONNECTED_FIELD_IFACE_ADDR;

    if (iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        ep_addr.resize(iface_attr.ep_addr_len);
        auto addr_buf = (uct_ep_addr_t*)ep_addr.data();
        ASSERT_UCS_OK(uct_ep_get_address(e.ep(0), addr_buf));
        params.ep_addr     = (uct_ep_addr_t*)ep_addr.data();
        params.field_mask |= UCT_EP_IS_CONNECTED_FIELD_EP_ADDR;
    }

    return uct_ep_is_connected(m_sender->ep(0), &params);
}

void test_uct_ep::modify_remote_id(entity &e) const
{
    uct_iface_attr_t iface_attr;

    ASSERT_UCS_OK(uct_iface_query(e.iface(), &iface_attr));

    if (has_transport("knem")) {
        ASSERT_EQ(iface_attr.device_addr_len, sizeof(uint64_t));
        auto addr_hook = [](uct_iface_h iface, uct_device_addr_t *addr) {
            *(uint64_t*)addr = ucs_rand();
            return UCS_OK;
        };
        e.iface()->ops.iface_get_device_address = addr_hook;
    } else {
        ASSERT_GE(iface_attr.iface_addr_len, sizeof(pid_t));
        auto addr_hook = [](uct_iface_h iface, uct_iface_addr_t *addr) {
            *(pid_t*)addr = getppid();
            return UCS_OK;
        };
        e.iface()->ops.iface_get_address = addr_hook;
    }
}

UCS_TEST_SKIP_COND_P(test_uct_ep, disconnect_after_send,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY)) {
    ucs_status_t status;

    create_sender();

    mapped_buffer buffer(256, 0, *m_sender);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buffer.ptr(),
                            (ucs_min(buffer.length(),
                                     m_sender->iface_attr().cap.am.max_zcopy)),
                            buffer.memh(),
                            m_sender->iface_attr().cap.am.max_iov);

    unsigned max_iter = 300 / ucs::test_time_multiplier();

    /* FIXME: need to investigate RC/VERBS hang after ~200 iterations, when a
     * sender entity is created after a receiver one */
    unsigned max_retry_iter = has_transport("rc_verbs") ? 1 : max_iter;
    for (unsigned i = 0; i < max_retry_iter; ++i) {
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

UCS_TEST_SKIP_COND_P(test_uct_ep, is_connected, has_transport("gga_mlx5"))
{
    create_sender();

    auto e = create_entity(0);
    m_entities.push_back(e);

    /* The following transports are always connected to remote side on the same
     * process/machine, so we modify remote id to simulate no-connection
     * scenario. */
    if (has_cma() || has_cuda_ipc() || has_transport("knem")) {
        modify_remote_id(has_cuda_ipc() ? *m_receiver : *e);
    }

    connect();
    e->connect(0, *m_receiver, 1);
    flush();

    EXPECT_TRUE(is_connected_to_sender(*m_receiver));
    EXPECT_FALSE(is_connected_to_sender(*e));
}

UCS_TEST_SKIP_COND_P(test_uct_ep, destroy_entity_after_send,
                     !check_caps(UCT_IFACE_FLAG_AM_ZCOPY))
{
    const unsigned max_iter = 300 / ucs::test_time_multiplier();

    for (unsigned i = 0; i < max_iter; ++i) {
        create_sender();
        connect();

        const uct_iface_attr &iface_attr = m_sender->iface_attr();
        const size_t msg_length          = 256 * UCS_KBYTE;
        ucs::auto_ptr<mapped_buffer> buffer(
                new mapped_buffer(msg_length, 0, *m_sender));

        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buffer->ptr(),
                                ucs_min(buffer->length(),
                                        iface_attr.cap.am.max_zcopy),
                                buffer->memh(), iface_attr.cap.am.max_iov);

        test_ep_comp_t comp;

        comp.comp.status = UCS_OK;
        comp.comp.count  = 0;
        comp.comp.func   = completion_cb;
        comp.test        = this;
        comp.ep          = m_sender->ep(0);

        for (unsigned count = 0; count < max_iter;) {
            ucs_status_t status = uct_ep_am_zcopy(m_sender->ep(0), 1, NULL, 0,
                                                  iov, iovcnt, 0, &comp.comp);
            handle_status(status, comp);
            if (status == UCS_ERR_NO_RESOURCE) {
                if (count == 0) {
                    progress();
                } else {
                    status = uct_ep_flush(m_sender->ep(0), UCT_FLUSH_FLAG_LOCAL,
                                          &comp.comp);
                    handle_status(status, comp);
                    break;
                }
            }
            ++count;
        }

        if (comp.comp.count != 0) {
            scoped_log_handler slh(detect_uncomp_op_logger);
            /* Destroy EP without flushing prior to not complete AM Zcopy and
             * flush operations during progress() */
            disconnect(false);
            /* All outstanding operations must be completed in EP destroy */
            EXPECT_EQ(0, comp.comp.count);
        }

        /* Mapped buffer has to be released before destroying a sender entity */
        buffer.reset();

        m_entities.remove(m_sender);
    }
}

UCT_INSTANTIATE_TEST_CASE(test_uct_ep)
UCT_INSTANTIATE_CUDA_IPC_TEST_CASE(test_uct_ep)
