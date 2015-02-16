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

    static ucs_status_t get_bcopy_cb(void *desc, void *data, size_t length,
                                     void *arg)
    {
        bcopy_ctx *ctx = reinterpret_cast<bcopy_ctx*>(arg);

        memcpy(ctx->buf->ptr(), data, length);
        ++ctx->self->m_completion_count;
        delete ctx;
        return UCS_OK;
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
        return uct_ep_put_zcopy(ep,
                                sendbuf.ptr(), sendbuf.length(), sendbuf.lkey(),
                                recvbuf.addr(), recvbuf.rkey(),
                                &m_completion->uct);
    }

    ucs_status_t get_bcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        bcopy_ctx *ctx = new bcopy_ctx();
        ctx->self = this;
        ctx->buf  = &sendbuf;
        return uct_ep_get_bcopy(ep, sendbuf.length(), recvbuf.addr(),
                                recvbuf.rkey(), get_bcopy_cb,
                                reinterpret_cast<void*>(ctx));
    }

    ucs_status_t get_zcopy(uct_ep_h ep, const mapped_buffer &sendbuf,
                           const mapped_buffer &recvbuf)
    {
        return uct_ep_get_zcopy(ep,
                                sendbuf.ptr(), sendbuf.length(), sendbuf.lkey(),
                                recvbuf.addr(), recvbuf.rkey(),
                                &m_completion->uct);
    }

    virtual void test_xfer(send_func_t send, size_t length, direction_t direction) {
        mapped_buffer sendbuf(length, 1, SEED1, sender());
        mapped_buffer recvbuf(length, 1, SEED2, receiver());

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

    static void log_handler(const char *file, unsigned line, const char *function,
                            unsigned level, const char *prefix, const char *message,
                            va_list ap)
    {
        char buf[200] = {0};
        if (level <= UCS_LOG_LEVEL_WARN) {
            va_list ap_copy;
            va_copy(ap_copy, ap); /* Create a copy of arglist, to use it 2nd time */

            ucs_log_default_handler(file, line, function, UCS_LOG_LEVEL_DEBUG, prefix, message, ap);
            vsnprintf(buf, sizeof(buf), message, ap_copy);
            va_end(ap_copy);

            UCS_TEST_MESSAGE << "   < " << buf << " >";
            ++error_count;
        } else {
            ucs_log_default_handler(file, line, function, level, prefix, message, ap);
        }
    }

    void test_error_zcopy(void *buffer, size_t length, uct_lkey_t lkey,
                          uint64_t remote_addr, uct_rkey_t rkey)
    {
        error_count = 0;

        ucs_log_set_handler(log_handler);
        ucs_status_t status;
        do {
            status = uct_ep_put_zcopy(sender_ep(), buffer, length, lkey,
                                      remote_addr, rkey, NULL);
        } while (status == UCS_ERR_WOULD_BLOCK);
        wait_for_remote();

        /* Flush async events */
        ucs::safe_usleep(1e4);
        ucs_log_set_handler(ucs_log_default_handler);

        EXPECT_GT(error_count, 0u);
    }

    static unsigned error_count;

private:
    struct bcopy_ctx {
        uct_p2p_rma_test     *self;
        const mapped_buffer  *buf;
    };
};

unsigned uct_p2p_rma_test::error_count = 0;


UCS_TEST_P(uct_p2p_rma_test, local_access_error) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF);
    mapped_buffer sendbuf(16, 1, SEED1, sender());
    mapped_buffer recvbuf(16, 1, SEED2, receiver());

    test_error_zcopy(sendbuf.ptr(), sendbuf.length() + 4, sendbuf.lkey(),
                                  recvbuf.addr(), recvbuf.rkey());

    recvbuf.pattern_check(SEED2);
}

UCS_TEST_P(uct_p2p_rma_test, remote_access_error) {
    check_caps(UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF);
    mapped_buffer sendbuf(16, 1, SEED1, sender());
    mapped_buffer recvbuf(16, 1, SEED2, receiver());

    test_error_zcopy(sendbuf.ptr(), sendbuf.length(), sendbuf.lkey(),
                                  recvbuf.addr() + 4, recvbuf.rkey());

    recvbuf.pattern_check(SEED2);
}

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
