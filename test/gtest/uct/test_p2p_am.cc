/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_p2p_test.h"

#include <string>
#include <vector>

class uct_p2p_am_test : public uct_p2p_test
{
public:
    static const uint8_t AM_ID = 11;
    static const uint64_t SEED1 = 0xa1a1a1a1a1a1a1a1ul;
    static const uint64_t SEED2 = 0xa2a2a2a2a2a2a2a2ul;
    static const uint64_t MAGIC_DESC  = 0xdeadbeef12345678ul;
    static const uint64_t MAGIC_ALLOC = 0xbaadf00d12345678ul;

    typedef struct {
        uint64_t magic;
        unsigned length;
        /* data follows */
    } receive_desc_t;

    typedef struct {
        unsigned        count;
    } tracer_ctx_t;

    uct_p2p_am_test() :
        uct_p2p_test(sizeof(receive_desc_t)),
        m_am_count(0),
        m_keep_data(false)
    {
        m_send_tracer.count = 0;
        m_recv_tracer.count = 0;
        pthread_mutex_init(&m_lock, NULL);
    }

    virtual void init() {
        uct_p2p_test::init();
        m_am_count = 0;
        uct_iface_set_am_tracer(sender().iface(),   am_tracer, &m_send_tracer);
        if (&sender() != &receiver()) {
            uct_iface_set_am_tracer(receiver().iface(), am_tracer, &m_recv_tracer);
        }
    }

    virtual void cleanup() {
        uct_iface_set_am_tracer(receiver().iface(), NULL, NULL);
        uct_iface_set_am_tracer(sender().iface(), NULL, NULL);
        uct_p2p_test::cleanup();
    }

    static ucs_status_t am_handler(void *arg, void *data, size_t length,
                                   unsigned flags) {
        uct_p2p_am_test *self = reinterpret_cast<uct_p2p_am_test*>(arg);
        return self->am_handler(data, length, flags);
    }

    static void am_tracer(void *arg, uct_am_trace_type_t type, uint8_t id,
                          const void *data, size_t length, char *buffer,
                          size_t max)
    {
        tracer_ctx_t *ctx = (tracer_ctx_t *)arg;

        EXPECT_EQ(uint8_t(AM_ID), id);
        mapped_buffer::pattern_check(data, length, SEED1);
        *buffer = '\0';
        ++ctx->count;
    }

    ucs_status_t am_handler(void *data, size_t length, unsigned flags) {
        pthread_mutex_lock(&m_lock);
        ++m_am_count;
        pthread_mutex_unlock(&m_lock);

        if (m_keep_data) {
            receive_desc_t *my_desc;
            if (flags & UCT_CB_PARAM_FLAG_DESC) {
                my_desc = (receive_desc_t *)data - 1;
                my_desc->magic  = MAGIC_DESC;
            } else {
                my_desc = (receive_desc_t *)ucs_malloc(sizeof(*my_desc) + length,
                                                       "TODO: remove allocation");
                memcpy(my_desc + 1, data, length);
                my_desc->magic  = MAGIC_ALLOC;
            }
            my_desc->length = length;
            pthread_mutex_lock(&m_lock);
            m_backlog.push_back(my_desc);
            pthread_mutex_unlock(&m_lock);
            return (my_desc->magic == MAGIC_DESC) ? UCS_INPROGRESS : UCS_OK;
        }
        mapped_buffer::pattern_check(data, length, SEED1);
        return UCS_OK;
    }

    void check_backlog() {
        pthread_mutex_lock(&m_lock);
        while (!m_backlog.empty()) {
            receive_desc_t *my_desc = m_backlog.back();
            m_backlog.pop_back();
            mapped_buffer::pattern_check(my_desc + 1, my_desc->length, SEED1);
            pthread_mutex_unlock(&m_lock);
            if (my_desc->magic == MAGIC_DESC) {
                uct_iface_release_desc(my_desc);
            } else {
                EXPECT_EQ(uint64_t(MAGIC_ALLOC), my_desc->magic);
                ucs_free(my_desc);
            }
            pthread_mutex_lock(&m_lock);
        }
        pthread_mutex_unlock(&m_lock);
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
        ssize_t packed_len;
        packed_len = uct_ep_am_bcopy(ep, AM_ID, mapped_buffer::pack,
                                     (void*)&sendbuf, 0);
        if (packed_len >= 0) {
            EXPECT_EQ(sendbuf.length(), (size_t)packed_len);
            return UCS_OK;
        } else {
            return (ucs_status_t)packed_len;
        }
    }

    ucs_status_t am_zcopy(uct_ep_h ep, const mapped_buffer& sendbuf,
                          const mapped_buffer& recvbuf)
    {
        size_t max_hdr  = ucs_min(sender().iface_attr().cap.am.max_hdr,
                                  sendbuf.length());
        size_t hdr_size = ucs::rand() % (max_hdr + 1);

        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, ((char*)sendbuf.ptr() + hdr_size),
                                (sendbuf.length() - hdr_size), sendbuf.memh(),
                                sender().iface_attr().cap.am.max_iov);

        return uct_ep_am_zcopy(ep,
                               AM_ID,
                               sendbuf.ptr(),
                               hdr_size,
                               iov,
                               iovcnt,
                               0,
                               comp());
    }

    void test_xfer_do(send_func_t send, size_t length, unsigned flags,
                      uint32_t am_mode, uct_memory_type_t mem_type)
    {
        ucs_status_t status;

        m_am_count = 0;
        m_send_tracer.count = 0;
        m_recv_tracer.count = 0;

        status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler,
                                          this, am_mode);
        ASSERT_UCS_OK(status);

        mapped_buffer sendbuf(length, SEED1, sender(), 0, mem_type);
        mapped_buffer recvbuf(0, 0, sender(), 0, mem_type); /* dummy */

        blocking_send(send, sender_ep(), sendbuf, recvbuf, true);
        sendbuf.pattern_fill(SEED2);

        while (m_am_count == 0) {
            short_progress_loop();
        }

        if (!(receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_DUP)) {
            flush();
            EXPECT_EQ(1u, m_am_count);
        } else {
            EXPECT_GE(m_am_count, 1u);
        }

        status = uct_iface_set_am_handler(receiver().iface(), AM_ID, NULL, NULL,
                                          am_mode);
        ASSERT_UCS_OK(status);

        check_backlog();

        if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) {
            if (&sender() == &receiver()) {
                EXPECT_UD_CHECK(2u, m_send_tracer.count, LE, EQ);
            } else {
                EXPECT_UD_CHECK(1u, m_send_tracer.count, LE, EQ);
                EXPECT_UD_CHECK(1u, m_recv_tracer.count, LE, EQ);
            }
        }
    }

    virtual void test_xfer(send_func_t send, size_t length, unsigned flags,
                           uct_memory_type_t mem_type) {

        if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_CB_SYNC) {
            test_xfer_do(send, length, flags, UCT_CB_FLAG_SYNC, mem_type);
        }
        if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_CB_ASYNC) {
            test_xfer_do(send, length, flags, UCT_CB_FLAG_ASYNC, mem_type);
        }
    }

    void set_keep_data(bool keep) {
        m_keep_data = keep;
    }

    void am_sync_finish(unsigned prev_am_count) {
        /* am message handler must be only invoked from progress */
        twait(500);
        EXPECT_EQ(prev_am_count, m_am_count);
        wait_for_value(&m_am_count, prev_am_count + 1, true);
        EXPECT_EQ(prev_am_count+1, m_am_count);
    }

    void am_async_finish(unsigned prev_am_count) {
        /* am message handler must be only invoked within reasonable time if
         * progress is not called */
        wait_for_value(&m_am_count, prev_am_count + 1, false);
        EXPECT_EQ(prev_am_count+1, m_am_count);
    }

protected:
    inline size_t backlog_size() const {
        return m_backlog.size();
    }

protected:
    unsigned                     m_am_count;
private:
    bool                         m_keep_data;
    std::vector<receive_desc_t*> m_backlog;
    pthread_mutex_t              m_lock;
    tracer_ctx_t                 m_send_tracer;
    tracer_ctx_t                 m_recv_tracer;
};

UCS_TEST_P(uct_p2p_am_test, am_sync) {

    ucs_status_t status;

    if (UCT_DEVICE_TYPE_SELF == GetParam()->dev_type) {
        UCS_TEST_SKIP_R("SELF doesn't use progress");
    }

    check_caps(UCT_IFACE_FLAG_CB_SYNC, UCT_IFACE_FLAG_AM_DUP);

    mapped_buffer recvbuf(0, 0, sender()); /* dummy */
    unsigned am_count = m_am_count = 0;

    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler,
                                      this, UCT_CB_FLAG_SYNC);
    ASSERT_UCS_OK(status);

    if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        mapped_buffer sendbuf_short(sender().iface_attr().cap.am.max_short,
                                    SEED1, sender());
        am_count = m_am_count;
        blocking_send(static_cast<send_func_t>(&uct_p2p_am_test::am_short),
                      sender_ep(), sendbuf_short, recvbuf, false);
        am_sync_finish(am_count);
    }

    if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
        mapped_buffer sendbuf_bcopy(sender().iface_attr().cap.am.max_bcopy,
                                    SEED1, sender());
        am_count = m_am_count;
        blocking_send(static_cast<send_func_t>(&uct_p2p_am_test::am_bcopy),
                      sender_ep(), sendbuf_bcopy, recvbuf, false);
        am_sync_finish(am_count);
    }

    if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
        mapped_buffer sendbuf_zcopy(sender().iface_attr().cap.am.max_zcopy,
                                    SEED1, sender());
        am_count = m_am_count;
        blocking_send(static_cast<send_func_t>(&uct_p2p_am_test::am_zcopy),
                      sender_ep(), sendbuf_zcopy, recvbuf, false);
        am_sync_finish(am_count);
    }

    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, NULL, NULL, 0);
    ASSERT_UCS_OK(status);
}

UCS_TEST_P(uct_p2p_am_test, am_async) {
    ucs_status_t status;

    check_caps(UCT_IFACE_FLAG_CB_ASYNC, UCT_IFACE_FLAG_AM_DUP);

    mapped_buffer recvbuf(0, 0, sender()); /* dummy */
    unsigned am_count = m_am_count = 0;

    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler,
                                      this, UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);

    if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        mapped_buffer sendbuf_short(sender().iface_attr().cap.am.max_short,
                                    SEED1, sender());
        am_count = m_am_count;
        blocking_send(static_cast<send_func_t>(&uct_p2p_am_test::am_short),
                      sender_ep(), sendbuf_short, recvbuf, false);
        am_async_finish(am_count);
    }

    if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
        mapped_buffer sendbuf_bcopy(sender().iface_attr().cap.am.max_bcopy,
                                    SEED1, sender());
        am_count = m_am_count;
        blocking_send(static_cast<send_func_t>(&uct_p2p_am_test::am_bcopy),
                      sender_ep(), sendbuf_bcopy, recvbuf, false);
        am_async_finish(am_count);
    }

    if (receiver().iface_attr().cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) {
        mapped_buffer sendbuf_zcopy(sender().iface_attr().cap.am.max_zcopy,
                                    SEED1, sender());
        am_count = m_am_count;
        blocking_send(static_cast<send_func_t>(&uct_p2p_am_test::am_zcopy),
                      sender_ep(), sendbuf_zcopy, recvbuf, false);
        am_async_finish(am_count);
    }

    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, NULL, NULL,
                                      UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);
}

class uct_p2p_am_misc : public uct_p2p_am_test
{
public:
    static const unsigned RX_MAX_BUFS;
    static const unsigned RX_QUEUE_LEN;

    template<typename T>
    std::string to_string(T v) {
        std::stringstream ss;
        ss << v;
        return ss.str();
    }

    uct_p2p_am_misc() :
        uct_p2p_am_test() {
        ucs_status_t status_ib_bufs, status_ib_qlen, status_bufs;
        m_rx_buf_limit_failed = 0;
        status_ib_bufs = uct_config_modify(m_iface_config, "IB_RX_MAX_BUFS" , to_string(RX_MAX_BUFS).c_str());
        status_ib_qlen = uct_config_modify(m_iface_config, "IB_RX_QUEUE_LEN", to_string(RX_QUEUE_LEN).c_str());
        status_bufs    = uct_config_modify(m_iface_config, "RX_MAX_BUFS"    , to_string(RX_MAX_BUFS).c_str());
        if ((status_ib_bufs != UCS_OK) && (status_ib_qlen != UCS_OK) &&
            (status_bufs != UCS_OK)) {
            /* none of the above environment parameters were set successfully
             * (for UCTs that don't have them) */
            m_rx_buf_limit_failed = 1;
        }

    }

    ucs_status_t send_with_timeout(uct_ep_h ep, const mapped_buffer& sendbuf,
                                   const mapped_buffer& recvbuf, double timeout)
    {
        ucs_time_t loop_end_limit;
        ucs_status_t status = UCS_ERR_NO_RESOURCE;

        loop_end_limit = ucs_get_time() + ucs_time_from_sec(timeout);

        while ((ucs_get_time() < loop_end_limit) && (status != UCS_OK) &&
               (backlog_size() < 1000000)) {
            status = am_short(sender_ep(), sendbuf, recvbuf);
            progress();
        }
        return status;
    }

    static ucs_log_func_rc_t
    no_rx_buffs_log_handler(const char *file, unsigned line, const char *function,
                            ucs_log_level_t level, const char *message, va_list ap)
    {
        /* Ignore warnings about empty memory pool */
        if ((level == UCS_LOG_LEVEL_WARN) &&
            !strcmp(function, UCS_PP_QUOTE(uct_iface_mpool_empty_warn)))
        {
            UCS_TEST_MESSAGE << file << ":" << line << ": "
                             << format_message(message, ap);
            return UCS_LOG_FUNC_RC_STOP;
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    unsigned m_rx_buf_limit_failed;
};


UCS_TEST_P(uct_p2p_am_test, am_short) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_short),
                    sizeof(uint64_t),
                    sender().iface_attr().cap.am.max_short,
                    TEST_UCT_FLAG_DIR_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_bcopy) {
    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_bcopy),
                    0ul,
                    sender().iface_attr().cap.am.max_bcopy,
                    TEST_UCT_FLAG_DIR_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_short_keep_data) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT, UCT_IFACE_FLAG_AM_DUP);
    set_keep_data(true);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_short),
                    sizeof(uint64_t),
                    sender().iface_attr().cap.am.max_short,
                    TEST_UCT_FLAG_DIR_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_bcopy_keep_data) {
    check_caps(UCT_IFACE_FLAG_AM_BCOPY);
    set_keep_data(true);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_bcopy),
                    sizeof(uint64_t),
                    sender().iface_attr().cap.am.max_bcopy,
                    TEST_UCT_FLAG_DIR_SEND_TO_RECV);
}

UCS_TEST_P(uct_p2p_am_test, am_zcopy) {
    check_caps(UCT_IFACE_FLAG_AM_ZCOPY);
    test_xfer_multi(static_cast<send_func_t>(&uct_p2p_am_test::am_zcopy),
                    0ul,
                    sender().iface_attr().cap.am.max_zcopy,
                    TEST_UCT_FLAG_DIR_SEND_TO_RECV);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_am_test)

const unsigned uct_p2p_am_misc::RX_MAX_BUFS = 1024; /* due to hard coded 'grow'
                                                       parameter in uct_ib_iface_recv_mpool_init */
const unsigned uct_p2p_am_misc::RX_QUEUE_LEN = 64;

UCS_TEST_P(uct_p2p_am_misc, no_rx_buffs) {

    mapped_buffer sendbuf(10 * sizeof(uint64_t), SEED1, sender());
    mapped_buffer recvbuf(0, 0, sender()); /* dummy */
    ucs_status_t status;

    if (RUNNING_ON_VALGRIND) {
        UCS_TEST_SKIP_R("skipping on valgrind");
    }

    if (&sender() == &receiver()) {
        UCS_TEST_SKIP_R("skipping on loopback");
    }

    if (m_rx_buf_limit_failed) {
        UCS_TEST_SKIP_R("Current transport doesn't have rx memory pool");
    }

    check_caps(UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_CB_SYNC);

    /* set a callback for the uct to invoke for receiving the data */
    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler,
                                      (void*)this, UCT_CB_FLAG_SYNC);
    ASSERT_UCS_OK(status);

    /* send many messages and progress the receiver. the receiver will keep getting
     * UCS_INPROGRESS from the recv-handler and will keep consuming its rx memory pool.
     * the goal is to make the receiver's rx memory pool run out.
     * once this happens, the sender shouldn't be able to send */
    ucs_log_push_handler(no_rx_buffs_log_handler);
    set_keep_data(true);
    status = send_with_timeout(sender_ep(), sendbuf, recvbuf, 1);
    while (status != UCS_ERR_NO_RESOURCE) {
        ASSERT_UCS_OK(status);
        status = send_with_timeout(sender_ep(), sendbuf, recvbuf, 3);
    }
    set_keep_data(false);
    check_backlog();
    short_progress_loop();
    ucs_log_pop_handler();

    /* check that now the sender is able to send */
    EXPECT_EQ(UCS_OK, send_with_timeout(sender_ep(), sendbuf, recvbuf, 6));
}

UCS_TEST_P(uct_p2p_am_misc, am_max_short_multi) {
    check_caps(UCT_IFACE_FLAG_AM_SHORT);

    ucs_status_t status;

    m_am_count = 0;
    set_keep_data(false);

    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler,
                                      this, UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);

    size_t size = ucs_min(sender().iface_attr().cap.am.max_short, 8192ul);
    std::string sendbuf(size, 0);
    mapped_buffer::pattern_fill(&sendbuf[0], sendbuf.size(), SEED1);
    ucs_assert(SEED1 == *(uint64_t*)&sendbuf[0]);

    /* exhaust all resources or time out 1sec */
    ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(1.0);
    do {
        status = uct_ep_am_short(sender_ep(), AM_ID, SEED1,
                                 ((uint64_t*)&sendbuf[0]) + 1,
                                 sendbuf.size() - sizeof(uint64_t));
    } while ((ucs_get_time() < loop_end_limit) && (status == UCS_OK));
    if (status != UCS_ERR_NO_RESOURCE) {
        ASSERT_UCS_OK(status);
    }

    /* do some progress */
    short_progress_loop(50);

    /* should be able to send again */
    status = uct_ep_am_short(sender_ep(), AM_ID, SEED1, NULL, 0);
    EXPECT_EQ(UCS_OK, status);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_am_misc)

class uct_p2p_am_tx_bufs : public uct_p2p_am_test
{
public:
    uct_p2p_am_tx_bufs() : uct_p2p_am_test() {
        ucs_status_t status1, status2;

        /* can not reduce mpool size below retransmission window
         * for ud
         */
        if ((GetParam()->tl_name.compare("ud") == 0) || 
            (GetParam()->tl_name.compare("ud_mlx5") == 0)) { 
            m_inited = false;
            return;
        }

        status1 = uct_config_modify(m_iface_config, "IB_TX_MAX_BUFS" , "32");
        status2 = uct_config_modify(m_iface_config, "IB_TX_BUFS_GROW" , "32");
        if ((status1 != UCS_OK) || (status2 != UCS_OK)) {
            m_inited = false;
        } else {
            m_inited = true;
        }
    }
    bool m_inited;
};

UCS_TEST_P(uct_p2p_am_tx_bufs, am_tx_max_bufs) {
    ucs_status_t status;
    mapped_buffer recvbuf(0, 0, sender()); /* dummy */
    mapped_buffer sendbuf_bcopy(sender().iface_attr().cap.am.max_bcopy,
            SEED1, sender());

    status = uct_iface_set_am_handler(receiver().iface(), AM_ID, am_handler,
                                      this, UCT_CB_FLAG_ASYNC);
    ASSERT_UCS_OK(status);
    /* skip on cm, ud */
    if (!m_inited) { 
        UCS_TEST_SKIP_R("Test does not apply to the current transport");
    }
    if (GetParam()->tl_name.compare("cm") == 0) { 
        UCS_TEST_SKIP_R("Test does not work with IB CM transport");
    }
    if ((GetParam()->tl_name.compare("rc") == 0) ||
        (GetParam()->tl_name.compare("rc_mlx5") == 0)) { 
        UCS_TEST_SKIP_R("Test does not work with IB RC transports");
    }
    do {
        status = am_bcopy(sender_ep(), sendbuf_bcopy, recvbuf);
        if (status == UCS_OK) {
        }
    } while (status == UCS_OK);

    /* short progress shall release tx buffers and 
     * the next message shall go out */
    ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(1.0);
    do {
        progress();
        status = am_bcopy(sender_ep(), sendbuf_bcopy, recvbuf);
        if (status == UCS_OK) {
            break;
        }
    } while (ucs_get_time() < loop_end_limit);

    EXPECT_EQ(UCS_OK, status);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_am_tx_bufs)
